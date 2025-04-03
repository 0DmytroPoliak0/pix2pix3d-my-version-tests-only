import sys
sys.path.append('.')

import subprocess
import os
from pathlib import Path

import numpy as np
import PIL.Image
import cv2
import pytest
from skimage.metrics import structural_similarity as ssim

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def compute_iou(binary1: np.ndarray, binary2: np.ndarray) -> float:
    """Compute Intersection over Union (IoU) between two binary images."""
    intersection = np.logical_and(binary1, binary2).sum()
    union = np.logical_or(binary1, binary2).sum()
    return intersection / union if union > 0 else 1.0

def compute_composite_score(iou_value: float, ssim_value: float) -> float:
    """Compute an overall quality score on a 0-100 scale (equal weight for IoU and SSIM)."""
    return 0.5 * (iou_value * 100) + 0.5 * (ssim_value * 100)

def compute_consistency_score(scores: np.ndarray) -> float:
    """
    Compute a consistency score based on the relative standard deviation of quality scores.
    A lower relative standard deviation yields a higher consistency score.
    We define it as: 100 - (std/mean * 100).
    """
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    relative_std = (std_score / mean_score * 100) if mean_score > 0 else 0
    return max(0, 100 - relative_std)

def run_generation(cfg: str, input_id: str, seed: int, outdir: str, network: str):
    """
    Run the generation pipeline using the provided configuration.
    Returns the CompletedProcess from subprocess.
    """
    cmd = [
        "python", "applications/generate_samples.py",
        "--network", network,
        "--outdir", outdir,
        "--random_seed", str(seed),
        "--input_id", input_id,
        "--cfg", cfg
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result

# -----------------------------------------------------------------------------
# Musa Advanced Integration Test with Usage Profiles
# -----------------------------------------------------------------------------
@pytest.mark.integration
def test_musa_advanced_score(tmp_path):
    """
    Musa Advanced Test with Usage Profiles:
      - Simulate three usage profiles:
          * seg2face (usage weight: 50%)
          * seg2cat  (usage weight: 30%)
          * edge2car (usage weight: 20%)
      - For each profile, run the generation pipeline over seeds [42, 43, 44, 45, 46].
      - For each run, compute IoU, SSIM, and a composite quality score.
      - Compute the average quality and a consistency score (based on variability) for each configuration.
      - Compute an overall Musa score per configuration as:
            Overall Score = 0.7 * (Average Quality) + 0.3 * (Consistency Score)
      - Finally, compute a weighted overall Musa Score using the usage weights.
      - Expected: The weighted overall Musa Score should be above a defined threshold (e.g., >65/100).
    
    Test Steps:
      1. For each configuration, create a temporary output folder.
      2. Run the generation pipeline with the specified seeds.
      3. For each run, load generated segmentation and color images, compute IoU and SSIM.
      4. Compute the composite quality score for each run.
      5. Calculate the average quality and consistency for each configuration.
      6. Compute overall Musa scores and a weighted overall score.
      7. Assert that the weighted overall Musa Score exceeds the threshold.
    """
    # Define usage weights for each profile.
    usage_weights = {
        "seg2face": 0.50,
        "seg2cat": 0.30,
        "edge2car": 0.20
    }
    # Define network checkpoint and input id for each configuration.
    networks = {
        "seg2face": "checkpoints/pix2pix3d_seg2face.pkl",
        "seg2cat": "checkpoints/pix2pix3d_seg2cat.pkl",
        "edge2car": "checkpoints/pix2pix3d_edge2car.pkl"
    }
    input_ids = {
        "seg2face": "100",
        "seg2cat": "1666",
        "edge2car": "0"
    }
    seeds = [42, 43, 44, 45, 46]
    config_list = ["seg2face", "seg2cat", "edge2car"]
    config_scores = {}
    
    for cfg in config_list:
        outdir_cfg = tmp_path / f"{cfg}_output"
        outdir_cfg.mkdir()
        quality_scores = []
        for seed in seeds:
            result = run_generation(cfg, input_ids[cfg], seed, str(outdir_cfg), networks[cfg])
            assert result.returncode == 0, f"{cfg} generation failed for seed {seed}: {result.stderr}"
            
            # Load generated segmentation (label) image and input segmentation image.
            gen_label_path = outdir_cfg / f"{cfg}_{input_ids[cfg]}_{seed}_label.png"
            input_label_path = outdir_cfg / f"{cfg}_{input_ids[cfg]}_input.png"
            assert gen_label_path.exists(), f"{cfg} generated segmentation not found for seed {seed}"
            assert input_label_path.exists(), f"{cfg} input segmentation not found for seed {seed}"
            gen_label = np.array(PIL.Image.open(gen_label_path).convert("L"))
            input_label = np.array(PIL.Image.open(input_label_path).convert("L"))
            gen_bin = (gen_label > 0).astype(np.uint8)
            input_bin = (input_label > 0).astype(np.uint8)
            iou_val = compute_iou(gen_bin, input_bin)
            
            # Load generated color image and compute SSIM.
            gen_color_path = outdir_cfg / f"{cfg}_{input_ids[cfg]}_{seed}_color.png"
            assert gen_color_path.exists(), f"{cfg} generated color image not found for seed {seed}"
            gen_color = np.array(PIL.Image.open(gen_color_path).convert("RGB"))
            gen_gray = cv2.cvtColor(gen_color, cv2.COLOR_RGB2GRAY)
            ssim_val = ssim(gen_gray, input_label)
            
            qs = compute_composite_score(iou_val, ssim_val)
            quality_scores.append(qs)
            print(f"[{cfg}] Seed {seed}: IoU={iou_val:.2f}, SSIM={ssim_val:.2f}, Quality Score={qs:.1f}/100")
        
        avg_quality = np.mean(quality_scores)
        consistency = compute_consistency_score(np.array(quality_scores))
        overall_cfg_score = 0.7 * avg_quality + 0.3 * consistency
        config_scores[cfg] = overall_cfg_score
        print(f"[{cfg}] Average Quality Score: {avg_quality:.1f}/100")
        print(f"[{cfg}] Consistency Score: {consistency:.1f}/100")
        print(f"[{cfg}] Overall {cfg} Musa Score: {overall_cfg_score:.1f}/100")
    
    overall_musa_score = sum(usage_weights[cfg] * config_scores[cfg] for cfg in config_list)
    print(f"Weighted Overall Musa Score: {overall_musa_score:.1f}/100")
    
    # Assert that the weighted overall Musa Score meets the threshold.
    assert overall_musa_score > 65, f"Overall Musa Score too low: {overall_musa_score:.1f}/100"