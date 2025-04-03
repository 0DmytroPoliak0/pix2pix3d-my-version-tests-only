import sys
sys.path.append('.')

import subprocess
import os
from pathlib import Path
import time

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

def compute_composite_quality(iou_value: float, ssim_value: float) -> float:
    """Compute an overall quality score on a 0-100 scale (equal weight for IoU and SSIM)."""
    return 0.5 * (iou_value * 100) + 0.5 * (ssim_value * 100)

def compute_consistency_score(scores: np.ndarray) -> float:
    """
    Compute a consistency score based on the relative standard deviation of quality scores.
    Defined as: 100 - (std/mean * 100).
    """
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    relative_std = (std_score / mean_score * 100) if mean_score > 0 else 0
    return max(0, 100 - relative_std)

def compute_performance_score(elapsed_time: float, ideal_time: float = 9.0) -> float:
    """
    Compute a performance score on a 0-100 scale.
    If the elapsed time is at or below ideal_time, score is 100.
    Otherwise, subtract penalty points proportional to the excess time.
    For example, for each second above ideal_time, subtract 10 points.
    """
    penalty = max(0, (elapsed_time - ideal_time)) * 10
    score = max(0, 100 - penalty)
    return score

def run_generation(cfg: str, input_id: str, seed: int, outdir: str, network: str):
    """
    Run the generation pipeline using the provided configuration.
    Returns a tuple: (subprocess.CompletedProcess, elapsed_time)
    """
    cmd = [
        "python", "applications/generate_samples.py",
        "--network", network,
        "--outdir", outdir,
        "--random_seed", str(seed),
        "--input_id", input_id,
        "--cfg", cfg
    ]
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start
    return result, elapsed

# -----------------------------------------------------------------------------
# Musa Advanced Integration Test with Performance Metrics
# -----------------------------------------------------------------------------
@pytest.mark.integration
def test_musa_advanced_with_performance(tmp_path):
    """
    Musa Advanced Test with Usage Profiles and Performance:
      - Simulate three usage profiles: seg2face (50%), seg2cat (30%), edge2car (20%).
      - For each profile, run the generation pipeline for seeds [42, 43, 44, 45, 46],
        and measure quality metrics (IoU, SSIM, composite quality) and performance (execution time).
      - For each run, compute a performance score (ideal if <=9 seconds).
      - For each configuration, compute:
            * Average quality score.
            * Consistency score from quality variability.
            * Average performance score.
            * Overall configuration score = 0.7 * (quality-based score) + 0.3 * (performance score).
      - Finally, compute a weighted overall Musa score using usage weights.
      - Assert that the weighted overall Musa score exceeds a defined threshold (e.g., >65/100).
    """
    usage_weights = {"seg2face": 0.50, "seg2cat": 0.30, "edge2car": 0.20}
    networks = {
        "seg2face": "checkpoints/pix2pix3d_seg2face.pkl",
        "seg2cat": "checkpoints/pix2pix3d_seg2cat.pkl",
        "edge2car": "checkpoints/pix2pix3d_edge2car.pkl"
    }
    input_ids = {"seg2face": "100", "seg2cat": "1666", "edge2car": "0"}
    seeds = [42, 43, 44, 45, 46]
    config_list = ["seg2face", "seg2cat", "edge2car"]
    config_scores = {}
    
    for cfg in config_list:
        outdir_cfg = tmp_path / f"{cfg}_output"
        outdir_cfg.mkdir()
        quality_scores = []
        performance_scores = []
        for seed in seeds:
            result, elapsed = run_generation(cfg, input_ids[cfg], seed, str(outdir_cfg), networks[cfg])
            assert result.returncode == 0, f"{cfg} generation failed for seed {seed}: {result.stderr}"
            
            # Load generated segmentation (label) and input segmentation.
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
            
            qscore = compute_composite_quality(iou_val, ssim_val)
            quality_scores.append(qscore)
            pscore = compute_performance_score(elapsed)
            performance_scores.append(pscore)
            print(f"[{cfg}] Seed {seed}: IoU={iou_val:.2f}, SSIM={ssim_val:.2f}, Quality Score={qscore:.1f}/100, Time={elapsed:.2f}s, Perf Score={pscore:.1f}/100")
        
        avg_quality = np.mean(quality_scores)
        consistency = compute_consistency_score(np.array(quality_scores))
        avg_perf = np.mean(performance_scores)
        # Overall configuration score: 70% from quality and consistency combined (we average them) and 30% from performance.
        overall_cfg_score = 0.7 * ((avg_quality + consistency) / 2) + 0.3 * avg_perf
        config_scores[cfg] = overall_cfg_score
        print(f"[{cfg}] Average Quality Score: {avg_quality:.1f}/100")
        print(f"[{cfg}] Consistency Score: {consistency:.1f}/100")
        print(f"[{cfg}] Average Performance Score: {avg_perf:.1f}/100")
        print(f"[{cfg}] Overall {cfg} Musa Score: {overall_cfg_score:.1f}/100")
    
    weighted_overall = sum(usage_weights[cfg] * config_scores[cfg] for cfg in config_list)
    print(f"Weighted Overall Musa Score: {weighted_overall:.1f}/100")
    assert weighted_overall > 65, f"Overall Musa Score too low: {weighted_overall:.1f}/100"