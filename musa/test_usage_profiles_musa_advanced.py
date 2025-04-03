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
    """
    Compute Intersection over Union (IoU) between two binary images.
    """
    intersection = np.logical_and(binary1, binary2).sum()
    union = np.logical_or(binary1, binary2).sum()
    return intersection / union if union > 0 else 1.0

def compute_composite_score(iou_value: float, ssim_value: float) -> float:
    """
    Compute an overall quality score on a 0-100 scale.
    We weight IoU and SSIM equally.
    """
    return 0.5 * (iou_value * 100) + 0.5 * (ssim_value * 100)

def compute_consistency_score(scores: np.ndarray) -> float:
    """
    Compute a consistency score based on the relative standard deviation of quality scores.
    A lower variability (lower relative standard deviation) yields a higher consistency score.
    Here we define it as: 100 - (relative_std), where relative_std = (std/mean * 100).
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
# Musa Advanced Integration Test: Combined Quality & Consistency
# -----------------------------------------------------------------------------
@pytest.mark.integration
def test_musa_advanced_score(tmp_path):
    """
    Musa Advanced Test:
      - For a given configuration (e.g., seg2face), run the generation pipeline over multiple seeds.
      - For each run:
          * Load the generated segmentation (label) image and the input segmentation image.
          * Binarize both and compute IoU.
          * Load the generated color image, convert it to grayscale, and compute SSIM against the input segmentation.
          * Compute a composite quality score.
      - Compute:
          * The average quality score across runs.
          * A consistency score based on the relative variability of the quality scores.
      - Calculate an overall Musa Score as a weighted sum (70% average quality + 30% consistency).
      - Expected: Overall Musa Score should exceed a defined threshold (e.g., >65/100).
    
    Test Steps:
      1. Create a temporary output folder.
      2. Run the generation pipeline with the specified configuration for seeds [42, 43, 44, 45, 46].
      3. For each run, load the generated images and compute IoU, SSIM, and composite quality score.
      4. Compute the average quality score and the consistency score.
      5. Compute the overall Musa Score.
      6. Assert that the overall Musa Score is above the threshold.
    """
    cfg = "seg2face"
    network = "checkpoints/pix2pix3d_seg2face.pkl"
    input_id = "100"
    seeds = [42, 43, 44, 45, 46]
    outdir = tmp_path / "musa_advanced_output"
    outdir.mkdir()

    quality_scores = []
    for seed in seeds:
        result = run_generation(cfg, input_id, seed, str(outdir), network)
        assert result.returncode == 0, f"Generation failed for seed {seed}: {result.stderr}"
        
        # Load generated segmentation (label) image and the corresponding input segmentation image.
        gen_label_path = outdir / f"{cfg}_{input_id}_{seed}_label.png"
        input_label_path = outdir / f"{cfg}_{input_id}_input.png"
        assert gen_label_path.exists(), f"Generated segmentation image not found for seed {seed}."
        assert input_label_path.exists(), f"Input segmentation image not found for seed {seed}."

        gen_label = np.array(PIL.Image.open(gen_label_path).convert("L"))
        input_label = np.array(PIL.Image.open(input_label_path).convert("L"))
        gen_bin = (gen_label > 0).astype(np.uint8)
        input_bin = (input_label > 0).astype(np.uint8)
        iou_val = compute_iou(gen_bin, input_bin)

        # Load generated color image and compute SSIM against input segmentation.
        gen_color_path = outdir / f"{cfg}_{input_id}_{seed}_color.png"
        assert gen_color_path.exists(), f"Generated color image not found for seed {seed}."
        gen_color = np.array(PIL.Image.open(gen_color_path).convert("RGB"))
        gen_gray = cv2.cvtColor(gen_color, cv2.COLOR_RGB2GRAY)
        ssim_val = ssim(gen_gray, input_label)
        
        qs = compute_composite_score(iou_val, ssim_val)
        quality_scores.append(qs)
        print(f"Seed {seed}: IoU={iou_val:.2f}, SSIM={ssim_val:.2f}, Quality Score={qs:.1f}/100")
    
    avg_quality = np.mean(quality_scores)
    consistency = compute_consistency_score(np.array(quality_scores))
    overall_musa_score = 0.7 * avg_quality + 0.3 * consistency
    print(f"Average Quality Score: {avg_quality:.1f}/100")
    print(f"Consistency Score: {consistency:.1f}/100")
    print(f"Overall Musa Score: {overall_musa_score:.1f}/100")

    # Assert that the overall Musa Score meets the threshold.
    assert overall_musa_score > 65, f"Overall Musa Score too low: {overall_musa_score:.1f}/100"