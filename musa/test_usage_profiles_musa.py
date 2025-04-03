import sys
sys.path.append('.')

import subprocess
import time
import os
from pathlib import Path
import numpy as np
import PIL.Image
import cv2
import pytest
from skimage.metrics import structural_similarity as ssim

# Helper functions
def run_command(cmd):
    """Run a command in a subprocess and measure its execution time."""
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start
    return result, elapsed

def compute_iou(binary1: np.ndarray, binary2: np.ndarray) -> float:
    """Compute Intersection over Union (IoU) between two binary images."""
    intersection = np.logical_and(binary1, binary2).sum()
    union = np.logical_or(binary1, binary2).sum()
    return intersection / union if union > 0 else 1.0

def compute_quality_score(iou: float, ssim_val: float) -> float:
    """
    Compute an overall quality score based on IoU and SSIM.
    For example, we may take a simple average of the IoU (scaled to 100)
    and SSIM (scaled to 100), or use another formula.
    """
    # Scale IoU and SSIM to a 0-100 range
    score_iou = iou * 100
    score_ssim = ssim_val * 100
    # Here we simply take the average.
    overall_score = (score_iou + score_ssim) / 2.0
    return overall_score

@pytest.mark.integration
def test_usage_profiles_musa(tmp_path):
    """
    Integration test simulating Musa’s operational profiles.
    
    Steps:
    1. Define an operational profile for seg2cat by assigning weights
       to different random seeds (simulate different usage frequencies).
    2. For each seed, run the generate_samples.py script to generate images.
    3. Compute quality metrics:
         - IoU between generated segmentation and input segmentation.
         - SSIM between generated color image (converted to grayscale) and input segmentation (as a proxy).
    4. Compute an overall quality score for each seed.
    5. Compute the weighted average quality score based on the operational profile.
    6. Print out the scores and assert that the weighted quality score is above a threshold.
    
    Expected Result:
      The weighted quality score should be above a predefined threshold (e.g., 60/100)
      to indicate that, under realistic usage conditions, the model produces acceptable outputs.
    """
    # Temporary output directory for generated images.
    outdir = tmp_path / "usage_output"
    outdir.mkdir()
    
    # Define an operational profile (for example purposes):
    # Here, each seed represents a scenario, and the weight is the probability of its occurrence.
    # (These values are illustrative; adjust them based on real usage estimates.)
    operational_profile = {
        42: 0.5,   # Seed 42 is used 50% of the time.
        43: 0.2,   # Seed 43 is used 20% of the time.
        44: 0.15,  # Seed 44 is used 15% of the time.
        45: 0.1,   # Seed 45 is used 10% of the time.
        46: 0.05   # Seed 46 is used 5% of the time.
    }
    
    quality_scores = []
    
    # Run the generation pipeline for each seed in the operational profile.
    for seed, weight in operational_profile.items():
        cmd = [
            "python", "applications/generate_samples.py",
            "--network", "checkpoints/pix2pix3d_seg2cat.pkl",
            "--outdir", str(outdir),
            "--random_seed", str(seed),
            "--input_id", "1666",
            "--cfg", "seg2cat"
        ]
        result, elapsed = run_command(cmd)
        assert result.returncode == 0, f"Generation failed for seed {seed}: {result.stderr}"
        
        # Load generated segmentation image and input segmentation image.
        gen_label_path = outdir / f"seg2cat_1666_{seed}_label.png"
        input_label_path = outdir / f"seg2cat_1666_input.png"
        assert gen_label_path.exists(), f"Missing generated segmentation image for seed {seed}"
        assert input_label_path.exists(), f"Missing input segmentation image for seed {seed}"
        
        gen_label = np.array(PIL.Image.open(gen_label_path).convert("L"))
        input_label = np.array(PIL.Image.open(input_label_path).convert("L"))
        
        # Binarize (assume nonzero is foreground)
        gen_bin = (gen_label > 0).astype(np.uint8)
        input_bin = (input_label > 0).astype(np.uint8)
        
        iou_value = compute_iou(gen_bin, input_bin)
        
        # Load generated color image, convert to grayscale.
        gen_color_path = outdir / f"seg2cat_1666_{seed}_color.png"
        assert gen_color_path.exists(), f"Missing generated color image for seed {seed}"
        gen_color = np.array(PIL.Image.open(gen_color_path).convert("RGB"))
        gen_gray = cv2.cvtColor(gen_color, cv2.COLOR_RGB2GRAY)
        
        # Use the input segmentation as a proxy reference for structure.
        ssim_value = ssim(gen_gray, input_label)
        
        overall_score = compute_quality_score(iou_value, ssim_value)
        quality_scores.append(overall_score * weight)
        
        print(f"Seed {seed} | IoU: {iou_value:.2f} | SSIM: {ssim_value:.2f} | Quality Score: {overall_score:.1f}")
    
    weighted_quality = sum(quality_scores)
    print(f"Weighted Overall Quality Score: {weighted_quality:.1f}/100")
    
    # Assert that the weighted overall quality score is above threshold.
    # (Threshold can be adjusted; here we require at least 60/100.)
    assert weighted_quality > 60, f"Weighted quality score too low: {weighted_quality:.1f}"