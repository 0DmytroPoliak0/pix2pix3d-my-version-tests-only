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

def compute_iou(binary1: np.ndarray, binary2: np.ndarray) -> float:
    intersection = np.logical_and(binary1, binary2).sum()
    union = np.logical_or(binary1, binary2).sum()
    return intersection / union if union > 0 else 1.0

def compute_quality_score(iou: float, ssim_val: float, fid: float = None, fid_max: float = 100) -> float:
    """
    Compute a composite quality score based on IoU, SSIM, and optionally FID.
    """
    w_iou = 0.5
    w_ssim = 0.5
    if fid is not None:
        w_fid = 0.2
        norm_fid = max(0, 1 - fid / fid_max)
        quality = (w_iou * iou + w_ssim * ssim_val + w_fid * norm_fid) / (w_iou + w_ssim + w_fid)
    else:
        quality = (w_iou * iou + w_ssim * ssim_val) / (w_iou + w_ssim)
    return quality * 100

@pytest.mark.integration
def test_model_accuracy(tmp_path):
    """
    Integration test for model accuracy using IoU and SSIM.
    
    1. Runs the generate_samples.py script.
    2. Loads generated segmentation and color images.
    3. Computes IoU between generated and input segmentation masks.
    4. Computes SSIM between the generated grayscale image and input segmentation.
    5. Computes a composite quality score.
    6. Asserts that the quality score is above a threshold.
    """
    outdir = tmp_path / "output"
    outdir.mkdir()

    cmd = [
        "python", "applications/generate_samples.py",
        "--network", "checkpoints/pix2pix3d_seg2cat.pkl",
        "--outdir", str(outdir),
        "--random_seed", "1",
        "--input_id", "1666",
        "--cfg", "seg2cat"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"Generation failed: {result.stderr}"

    gen_label_path = outdir / "seg2cat_1666_1_label.png"
    input_label_path = outdir / "seg2cat_1666_input.png"
    assert gen_label_path.exists(), "Generated segmentation image not found."
    assert input_label_path.exists(), "Input segmentation image not found."

    gen_label = np.array(PIL.Image.open(gen_label_path).convert("L"))
    input_label = np.array(PIL.Image.open(input_label_path).convert("L"))
    gen_bin = (gen_label > 0).astype(np.uint8)
    input_bin = (input_label > 0).astype(np.uint8)
    iou_value = compute_iou(gen_bin, input_bin)
    print(f"IoU: {iou_value:.2f}")

    gen_color_path = outdir / "seg2cat_1666_1_color.png"
    assert gen_color_path.exists(), "Generated color image not found."
    gen_color = np.array(PIL.Image.open(gen_color_path).convert("RGB"))
    gen_gray = cv2.cvtColor(gen_color, cv2.COLOR_RGB2GRAY)
    input_gray = input_label
    ssim_value = ssim(gen_gray, input_gray)
    print(f"SSIM: {ssim_value:.2f}")

    quality_score = compute_quality_score(iou_value, ssim_value)
    print(f"Overall Quality Score: {quality_score:.1f}/100")

    # Set thresholds based on your observations
    assert iou_value > 0.4, f"Segmentation IoU too low: {iou_value}"
    assert ssim_value > 0.3, f"SSIM too low: {ssim_value}"
    assert quality_score > 50, f"Overall quality score too low: {quality_score}"