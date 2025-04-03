import sys
sys.path.append('.')


# tests/test_model_accuracy.py

import subprocess
import sys
import os
from pathlib import Path

import numpy as np
import PIL.Image
import cv2
import pytest
from skimage.metrics import structural_similarity as ssim

def compute_iou(binary1: np.ndarray, binary2: np.ndarray) -> float:
    """
    Compute the Intersection over Union (IoU) between two binary images.
    
    Args:
        binary1: A binary numpy array.
        binary2: A binary numpy array.
        
    Returns:
        The IoU value (float). If the union is zero, returns 1.0.
    """
    intersection = np.logical_and(binary1, binary2).sum()
    union = np.logical_or(binary1, binary2).sum()
    return intersection / union if union > 0 else 1.0

@pytest.mark.integration
def test_model_accuracy(tmp_path):
    """
    Integration test for model accuracy using IoU and SSIM:
    
    1. Run the generate_samples.py script with a known configuration.
    2. Load the generated segmentation output image and the input segmentation image.
    3. Binarize both images (assuming that nonzero pixels represent the foreground).
    4. Compute the IoU between the generated and input segmentation masks.
    5. Load the generated color image and convert it to grayscale.
    6. Compute the Structural Similarity Index (SSIM) between the generated grayscale image and the input segmentation.
    7. Assert that the IoU and SSIM are above predefined thresholds, which indicates that the model is preserving the input structure.
    
    The thresholds (e.g., IoU > 0.4 and SSIM > 0.3) are adjustable based on your empirical findings.
    """
    # Create a temporary output directory.
    outdir = tmp_path / "output"
    outdir.mkdir()

    # Command to run the generation pipeline (adapt paths and parameters as needed).
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

    # Load the generated segmentation (label) image and the input segmentation image.
    gen_label_path = outdir / "seg2cat_1666_1_label.png"
    input_label_path = outdir / "seg2cat_1666_input.png"
    assert gen_label_path.exists(), "Generated segmentation image not found."
    assert input_label_path.exists(), "Input segmentation image not found."

    # Convert images to grayscale (if not already) and then to numpy arrays.
    gen_label = np.array(PIL.Image.open(gen_label_path).convert("L"))
    input_label = np.array(PIL.Image.open(input_label_path).convert("L"))

    # Binarize the images: assume any pixel value > 0 is foreground.
    gen_bin = (gen_label > 0).astype(np.uint8)
    input_bin = (input_label > 0).astype(np.uint8)
    
    # Compute Intersection over Union (IoU) for segmentation.
    iou_value = compute_iou(gen_bin, input_bin)
    print(f"IoU: {iou_value:.2f}")

    # Load the generated color image and convert it to grayscale.
    gen_color_path = outdir / "seg2cat_1666_1_color.png"
    assert gen_color_path.exists(), "Generated color image not found."
    gen_color = np.array(PIL.Image.open(gen_color_path).convert("RGB"))
    gen_gray = cv2.cvtColor(gen_color, cv2.COLOR_RGB2GRAY)
    
    # For this test, we use the input segmentation image (already grayscale) as a proxy for structural content.
    # (In a real test you might have a proper reference image.)
    input_gray = input_label

    # Compute SSIM between the generated grayscale image and the input.
    ssim_value = ssim(gen_gray, input_gray)
    print(f"SSIM: {ssim_value:.2f}")

    # Assert that the metrics are above the thresholds.
    # Thresholds can be adjusted based on your empirical evaluation.
    assert iou_value > 0.4, f"Segmentation IoU too low: {iou_value}"
    assert ssim_value > 0.3, f"SSIM too low: {ssim_value}"