import sys
sys.path.append('.')


# tests/test_model_accuracy_by_category.py
import subprocess
import sys
import os
from pathlib import Path

import numpy as np
import PIL.Image
import cv2
import pytest
from skimage.metrics import structural_similarity as ssim

# Define helper functions
def compute_iou(binary1: np.ndarray, binary2: np.ndarray) -> float:
    intersection = np.logical_and(binary1, binary2).sum()
    union = np.logical_or(binary1, binary2).sum()
    return intersection / union if union > 0 else 1.0

def compute_composite_score(iou_value: float, ssim_value: float) -> float:
    # Example: a weighted combination; adjust weights as needed.
    # For instance, give 50% weight to IoU and 50% to SSIM, scaled to 100.
    return (iou_value * 50 + ssim_value * 50)

# Define test parameters for different configurations.
# Each tuple is: (configuration, network_checkpoint, input_id, expected_image_size)
# You may need to adjust these values to match your datasets.
test_configs = [
    ("seg2cat", "checkpoints/pix2pix3d_seg2cat.pkl", "1666", (128, 128)),  # Cats
    ("seg2face", "checkpoints/pix2pix3d_seg2face.pkl", "0", (128, 128)),    # Faces
    ("edge2car", "checkpoints/pix2pix3d_edge2car.pkl", "0", (128, 128)),     # Cars (edge version)
]

@pytest.mark.integration
@pytest.mark.parametrize("cfg, network, input_id, expected_size", test_configs)
def test_model_accuracy_by_category(tmp_path, cfg, network, input_id, expected_size):
    """
    Integration test for model accuracy for different categories.
    For each configuration:
      1. Run the generate_samples.py script.
      2. Load the generated segmentation (label) image and the corresponding input.
      3. Binarize both images and compute IoU.
      4. Load the generated color image, convert it to grayscale, and compute SSIM with the input.
      5. Compute a composite quality score and print the values.
      6. Assert that the quality scores are above defined thresholds.
    """
    outdir = tmp_path / "output"
    outdir.mkdir()

    cmd = [
        "python", "applications/generate_samples.py",
        "--network", network,
        "--outdir", str(outdir),
        "--random_seed", "1",
        "--input_id", input_id,
        "--cfg", cfg
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"Generation failed: {result.stderr}"

    # Load generated segmentation (label) image and input segmentation image.
    gen_label_path = outdir / f"{cfg}_{input_id}_1_label.png"
    input_label_path = outdir / f"{cfg}_{input_id}_input.png"
    assert gen_label_path.exists(), "Generated segmentation image not found."
    assert input_label_path.exists(), "Input segmentation image not found."

    gen_label = np.array(PIL.Image.open(gen_label_path).convert("L"))
    input_label = np.array(PIL.Image.open(input_label_path).convert("L"))
    gen_bin = (gen_label > 0).astype(np.uint8)
    input_bin = (input_label > 0).astype(np.uint8)
    iou_value = compute_iou(gen_bin, input_bin)
    print(f"{cfg} IoU: {iou_value:.2f}")

    # Load the generated color image and convert it to grayscale.
    gen_color_path = outdir / f"{cfg}_{input_id}_1_color.png"
    assert gen_color_path.exists(), "Generated color image not found."
    gen_color = np.array(PIL.Image.open(gen_color_path).convert("RGB"))
    gen_gray = cv2.cvtColor(gen_color, cv2.COLOR_RGB2GRAY)
    input_gray = input_label  # Using input segmentation as a proxy

    ssim_value = ssim(gen_gray, input_gray)
    print(f"{cfg} SSIM: {ssim_value:.2f}")

    quality_score = compute_composite_score(iou_value, ssim_value)
    print(f"{cfg} Overall Quality Score: {quality_score:.1f}/100")

    # Assertions (thresholds can be adjusted based on empirical results)
    assert iou_value > 0.4, f"{cfg} Segmentation IoU too low: {iou_value}"
    assert ssim_value > 0.3, f"{cfg} SSIM too low: {ssim_value}"
    assert quality_score > 60, f"{cfg} Quality score too low: {quality_score}"