import sys
sys.path.append('.')

import pytest
import subprocess
import numpy as np
import PIL.Image
import cv2
import os
from pathlib import Path

# A simple function to compute Intersection over Union (IoU)
def compute_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0
    return intersection / union

@pytest.mark.integration
def test_edge2car_edge_overlap(tmp_path):
    """
    Integration test for the edge2car generation pipeline:
    
    1. Run the generation script using a known configuration.
    2. Load the generated color image.
    3. Convert the image to grayscale, apply a Gaussian blur, and then extract edges using Canny.
    4. Load the original input edge map (saved by the generation script).
    5. Binarize both the extracted edges and the input edge map.
    6. Compute the IoU between the two edge maps.
    7. Assert that the IoU is above a given threshold (here, 0.1 or 10% overlap),
       indicating that the generated image preserves a reasonable amount of the input structure.
    """
    # Create a temporary directory for outputs.
    outdir = tmp_path / "output"
    outdir.mkdir()

    # Command to run the generation script.
    cmd = [
        "python", "applications/generate_samples.py",
        "--network", "checkpoints/pix2pix3d_edge2car.pkl",
        "--outdir", str(outdir),
        "--random_seed", "42",
        "--input_id", "0",
        "--cfg", "edge2car"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"Generation failed: {result.stderr}"

    # Load the generated color image.
    color_img_path = outdir / "edge2car_0_42_color.png"
    assert color_img_path.exists(), "Missing generated color image."
    gen_color_img = np.array(PIL.Image.open(color_img_path).convert("RGB"))

    # Convert to grayscale and apply a Gaussian blur to smooth the image.
    gray = cv2.cvtColor(gen_color_img, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Canny edge detection with adjusted thresholds.
    extracted_edges = cv2.Canny(blurred, 30, 90)

    # Load the original input edge map.
    input_edge_path = outdir / "edge2car_0_input.png"
    assert input_edge_path.exists(), "Missing input edge map."
    input_edge = np.array(PIL.Image.open(input_edge_path).convert("L"))

    # Binarize the images using a fixed threshold.
    extracted_edges_bin = (extracted_edges > 127).astype(np.uint8)
    input_edge_bin = (input_edge > 127).astype(np.uint8)

    # Compute IoU between the binarized edge maps.
    iou = compute_iou(extracted_edges_bin, input_edge_bin)
    print(f"Edge overlap IoU: {iou:.2f} ({iou*100:.0f}%)")

    # For our current model and extraction parameters, we require at least 10% overlap.
    assert iou > 0.1, f"Edge overlap IoU too low: {iou}"