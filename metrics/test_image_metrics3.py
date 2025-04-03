import sys
sys.path.append('.')

import pytest
import subprocess
import numpy as np
import PIL.Image
import cv2
import os
from pathlib import Path

def compute_iou(mask1, mask2):
    """Compute Intersection over Union (IoU) of two binary masks."""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0
    return intersection / union

@pytest.mark.integration
def test_edge2car_edge_overlap(tmp_path):
    """
    Integration test for the edge2car generation pipeline:
    
    1. Run the generation script for several random seeds.
    2. For each run, load the generated color image and apply preprocessing:
       - Convert to grayscale.
       - Apply a Gaussian blur to reduce noise.
       - Use Canny edge detection with adjusted thresholds.
    3. Load the original input edge map (saved by the generation script).
    4. Binarize both the extracted and input edges.
    5. Compute the IoU for each run.
    6. Calculate the average IoU over all seeds and assert that it is above a threshold.
    
    This test provides a more comprehensive measure of edge preservation.
    """
    # Create a temporary directory for outputs.
    outdir = tmp_path / "output"
    outdir.mkdir()

    # Define a list of random seeds to test over.
    seeds = [42, 43, 44, 45, 46]
    iou_values = []

    for seed in seeds:
        # Command to run the generation script for edge2car.
        cmd = [
            "python", "applications/generate_samples.py",
            "--network", "checkpoints/pix2pix3d_edge2car.pkl",
            "--outdir", str(outdir),
            "--random_seed", str(seed),
            "--input_id", "0",
            "--cfg", "edge2car"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0, f"Generation failed for seed {seed}: {result.stderr}"

        # Load the generated color image.
        color_img_path = outdir / f"edge2car_0_{seed}_color.png"
        assert color_img_path.exists(), f"Missing generated color image for seed {seed}."
        gen_color_img = np.array(PIL.Image.open(color_img_path).convert("RGB"))

        # Convert the image to grayscale and apply Gaussian blur.
        gray = cv2.cvtColor(gen_color_img, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # Apply Canny edge detection with lower thresholds.
        extracted_edges = cv2.Canny(blurred, 50, 150)

        # Load the original input edge map.
        input_edge_path = outdir / "edge2car_0_input.png"
        assert input_edge_path.exists(), "Missing input edge map."
        input_edge = np.array(PIL.Image.open(input_edge_path).convert("L"))

        # Binarize both images using a threshold.
        extracted_edges_bin = (extracted_edges > 127).astype(np.uint8)
        input_edge_bin = (input_edge > 127).astype(np.uint8)

        # Compute IoU and store it.
        iou = compute_iou(extracted_edges_bin, input_edge_bin)
        iou_values.append(iou)
        print(f"Seed {seed} IoU: {iou:.2f}")

    avg_iou = np.mean(iou_values)
    print(f"Average IoU over seeds: {avg_iou:.2f} ({avg_iou*100:.0f}%)")

    # Assert that the average IoU is above a threshold (here, 20%).
    assert avg_iou > 0.2, f"Average IoU too low: {avg_iou}"