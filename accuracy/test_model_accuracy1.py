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

def compute_iou(binary1: np.ndarray, binary2: np.ndarray) -> float:
    """
    Compute the Intersection over Union (IoU) between two binary images.
    
    Args:
        binary1: A binary numpy array.
        binary2: A binary numpy array.
        
    Returns:
        IoU value (float).
    """
    intersection = np.logical_and(binary1, binary2).sum()
    union = np.logical_or(binary1, binary2).sum()
    if union == 0:
        return 1.0
    return intersection / union

@pytest.mark.integration
def test_seg2cat_segmentation_accuracy(tmp_path):
    """
    Integration test for the seg2cat generation pipeline:
    
    1. Run the generate_samples.py script using a known dataset sample.
    2. Load the generated segmentation label map and the input segmentation map.
    3. Binarize both images (assuming background=0 and non-zero indicates foreground).
    4. Compute the IoU between these binary maps.
    5. Assert that the IoU is above a threshold (e.g., 0.4), which indicates that the generated output
       preserves a reasonable amount of the input structure.
    
    This metric is not an "accuracy" in the traditional sense, but it gives a measurable indication
    of how well the model adheres to the provided input label map.
    """
    # Create a temporary output directory.
    outdir = tmp_path / "output"
    outdir.mkdir()

    # Command to run the generation script.
    # Adjust the parameters (paths, input_id, random seed, etc.) as necessary.
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

    # Load the generated segmentation output image.
    gen_label_path = outdir / "seg2cat_1666_1_label.png"
    # Also load the original input segmentation image saved by the pipeline.
    input_label_path = outdir / "seg2cat_1666_input.png"
    assert gen_label_path.exists(), "Generated segmentation label image not found."
    assert input_label_path.exists(), "Input segmentation image not found."

    # Convert images to grayscale (if not already) and to numpy arrays.
    gen_label = np.array(PIL.Image.open(gen_label_path).convert("L"))
    input_label = np.array(PIL.Image.open(input_label_path).convert("L"))

    # Binarize the images: assume pixels > 0 are foreground.
    gen_bin = (gen_label > 0).astype(np.uint8)
    input_bin = (input_label > 0).astype(np.uint8)

    # Compute the Intersection over Union (IoU)
    iou = compute_iou(gen_bin, input_bin)
    print(f"Segmentation IoU: {iou:.2f}")

    # Assert that the IoU is above a chosen threshold.
    # Here we assume that an IoU above 0.4 indicates a reasonable structural similarity.
    assert iou > 0.4, f"Segmentation IoU too low: {iou}"