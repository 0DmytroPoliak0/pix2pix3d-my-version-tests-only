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
    
    Args:
        binary1 (np.ndarray): First binary image.
        binary2 (np.ndarray): Second binary image.
    
    Returns:
        float: IoU value (if union is zero, returns 1.0).
    """
    intersection = np.logical_and(binary1, binary2).sum()
    union = np.logical_or(binary1, binary2).sum()
    return intersection / union if union > 0 else 1.0

def compute_composite_score(iou_value: float, ssim_value: float) -> float:
    """
    Compute an overall quality score on a 0-100 scale.
    Here we weight IoU and SSIM equally.
    
    Args:
        iou_value (float): Intersection over Union.
        ssim_value (float): Structural Similarity Index.
        
    Returns:
        float: Composite quality score.
    """
    return 0.5 * (iou_value * 100) + 0.5 * (ssim_value * 100)

def run_generation(cfg: str, input_id: str, seed: int, outdir: str, network: str):
    """
    Run the generation pipeline using the provided configuration.
    
    Args:
        cfg (str): The configuration (e.g., "seg2cat", "seg2face", "edge2car").
        input_id (str): The input id from the dataset.
        seed (int): Random seed.
        outdir (str): Output directory.
        network (str): Path to the network checkpoint.
        
    Returns:
        subprocess.CompletedProcess: The result of the subprocess execution.
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
# Test Configurations
# -----------------------------------------------------------------------------
# Each tuple is: (configuration, network_checkpoint, input_id, expected_image_size)
test_configs = [
    ("seg2cat", "checkpoints/pix2pix3d_seg2cat.pkl", "1666", (128, 128)),  # Cats
    ("seg2face", "checkpoints/pix2pix3d_seg2face.pkl", "0", (128, 128)),    # Faces
    ("edge2car", "checkpoints/pix2pix3d_edge2car.pkl", "0", (128, 128)),     # Cars (edge version)
]

# -----------------------------------------------------------------------------
# Integration Tests for Accuracy and Metrics
# -----------------------------------------------------------------------------
@pytest.mark.integration
@pytest.mark.parametrize("cfg, network, input_id, expected_size", test_configs)
def test_model_accuracy_by_category(tmp_path, cfg, network, input_id, expected_size):
    """
    Accuracy and Metrics Test for different categories.
    
    Test Steps:
      1. Create a temporary output folder.
      2. Run generate_samples.py with the given configuration, seed = 1.
      3. Load the generated segmentation (label) image and the corresponding input segmentation.
      4. Binarize both images and compute IoU.
      5. Load the generated color image, convert it to grayscale, and compute SSIM against the input segmentation.
      6. Compute the composite quality score.
      7. Assert that IoU, SSIM, and quality score exceed the defined thresholds.
    
    Expected Results/Thresholds:
      - IoU > 0.4
      - SSIM > 0.3
      - Composite Quality Score > 60/100
    """
    outdir = tmp_path / "output"
    outdir.mkdir()

    # Run generation pipeline.
    result = run_generation(cfg, input_id, 1, str(outdir), network)
    assert result.returncode == 0, f"Generation failed: {result.stderr}"

    # Load generated segmentation (label) and input segmentation images.
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

    # Load the generated color image and compute SSIM.
    gen_color_path = outdir / f"{cfg}_{input_id}_1_color.png"
    assert gen_color_path.exists(), "Generated color image not found."
    gen_color = np.array(PIL.Image.open(gen_color_path).convert("RGB"))
    gen_gray = cv2.cvtColor(gen_color, cv2.COLOR_RGB2GRAY)
    input_gray = input_label  # Using input segmentation as a proxy for structure.
    ssim_value = ssim(gen_gray, input_gray)
    print(f"{cfg} SSIM: {ssim_value:.2f}")

    # Compute the composite quality score.
    qs = compute_composite_score(iou_value, ssim_value)
    print(f"{cfg} Overall Quality Score: {qs:.1f}/100")

    # Assertions.
    assert iou_value > 0.4, f"{cfg} Segmentation IoU too low: {iou_value}"
    assert ssim_value > 0.3, f"{cfg} SSIM too low: {ssim_value}"
    assert qs > 60, f"{cfg} Quality score too low: {qs}"
    #print(f"{cfg} All quality metrics met.")
    