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
    """Compute Intersection over Union (IoU) between two binary images."""
    intersection = np.logical_and(binary1, binary2).sum()
    union = np.logical_or(binary1, binary2).sum()
    return intersection / union if union > 0 else 1.0

def compute_composite_score(iou_value: float, ssim_value: float) -> float:
    """
    Compute an overall quality score on a 0-100 scale.
    We weight IoU and SSIM equally.
    """
    return 0.5 * (iou_value * 100) + 0.5 * (ssim_value * 100)

def run_generation(cfg: str, input_id: str, seed: int, outdir: str, network: str):
    """
    Run the generation pipeline using the provided configuration.
    Returns the subprocess.CompletedProcess.
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
# Integration Tests for Metrics and Accuracy
# -----------------------------------------------------------------------------
@pytest.mark.integration
@pytest.mark.parametrize("cfg, network, input_id, expected_size", test_configs)
def test_model_accuracy_by_category(tmp_path, cfg, network, input_id, expected_size):
    """
    Accuracy and Metrics Test for different categories.
    
    Test Steps:
      1. Create a temporary output folder.
      2. Run generate_samples.py with the given configuration (seed = 1).
      3. Load the generated segmentation (label) image and the corresponding input segmentation.
      4. Binarize both images and compute IoU.
      5. Load the generated color image, convert it to grayscale, and compute SSIM against the input segmentation.
      6. Compute the composite quality score.
      7. Assert that IoU, SSIM, and composite quality score exceed the thresholds.
    
    Expected Results/Thresholds:
      - IoU > 0.4
      - SSIM > 0.3
      - Composite Quality Score > 60/100
    """
    outdir = tmp_path / "output"
    outdir.mkdir()

    result = run_generation(cfg, input_id, 1, str(outdir), network)
    assert result.returncode == 0, f"Generation failed: {result.stderr}"

    # Load generated segmentation and input segmentation images.
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

    # Load generated color image and compute SSIM.
    gen_color_path = outdir / f"{cfg}_{input_id}_1_color.png"
    assert gen_color_path.exists(), "Generated color image not found."
    gen_color = np.array(PIL.Image.open(gen_color_path).convert("RGB"))
    gen_gray = cv2.cvtColor(gen_color, cv2.COLOR_RGB2GRAY)
    input_gray = input_label  # Using input segmentation as proxy for structure.
    ssim_value = ssim(gen_gray, input_gray)
    print(f"{cfg} SSIM: {ssim_value:.2f}")

    qs = compute_composite_score(iou_value, ssim_value)
    print(f"{cfg} Overall Quality Score: {qs:.1f}/100")

    assert iou_value > 0.4, f"{cfg} Segmentation IoU too low: {iou_value}"
    assert ssim_value > 0.3, f"{cfg} SSIM too low: {ssim_value}"
    assert qs > 60, f"{cfg} Quality score too low: {qs}"

@pytest.mark.integration
def test_multiple_seeds_quality(tmp_path):
    """
    Musa Test for repeated runs (aggregated quality):
      - Generate images for multiple seeds.
      - Compute average quality score.
      - Expected: Average quality score should be above 60/100.
    
    Test Steps:
      1. Create a temporary output folder.
      2. Run generation for seeds [42, 43, 44, 45, 46] using seg2face configuration.
      3. For each seed, load generated segmentation, input segmentation, and generated color image.
      4. Compute IoU, SSIM, and composite quality score.
      5. Calculate the average quality score across all seeds.
      6. Assert that the average quality score exceeds the threshold.
    """
    outdir = tmp_path / "multi_seed_output"
    outdir.mkdir()
    network = "checkpoints/pix2pix3d_seg2face.pkl"
    input_id = "100"
    seeds = [42, 43, 44, 45, 46]
    scores = []
    
    for seed in seeds:
        result = run_generation("seg2face", input_id, seed, str(outdir), network)
        assert result.returncode == 0, f"Generation failed for seed {seed}: {result.stderr}"
        
        gen_seg = np.array(PIL.Image.open(outdir / f"seg2face_{input_id}_{seed}_label.png").convert("L"))
        input_seg = np.array(PIL.Image.open(outdir / f"seg2face_{input_id}_input.png").convert("L"))
        gen_color = np.array(PIL.Image.open(outdir / f"seg2face_{input_id}_{seed}_color.png").convert("RGB"))
        
        gen_bin = (gen_seg > 0).astype(np.uint8)
        input_bin = (input_seg > 0).astype(np.uint8)
        iou_value = compute_iou(gen_bin, input_bin)
        gen_gray = cv2.cvtColor(gen_color, cv2.COLOR_RGB2GRAY)
        ssim_value = ssim(gen_gray, input_seg)
        qs = compute_composite_score(iou_value, ssim_value)
        scores.append(qs)
        print(f"Seed {seed}: IoU={iou_value:.2f}, SSIM={ssim_value:.2f}, Quality Score={qs:.1f}/100")
    
    avg_quality = np.mean(scores)
    print(f"Weighted Overall Quality Score over seeds: {avg_quality:.1f}/100")
    assert avg_quality > 60, f"Average quality score too low: {avg_quality:.1f}/100"

@pytest.mark.integration
def test_quality_with_custom_input(tmp_path):
    """
    Musa Test for custom input usage:
      - Generate image using a custom input (simulated sketch).
      - Compute quality metrics and quality score.
      - Expected: Quality score should be above 60/100.
    
    Test Steps:
      1. Create a temporary output folder.
      2. Create a dummy custom input image (e.g., a 512x512 red image).
      3. Run generation using the custom input.
      4. Load generated segmentation, input segmentation, and generated color image.
      5. Compute IoU, SSIM, and composite quality score.
      6. Assert that the quality score exceeds the threshold.
    """
    outdir = tmp_path / "custom_input_output"
    outdir.mkdir()
    network = "checkpoints/pix2pix3d_seg2cat.pkl"
    custom_input = outdir / "custom_input.png"
    red_img = PIL.Image.new("RGB", (512, 512), color="red")
    red_img.save(custom_input)
    
    cmd = [
        "python", "applications/generate_samples.py",
        "--network", network,
        "--outdir", str(outdir),
        "--random_seed", "47",
        "--input_id", "0",
        "--cfg", "seg2cat",
        "--input", str(custom_input)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"Generation with custom input failed: {result.stderr}"
    
    gen_seg = np.array(PIL.Image.open(outdir / f"seg2cat_0_47_label.png").convert("L"))
    input_seg = np.array(PIL.Image.open(outdir / f"seg2cat_0_input.png").convert("L"))
    gen_color = np.array(PIL.Image.open(outdir / f"seg2cat_0_47_color.png").convert("RGB"))
    
    gen_bin = (gen_seg > 0).astype(np.uint8)
    input_bin = (input_seg > 0).astype(np.uint8)
    iou_value = compute_iou(gen_bin, input_bin)
    gen_gray = cv2.cvtColor(gen_color, cv2.COLOR_RGB2GRAY)
    ssim_value = ssim(gen_gray, input_seg)
    qs = compute_composite_score(iou_value, ssim_value)
    
    print(f"[custom input] IoU={iou_value:.2f}, SSIM={ssim_value:.2f}, Quality Score={qs:.1f}/100")
    assert qs > 60, f"Custom input quality score too low: {qs:.1f}/100"