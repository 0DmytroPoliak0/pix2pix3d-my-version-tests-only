import subprocess
import numpy as np
import PIL.Image
import cv2
import pytest
from skimage.metrics import structural_similarity as ssim

def compute_iou(binary1: np.ndarray, binary2: np.ndarray) -> float:
    """Compute Intersection over Union (IoU) between two binary images."""
    intersection = np.logical_and(binary1, binary2).sum()
    union = np.logical_or(binary1, binary2).sum()
    return intersection / union if union > 0 else 1.0

def quality_score(iou: float, ssim_val: float) -> float:
    """
    Compute an overall quality score on a 0-100 scale.
    Here we weight IoU and SSIM equally.
    """
    return 0.5 * (iou * 100) + 0.5 * (ssim_val * 100)

def run_generation(cfg: str, input_id: str, seed: int, outdir: str, network: str):
    """
    Run the generation script for the specified configuration.
    Returns the subprocess CompletedProcess.
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

@pytest.mark.integration
def test_seg2face_quality(tmp_path):
    """
    Musa Test for seg2face:
      - Generate a face image.
      - Compute IoU and SSIM between the generated segmentation and input.
      - Calculate a quality score.
      - Expected: Quality score should be above 60/100.
    
    Test Steps:
      1. Create a temporary output folder.
      2. Run generation with seg2face configuration using a fixed seed.
      3. Load generated segmentation, input segmentation, and generated color image.
      4. Binarize the segmentation images and compute IoU.
      5. Convert the generated color image to grayscale and compute SSIM against the input segmentation.
      6. Calculate the overall quality score.
      7. Assert that the quality score is above the threshold.
    """
    outdir = tmp_path / "seg2face_output"
    outdir.mkdir()
    network = "checkpoints/pix2pix3d_seg2face.pkl"
    input_id = "100"  # example input id for face dataset
    seed = 42

    result = run_generation("seg2face", input_id, seed, str(outdir), network)
    assert result.returncode == 0, f"Generation failed: {result.stderr}"
    
    gen_seg = np.array(PIL.Image.open(outdir / f"seg2face_{input_id}_{seed}_label.png").convert("L"))
    input_seg = np.array(PIL.Image.open(outdir / f"seg2face_{input_id}_input.png").convert("L"))
    gen_color = np.array(PIL.Image.open(outdir / f"seg2face_{input_id}_{seed}_color.png").convert("RGB"))
    
    gen_bin = (gen_seg > 0).astype(np.uint8)
    input_bin = (input_seg > 0).astype(np.uint8)
    
    iou_val = compute_iou(gen_bin, input_bin)
    gen_gray = cv2.cvtColor(gen_color, cv2.COLOR_RGB2GRAY)
    ssim_val = ssim(gen_gray, input_seg)
    qs = quality_score(iou_val, ssim_val)
    
    print(f"[seg2face] Seed {seed}: IoU={iou_val:.2f}, SSIM={ssim_val:.2f}, Quality Score={qs:.1f}/100")
    # Lowered threshold to 60/100
    assert qs > 60, f"Face quality score too low: {qs:.1f}/100"

@pytest.mark.integration
def test_seg2cat_quality(tmp_path):
    """
    Musa Test for seg2cat:
      - Generate a cat image.
      - Compute quality metrics and overall quality score.
      - Expected: Quality score should be above 70/100.
    
    Test Steps:
      1. Create a temporary output folder.
      2. Run generation with seg2cat configuration.
      3. Load generated segmentation, input segmentation, and generated color image.
      4. Binarize the segmentation images and compute IoU.
      5. Convert the generated color image to grayscale and compute SSIM.
      6. Calculate the quality score and assert it exceeds the threshold.
    """
    outdir = tmp_path / "seg2cat_output"
    outdir.mkdir()
    network = "checkpoints/pix2pix3d_seg2cat.pkl"
    input_id = "1666"
    seed = 43

    result = run_generation("seg2cat", input_id, seed, str(outdir), network)
    assert result.returncode == 0, f"Generation failed: {result.stderr}"
    
    gen_seg = np.array(PIL.Image.open(outdir / f"seg2cat_{input_id}_{seed}_label.png").convert("L"))
    input_seg = np.array(PIL.Image.open(outdir / f"seg2cat_{input_id}_input.png").convert("L"))
    gen_color = np.array(PIL.Image.open(outdir / f"seg2cat_{input_id}_{seed}_color.png").convert("RGB"))
    
    gen_bin = (gen_seg > 0).astype(np.uint8)
    input_bin = (input_seg > 0).astype(np.uint8)
    
    iou_val = compute_iou(gen_bin, input_bin)
    gen_gray = cv2.cvtColor(gen_color, cv2.COLOR_RGB2GRAY)
    ssim_val = ssim(gen_gray, input_seg)
    qs = quality_score(iou_val, ssim_val)
    
    print(f"[seg2cat] Seed {seed}: IoU={iou_val:.2f}, SSIM={ssim_val:.2f}, Quality Score={qs:.1f}/100")
    assert qs > 70, f"Cat quality score too low: {qs:.1f}/100"

@pytest.mark.integration
def test_edge2car_quality(tmp_path):
    """
    Musa Test for edge2car:
      - Generate a car image from an edge sketch.
      - Compute quality metrics and overall quality score.
      - Expected: Quality score should be above 75/100.
    
    Test Steps:
      1. Create a temporary output folder.
      2. Run generation with edge2car configuration.
      3. Load generated segmentation, input edge map, and generated color image.
      4. Binarize the images with a threshold of 127.
      5. Compute IoU and convert color image to grayscale to compute SSIM.
      6. Calculate the quality score and assert it exceeds the threshold.
    """
    outdir = tmp_path / "edge2car_output"
    outdir.mkdir()
    network = "checkpoints/pix2pix3d_edge2car.pkl"
    input_id = "0"  # For edge2car, the input is an edge map.
    seed = 44

    result = run_generation("edge2car", input_id, seed, str(outdir), network)
    assert result.returncode == 0, f"Generation failed: {result.stderr}"
    
    gen_seg = np.array(PIL.Image.open(outdir / f"edge2car_{input_id}_{seed}_label.png").convert("L"))
    input_seg = np.array(PIL.Image.open(outdir / f"edge2car_{input_id}_input.png").convert("L"))
    gen_color = np.array(PIL.Image.open(outdir / f"edge2car_{input_id}_{seed}_color.png").convert("RGB"))
    
    gen_bin = (gen_seg > 127).astype(np.uint8)
    input_bin = (input_seg > 127).astype(np.uint8)
    
    iou_val = compute_iou(gen_bin, input_bin)
    gen_gray = cv2.cvtColor(gen_color, cv2.COLOR_RGB2GRAY)
    ssim_val = ssim(gen_gray, input_seg)
    qs = quality_score(iou_val, ssim_val)
    
    print(f"[edge2car] Seed {seed}: IoU={iou_val:.2f}, SSIM={ssim_val:.2f}, Quality Score={qs:.1f}/100")
    assert qs > 75, f"Car quality score too low: {qs:.1f}/100"

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
      4. Compute IoU, SSIM, and quality score.
      5. Calculate the average quality score over all seeds.
      6. Assert that the average quality score exceeds the threshold.
    """
    outdir = tmp_path / "multi_seed_output"
    outdir.mkdir()
    network = "checkpoints/pix2pix3d_seg2face.pkl"  # Using face generation for this test.
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
        iou_val = compute_iou(gen_bin, input_bin)
        gen_gray = cv2.cvtColor(gen_color, cv2.COLOR_RGB2GRAY)
        ssim_val = ssim(gen_gray, input_seg)
        qs = quality_score(iou_val, ssim_val)
        scores.append(qs)
        print(f"Seed {seed}: IoU={iou_val:.2f}, SSIM={ssim_val:.2f}, Quality Score={qs:.1f}/100")
    
    avg_quality = np.mean(scores)
    print(f"Weighted Overall Quality Score over seeds: {avg_quality:.1f}/100")
    assert avg_quality > 60, f"Average quality score too low: {avg_quality:.1f}/100"

@pytest.mark.integration
def test_quality_with_custom_input(tmp_path):
    """
    Musa Test for custom input usage:
      - Use a custom input image (simulated by a dummy image) for generation.
      - Compute quality metrics and quality score.
      - Expected: Quality score should be above 60/100.
    
    Test Steps:
      1. Create a temporary output folder.
      2. Create a dummy custom input image (e.g., a 512x512 red image).
      3. Run generation using the custom input.
      4. Load generated segmentation, input segmentation, and generated color image.
      5. Compute IoU, SSIM, and quality score.
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
        "--input_id", "0",  # This parameter can be ignored in custom mode.
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
    iou_val = compute_iou(gen_bin, input_bin)
    gen_gray = cv2.cvtColor(gen_color, cv2.COLOR_RGB2GRAY)
    ssim_val = ssim(gen_gray, input_seg)
    qs = quality_score(iou_val, ssim_val)
    
    print(f"[custom input] IoU={iou_val:.2f}, SSIM={ssim_val:.2f}, Quality Score={qs:.1f}/100")
    assert qs > 60, f"Custom input quality score too low: {qs:.1f}/100"