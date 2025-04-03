import sys
sys.path.append('.')

import subprocess
import os
from pathlib import Path
import time

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
    """
    intersection = np.logical_and(binary1, binary2).sum()
    union = np.logical_or(binary1, binary2).sum()
    return intersection / union if union > 0 else 1.0

def compute_composite_quality(iou_value: float, ssim_value: float) -> float:
    """
    Compute an overall quality score on a 0-100 scale,
    weighting IoU and SSIM equally.
    """
    score = 0.5 * (iou_value * 100) + 0.5 * (ssim_value * 100)
    # Boundaries check: ensure the score is between 0 and 100.
    return max(0, min(100, score))

def compute_consistency_score(scores: np.ndarray) -> float:
    """
    Compute a consistency score based on relative standard deviation.
    Defined as: 100 - (std/mean * 100)
    """
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    relative_std = (std_score / mean_score * 100) if mean_score > 0 else 0
    score = max(0, 100 - relative_std)
    # Boundaries check
    return max(0, min(100, score))

def compute_performance_score(elapsed_time: float, ideal_time: float) -> float:
    """
    Compute a performance score on a 0-100 scale.
    If elapsed_time <= ideal_time, score is 100.
    Otherwise, score = 100 * (ideal_time / elapsed_time), capped at 100.
    """
    if elapsed_time <= ideal_time:
        score = 100.0
    else:
        score = 100 * (ideal_time / elapsed_time)
    # Boundaries check
    return max(0, min(100, score))

def run_generation(cfg: str, input_id: str, seed: int, outdir: str, network: str):
    """
    Run the generation pipeline using the provided configuration.
    Returns a tuple: (CompletedProcess, elapsed_time)
    """
    cmd = [
        "python", "applications/generate_samples.py",
        "--network", network,
        "--outdir", outdir,
        "--random_seed", str(seed),
        "--input_id", input_id,
        "--cfg", cfg
    ]
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start
    return result, elapsed

# Ideal execution times (in seconds) for each configuration.
ideal_times = {
    "seg2face": 9.0,
    "seg2cat": 9.0,
    "edge2car": 22.0,
}

# -----------------------------------------------------------------------------
# Musa Advanced Integration Test with Performance Metrics, Boundaries, and Stress Testing
# -----------------------------------------------------------------------------
@pytest.mark.integration
def test_usage_profiles_with_performance(tmp_path):
    """
    Musa Advanced Test with Usage Profiles, Performance, Boundaries, and Stress:
      - For each configuration (seg2face, seg2cat, edge2car), run the generation pipeline for a set of seeds.
      - If STRESS_MODE is enabled, use a larger set of seeds to simulate stress.
      - For each run, compute quality metrics (IoU, SSIM, composite quality) and measure performance (execution time).
      - Compute a performance score using an ideal time for each configuration.
      - Verify that all computed scores are within 0-100.
      - For each configuration, compute:
            * Average quality score, and also min and max quality scores.
            * Consistency score (from quality variability).
            * Average performance score, and min and max performance scores.
            * Overall configuration score = 0.7 * ((avg_quality + consistency)/2) + 0.3 * (avg_performance).
      - Finally, compute a weighted overall Musa score using predefined usage weights.
      - Assert that the overall weighted Musa score exceeds a defined threshold (e.g., >65/100).
    """
    # Usage weights for each configuration.
    usage_weights = {"seg2face": 0.50, "seg2cat": 0.30, "edge2car": 0.20}
    networks = {
        "seg2face": "checkpoints/pix2pix3d_seg2face.pkl",
        "seg2cat": "checkpoints/pix2pix3d_seg2cat.pkl",
        "edge2car": "checkpoints/pix2pix3d_edge2car.pkl"
    }
    input_ids = {"seg2face": "100", "seg2cat": "1666", "edge2car": "0"}
    
    # Use stress mode if the environment variable is set.
    stress_mode = os.getenv("STRESS_MODE", "false").lower() == "true"
    if stress_mode:
        seeds = list(range(42, 42 + 50))
        print("Stress mode enabled: Running with 50 seeds.")
    else:
        seeds = [42, 43, 44, 45, 46]
        print("Standard mode: Running with 5 seeds.")
        
    config_list = ["seg2face", "seg2cat", "edge2car"]
    config_scores = {}

    for cfg in config_list:
        outdir_cfg = tmp_path / f"{cfg}_output"
        outdir_cfg.mkdir()
        quality_scores = []
        performance_scores = []
        for seed in seeds:
            result, elapsed = run_generation(cfg, input_ids[cfg], seed, str(outdir_cfg), networks[cfg])
            assert result.returncode == 0, f"{cfg} generation failed for seed {seed}: {result.stderr}"

            # Load generated segmentation (label) and input segmentation images.
            gen_label_path = outdir_cfg / f"{cfg}_{input_ids[cfg]}_{seed}_label.png"
            input_label_path = outdir_cfg / f"{cfg}_{input_ids[cfg]}_input.png"
            assert gen_label_path.exists(), f"{cfg} generated segmentation not found for seed {seed}"
            assert input_label_path.exists(), f"{cfg} input segmentation not found for seed {seed}"
            gen_label = np.array(PIL.Image.open(gen_label_path).convert("L"))
            input_label = np.array(PIL.Image.open(input_label_path).convert("L"))
            gen_bin = (gen_label > 0).astype(np.uint8)
            input_bin = (input_label > 0).astype(np.uint8)
            iou_val = compute_iou(gen_bin, input_bin)
            # Boundaries check for IoU.
            assert 0.0 <= iou_val <= 1.0, f"IoU out of bounds: {iou_val}"

            # Load generated color image and compute SSIM.
            gen_color_path = outdir_cfg / f"{cfg}_{input_ids[cfg]}_{seed}_color.png"
            assert gen_color_path.exists(), f"{cfg} generated color image not found for seed {seed}"
            gen_color = np.array(PIL.Image.open(gen_color_path).convert("RGB"))
            gen_gray = cv2.cvtColor(gen_color, cv2.COLOR_RGB2GRAY)
            ssim_val = ssim(gen_gray, input_label)
            # Boundaries check for SSIM.
            assert 0.0 <= ssim_val <= 1.0, f"SSIM out of bounds: {ssim_val}"
            
            qscore = compute_composite_quality(iou_val, ssim_val)
            # Boundaries check for composite quality.
            assert 0.0 <= qscore <= 100.0, f"Composite quality score out of bounds: {qscore}"
            quality_scores.append(qscore)
            
            pscore = compute_performance_score(elapsed, ideal_times[cfg])
            # Boundaries check for performance score.
            assert 0.0 <= pscore <= 100.0, f"Performance score out of bounds: {pscore}"
            performance_scores.append(pscore)
            
            print(f"[{cfg}] Seed {seed}: IoU={iou_val:.2f}, SSIM={ssim_val:.2f}, Quality Score={qscore:.1f}/100, Time={elapsed:.2f}s, Perf Score={pscore:.1f}/100")
        
        avg_quality = np.mean(quality_scores)
        min_quality = np.min(quality_scores)
        max_quality = np.max(quality_scores)
        consistency = compute_consistency_score(np.array(quality_scores))
        avg_perf = np.mean(performance_scores)
        min_perf = np.min(performance_scores)
        max_perf = np.max(performance_scores)
        overall_cfg_score = 0.7 * ((avg_quality + consistency) / 2) + 0.3 * avg_perf
        config_scores[cfg] = overall_cfg_score
        
        print(f"[{cfg}] Quality: min={min_quality:.1f}/100, max={max_quality:.1f}/100, avg={avg_quality:.1f}/100")
        print(f"[{cfg}] Consistency Score: {consistency:.1f}/100")
        print(f"[{cfg}] Performance: min={min_perf:.1f}/100, max={max_perf:.1f}/100, avg={avg_perf:.1f}/100")
        print(f"[{cfg}] Overall {cfg} Musa Score: {overall_cfg_score:.1f}/100")
    
    weighted_overall = sum(usage_weights[cfg] * config_scores[cfg] for cfg in config_list)
    print(f"Weighted Overall Musa Score: {weighted_overall:.1f}/100")
    assert weighted_overall > 65, f"Overall Musa Score too low: {weighted_overall:.1f}/100"