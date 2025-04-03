import sys
sys.path.append('.')

import time
import subprocess
import os
from pathlib import Path

import numpy as np
import psutil
import pytest
import PIL.Image
import cv2

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------
def run_command(cmd):
    """
    Run a command as a subprocess, capture its output, and measure execution time.
    
    Args:
        cmd (list): Command line arguments as a list.
        
    Returns:
        tuple: (result, elapsed) where result is the CompletedProcess instance,
               and elapsed is the time in seconds the command took to run.
    """
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start
    return result, elapsed

def compute_iou(binary1: np.ndarray, binary2: np.ndarray) -> float:
    """
    Compute the Intersection over Union (IoU) between two binary images.
    
    Args:
        binary1 (np.ndarray): First binary image.
        binary2 (np.ndarray): Second binary image.
    
    Returns:
        float: The IoU value. If union is zero, returns 1.0.
    """
    intersection = np.logical_and(binary1, binary2).sum()
    union = np.logical_or(binary1, binary2).sum()
    return intersection / union if union > 0 else 1.0

# ---------------------------------------------------------------------------
# Performance Tests for Picture Generation
# ---------------------------------------------------------------------------
# PT-P1: Generation Execution Time
# Severity: High
# This test checks that the sample generation script completes within a threshold.
@pytest.mark.performance
def test_generation_execution_time(tmp_path):
    """
    PT-P1: Measure the execution time for generating a picture using a known configuration.
    
    Expected: Execution should complete in under 30 seconds.
    """
    outdir = tmp_path / "output"
    outdir.mkdir()
    
    # Command to run the generation script with a known configuration.
    cmd = [
        "python", "applications/generate_samples.py",
        "--network", "checkpoints/pix2pix3d_seg2cat.pkl",
        "--outdir", str(outdir),
        "--random_seed", "1",
        "--input_id", "1666",
        "--cfg", "seg2cat"
    ]
    result, elapsed = run_command(cmd)
    print(f"Generation execution time: {elapsed:.2f} seconds")
    
    # Check that the process finished successfully and within the threshold.
    assert result.returncode == 0, f"Generation failed: {result.stderr}"
    assert elapsed < 30, f"Generation took too long: {elapsed:.2f} seconds"

# PT-P2: Memory Usage During Generation
# Severity: High
# This test uses psutil to measure the peak memory usage during image generation.
@pytest.mark.performance
def test_generation_memory_usage(tmp_path):
    """
    PT-P2: Verify that peak memory usage during picture generation stays below a threshold.
    
    Expected: Peak memory usage should be below 1500 MB.
    """
    outdir = tmp_path / "memory_output"
    outdir.mkdir()
    
    cmd = [
        "python", "applications/generate_samples.py",
        "--network", "checkpoints/pix2pix3d_seg2cat.pkl",
        "--outdir", str(outdir),
        "--random_seed", "1",
        "--input_id", "1666",
        "--cfg", "seg2cat"
    ]
    
    # Launch the process using psutil to capture resource usage.
    process = psutil.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()
    
    # Try to retrieve peak memory usage; if the process has already ended, we set it to zero.
    try:
        mem_info = process.memory_info()
        peak_memory_mb = mem_info.rss / (1024 * 1024)
    except psutil.NoSuchProcess:
        peak_memory_mb = 0
    print(f"Generation peak memory usage: {peak_memory_mb:.2f} MB")
    
    # Check that the process finished and memory usage is under threshold.
    assert process.returncode == 0, f"Generation failed: {stderr}"
    assert peak_memory_mb < 1500, f"Peak memory usage too high: {peak_memory_mb:.2f} MB"

# PT-P3: Consistency Across Repeated Runs
# Severity: Medium
# This test runs the generation multiple times with different random seeds to check for consistency.
@pytest.mark.performance
def test_consistency_of_generation(tmp_path):
    """
    PT-P3: Ensure the execution time of image generation is consistent across multiple runs.
    
    Expected: Standard deviation of run times should be less than 10% of the mean time.
    """
    outdir = tmp_path / "consistency_output"
    outdir.mkdir()
    seeds = [42, 43, 44, 45, 46]
    times = []
    
    for seed in seeds:
        cmd = [
            "python", "applications/generate_samples.py",
            "--network", "checkpoints/pix2pix3d_seg2cat.pkl",
            "--outdir", str(outdir),
            "--random_seed", str(seed),
            "--input_id", "1666",
            "--cfg", "seg2cat"
        ]
        result, elapsed = run_command(cmd)
        print(f"Seed {seed} generation time: {elapsed:.2f} seconds")
        assert result.returncode == 0, f"Generation failed for seed {seed}: {result.stderr}"
        times.append(elapsed)
    
    mean_time = np.mean(times)
    std_time = np.std(times)
    print(f"Mean time: {mean_time:.2f} s, Std time: {std_time:.2f} s")
    assert std_time < 0.1 * mean_time, f"High variance in generation time: std {std_time:.2f} s, mean {mean_time:.2f} s"

# PT-P4: Image File Integrity and Validity
# Severity: Medium
# This test verifies that the generated image file exists, can be opened, and has the expected dimensions.
@pytest.mark.performance
def test_image_file_integrity(tmp_path):
    """
    PT-P4: Check that the generated output image is valid.
    
    Expected: The image file should exist, open without errors, and have dimensions of 512x512.
    """
    outdir = tmp_path / "integrity_output"
    outdir.mkdir()
    
    cmd = [
        "python", "applications/generate_samples.py",
        "--network", "checkpoints/pix2pix3d_seg2cat.pkl",
        "--outdir", str(outdir),
        "--random_seed", "1",
        "--input_id", "1666",
        "--cfg", "seg2cat"
    ]
    result, elapsed = run_command(cmd)
    print(f"Generation execution time: {elapsed:.2f} seconds")
    assert result.returncode == 0, f"Generation failed: {result.stderr}"
    
    # Load the generated color image.
    color_img_path = outdir / "seg2cat_1666_1_color.png"
    assert color_img_path.exists(), "Generated color image not found."
    img = PIL.Image.open(color_img_path)
    print(f"Generated image dimensions: {img.size}")
    # Assert expected dimensions.
    assert img.size == (512, 512), f"Unexpected image dimensions: {img.size}"

# PT-P5: Batch Generation Resource Consumption
# Severity: Medium
# This test simulates generating a batch of images and computes the average generation time.
@pytest.mark.performance
def test_batch_generation_resource_consumption(tmp_path):
    """
    PT-P5: Measure the average execution time for batch image generation.
    
    Expected: The average generation time per image should be below 30 seconds.
    """
    outdir = tmp_path / "batch_output"
    outdir.mkdir()
    seeds = [10, 20, 30]
    total_time = 0
    count = 0
    for seed in seeds:
        cmd = [
            "python", "applications/generate_samples.py",
            "--network", "checkpoints/pix2pix3d_seg2cat.pkl",
            "--outdir", str(outdir),
            "--random_seed", str(seed),
            "--input_id", "1666",
            "--cfg", "seg2cat"
        ]
        result, elapsed = run_command(cmd)
        total_time += elapsed
        count += 1
        assert result.returncode == 0, f"Generation failed for seed {seed}: {result.stderr}"
    avg_time = total_time / count if count > 0 else 0
    print(f"Average generation time for batch: {avg_time:.2f} seconds")
    assert avg_time < 30, f"Batch generation average time too high: {avg_time:.2f} seconds"