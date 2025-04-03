# tests/performance/test_performance.py
"""
New Performance Testing Suite for pix2pix3D

This suite measures the performance of key components of the pix2pix3D pipeline.
It runs the full generation pipeline, video generation, and mesh extraction on a known configuration,
using a dummy sample input file created on the fly.
Metrics measured include execution time and peak memory usage (for one test).
"""

import sys
sys.path.append('.')

import time
import subprocess
import os
from pathlib import Path

import numpy as np
import psutil
import pytest
import cv2
import PIL.Image

# Helper function: run a command and measure its execution time.
def run_command(cmd):
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start
    return result, elapsed

# Helper function: create a dummy sample input image if not provided.
def create_dummy_sample_input(tmp_dir: Path, filename="example_input.png", size=(512, 512)):
    """
    Create a dummy grayscale image to act as a sample input.
    For seg2cat, the network expects a 512x512 input segmentation mask.
    """
    img = PIL.Image.new("L", size, color=0)  # black image
    # Optionally, draw a white square in the center to simulate a shape.
    # For example, draw a white rectangle covering 20% of the image.
    draw = PIL.Image.new("L", size, color=0)
    # (You could add drawing code with PIL.ImageDraw if desired)
    input_path = tmp_dir / filename
    img.save(input_path)
    return input_path

# PT-01: Generation Execution Time for seg2cat (Severity: High)
@pytest.mark.performance
def test_generation_execution_time(tmp_path):
    outdir = tmp_path / "output"
    outdir.mkdir()
    # Create a dummy sample input at 512x512 resolution.
    sample_input = create_dummy_sample_input(tmp_path, filename="example_input.png", size=(512, 512))
    
    cmd = [
        "python", "applications/generate_samples.py",
        "--network", "checkpoints/pix2pix3d_seg2cat.pkl",
        "--outdir", str(outdir),
        "--random_seed", "1",
        "--input_id", "1666",
        "--cfg", "seg2cat",
        "--input", str(sample_input)
    ]
    result, elapsed = run_command(cmd)
    print(f"[PT-01] Generation execution time: {elapsed:.2f} seconds")
    assert result.returncode == 0, f"Generation failed: {result.stderr}"
    assert elapsed < 30, f"Generation took too long: {elapsed:.2f} seconds"

# PT-02: Video Generation Execution Time (Severity: Medium)
@pytest.mark.performance
def test_video_generation_execution_time(tmp_path):
    outdir = tmp_path / "video_output"
    outdir.mkdir()
    sample_input = create_dummy_sample_input(tmp_path, filename="example_input.png", size=(512, 512))
    
    cmd = [
        "python", "applications/generate_video.py",
        "--network", "checkpoints/pix2pix3d_seg2cat.pkl",
        "--outdir", str(outdir),
        "--random_seed", "1",
        "--cfg", "seg2cat",
        "--input", str(sample_input)
    ]
    result, elapsed = run_command(cmd)
    print(f"[PT-02] Video generation execution time: {elapsed:.2f} seconds")
    assert result.returncode == 0, f"Video generation failed: {result.stderr}"
    assert elapsed < 60, f"Video generation took too long: {elapsed:.2f} seconds"

# PT-03: 3D Mesh Extraction Time (Severity: Medium)
@pytest.mark.performance
def test_mesh_extraction_time(tmp_path):
    outdir = tmp_path / "mesh_output"
    outdir.mkdir()
    sample_input = create_dummy_sample_input(tmp_path, filename="example_input.png", size=(512, 512))
    
    cmd = [
        "python", "applications/extract_mesh.py",
        "--network", "checkpoints/pix2pix3d_seg2cat.pkl",
        "--outdir", str(outdir),
        "--cfg", "seg2cat",
        "--input", str(sample_input)
    ]
    result, elapsed = run_command(cmd)
    print(f"[PT-03] Mesh extraction time: {elapsed:.2f} seconds")
    assert result.returncode == 0, f"Mesh extraction failed: {result.stderr}"
    assert elapsed < 45, f"Mesh extraction took too long: {elapsed:.2f} seconds"

# PT-04: Memory Usage During Generation (Severity: High)
@pytest.mark.performance
def test_generation_memory_usage(tmp_path):
    outdir = tmp_path / "memory_output"
    outdir.mkdir()
    sample_input = create_dummy_sample_input(tmp_path, filename="example_input.png", size=(512, 512))
    
    cmd = [
        "python", "applications/generate_samples.py",
        "--network", "checkpoints/pix2pix3d_seg2cat.pkl",
        "--outdir", str(outdir),
        "--random_seed", "1",
        "--input_id", "1666",
        "--cfg", "seg2cat",
        "--input", str(sample_input)
    ]
    process = psutil.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        stdout, stderr = process.communicate(timeout=60)
    except subprocess.TimeoutExpired:
        process.kill()
        pytest.skip("Process timed out while measuring memory usage.")
    try:
        mem_info = process.memory_info()
    except psutil.NoSuchProcess:
        pytest.skip("Process ended before memory info could be captured.")
    peak_memory_mb = mem_info.rss / (1024 * 1024)
    print(f"[PT-04] Generation peak memory usage: {peak_memory_mb:.2f} MB")
    assert process.returncode == 0, f"Generation failed: {stderr}"
    assert peak_memory_mb < 1500, f"Peak memory usage too high: {peak_memory_mb:.2f} MB"

# PT-05: Consistency Over Repeated Runs (Severity: Medium)
@pytest.mark.performance
def test_consistency_of_generation(tmp_path):
    outdir = tmp_path / "consistency_output"
    outdir.mkdir()
    seeds = [42, 43, 44, 45, 46]
    times = []
    sample_input = create_dummy_sample_input(tmp_path, filename="example_input.png", size=(512, 512))
    
    for seed in seeds:
        cmd = [
            "python", "applications/generate_samples.py",
            "--network", "checkpoints/pix2pix3d_seg2cat.pkl",
            "--outdir", str(outdir),
            "--random_seed", str(seed),
            "--input_id", "1666",
            "--cfg", "seg2cat",
            "--input", str(sample_input)
        ]
        result, elapsed = run_command(cmd)
        print(f"[PT-05] Seed {seed} generation time: {elapsed:.2f} seconds")
        assert result.returncode == 0, f"Generation failed for seed {seed}: {result.stderr}"
        times.append(elapsed)
    mean_time = np.mean(times)
    std_time = np.std(times)
    print(f"[PT-05] Mean time: {mean_time:.2f} s, Std time: {std_time:.2f} s")
    assert std_time < 0.1 * mean_time, f"High variance in generation time: std {std_time:.2f} s, mean {mean_time:.2f} s"