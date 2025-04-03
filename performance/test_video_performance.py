# tests/performance/test_video_performance.py
"""
Performance Testing for Video Generation in pix2pix3D

This test measures the execution time of the video generation pipeline.
It creates a dummy sample input image on the fly, runs the video generation,
and asserts that the process completes within a predefined threshold.
"""

import sys
sys.path.append('.')

import time
import subprocess
from pathlib import Path
import pytest
import PIL.Image

# Helper function: run a command and measure its execution time.
def run_command(cmd):
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start
    return result, elapsed

# Helper function: create a dummy sample input image.
def create_dummy_sample_input(tmp_dir: Path, filename="example_input.png", size=(512, 512)):
    """
    Create a dummy grayscale image as sample input.
    Adjust the image if necessary to match network expectations.
    """
    img = PIL.Image.new("L", size, color=0)  # Create a black image.
    input_path = tmp_dir / filename
    img.save(input_path)
    return input_path

@pytest.mark.performance
def test_video_generation_execution_time(tmp_path):
    """
    Test that video generation completes within 60 seconds.
    """
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
    print(f"[Video Performance] Execution time: {elapsed:.2f} seconds")
    assert result.returncode == 0, f"Video generation failed: {result.stderr}"
    assert elapsed < 60, f"Video generation took too long: {elapsed:.2f} seconds"
    
    # Check that at least one output video file exists.
    video_files = list(outdir.glob("*.gif"))
    assert video_files, "No video output file found."