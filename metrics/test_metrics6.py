# tests/test_metrics.py
import sys
sys.path.append('.')  # ensure repository modules are accessible

import os
import subprocess
from pathlib import Path
import time

import numpy as np
import PIL.Image
import pytest

# Import metric functions from the repository's metric files.
# (Adjust these import paths if necessary.)
from metrics.frechet_inception_distance import compute_fid
'''
from frechet_inception_distance import compute_fid
from inception_score import compute_inception_score
from kernel_inception_distance import compute_kid
'''

# Helper function to run a command and measure elapsed time.
def run_command(cmd):
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start_time
    return result, elapsed

@pytest.mark.integration
def test_generated_images_metrics(tmp_path):
    """
    Integration test to measure metrics (FID, Inception Score, and KID)
    on generated images. The test runs the generation pipeline with a known configuration,
    then uses the generated images as both the "real" and "generated" sets.
    
    Expected results (for self-comparison):
      - FID should be close to 0 (we assert < 5.0).
      - Inception Score should be within a plausible range (here, we assert > 8.0).
      - KID should be near 0 (we assert < 0.05).
    
    These thresholds are adjustable based on empirical evaluation.
    """
    # Create a temporary output directory for generated images.
    outdir = tmp_path / "generated"
    outdir.mkdir()

    # Run the generation pipeline.
    # (For this test we use seg2cat; adjust network and config as needed.)
    cmd = [
        "python", "applications/generate_samples.py",
        "--network", "checkpoints/pix2pix3d_seg2cat.pkl",
        "--outdir", str(outdir),
        "--random_seed", "1",
        "--input_id", "1666",
        "--cfg", "seg2cat"
    ]
    result, elapsed = run_command(cmd)
    assert result.returncode == 0, f"Generation failed: {result.stderr}"
    print(f"Generation completed in {elapsed:.2f} seconds.")

    # We expect the generation script to produce at least two images:
    # one for the color output and one for the segmentation/label output.
    # For metric computations, we'll use the generated color image as our test image.
    gen_color_path = outdir / "seg2cat_1666_1_color.png"
    assert gen_color_path.exists(), "Generated color image not found."
    
    # For a self-comparison test, we use the same folder as both "real" and "generated".
    generated_images_path = str(outdir)
    real_images_path = str(outdir)

    # Compute Frechet Inception Distance (FID).
    fid_value = compute_fid(
        generated_images_path,
        real_images_path,
        mode="clean",   # use clean mode for CPU
        batch_size=50,
        device="cpu"
    )
    print(f"FID: {fid_value:.2f}")
    assert fid_value < 5.0, f"FID too high: {fid_value}"

    # Compute Inception Score.
    is_score, is_std = compute_inception_score(generated_images_path, batch_size=50, device="cpu")
    print(f"Inception Score: {is_score:.2f} ± {is_std:.2f}")
    assert is_score > 8.0, f"Inception Score too low: {is_score}"

    # Compute Kernel Inception Distance (KID).
    kid_value = compute_kid(
        generated_images_path,
        real_images_path,
        mode="clean",
        batch_size=50,
        device="cpu"
    )
    print(f"KID: {kid_value:.4f}")
    assert kid_value < 0.05, f"KID too high: {kid_value}"

    # Compute an overall quality score (for demonstration) based on weighted metrics.
    # For example, quality_score = (100 - fid_value*2 - kid_value*100 + is_score) / 2
    quality_score = (100 - fid_value*2 - kid_value*100 + is_score) / 2
    print(f"Overall Quality Score: {quality_score:.1f}/100")
    # Assert that overall quality score is above a threshold (e.g., 60/100).
    assert quality_score > 60, f"Overall Quality Score too low: {quality_score}"

if __name__ == "__main__":
    # Allow running the test file directly.
    test_generated_images_metrics(Path("tmp_test"))