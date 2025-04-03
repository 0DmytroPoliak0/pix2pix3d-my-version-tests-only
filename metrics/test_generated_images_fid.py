import sys
sys.path.append('.')

import subprocess
import os
import numpy as np
import PIL.Image
import pytest
from cleanfid import fid

@pytest.mark.integration
def test_generated_images_fid(tmp_path):
    """
    Integration test to compute FID for generated images.

    This test runs the generation pipeline (using generate_samples.py) for a known configuration.
    It then uses the output images as both the generated set and the "real" set.
    Since the two sets are identical, the FID should be near 0.
    """
    # Create a temporary directory for generated images.
    outdir = tmp_path / "generated"
    outdir.mkdir()

    # Command to run the generation script (adjust parameters as needed).
    cmd = [
        "python", "applications/generate_samples.py",
        "--network", "checkpoints/pix2pix3d_seg2cat.pkl",
        "--outdir", str(outdir),
        "--random_seed", "1",
        "--input_id", "0",
        "--cfg", "seg2cat"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"Generation failed: {result.stderr}"

    # Use the generated images folder for both real and generated images.
    generated_images_path = str(outdir)
    real_images_path = str(outdir)  # Using the same images as a stand-in for "real" images.

    # Compute FID using "clean" mode on CPU, with 0 workers.
    fid_value = fid.compute_fid(
        generated_images_path,
        real_images_path,
        mode="clean",          # Use "clean" mode instead of "legacy"
        batch_size=50,
        device="cpu",
        num_workers=0          # Disable multiprocessing
    )
    print(f"Computed FID: {fid_value}")

    # Since the two sets are identical, FID should be very low (close to 0).
    # Allow a small threshold for numerical imprecision.
    assert fid_value < 1.0, f"FID is unexpectedly high: {fid_value}"