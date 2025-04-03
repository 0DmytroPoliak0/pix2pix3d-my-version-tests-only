import sys
sys.path.append('.')

import os
import subprocess
from pathlib import Path
import PIL.Image
import pytest

def test_generate_samples_integration(tmp_path):
    # Use a temporary directory for output.
    outdir = tmp_path / "output"
    outdir.mkdir()

    # Define the command with known arguments.
    cmd = [
        "python", "applications/generate_samples.py",
        "--network", "checkpoints/pix2pix3d_seg2cat.pkl",
        "--outdir", str(outdir),
        "--random_seed", "1",
        "--input_id", "1666",
        "--cfg", "seg2cat"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    # Check that the script exits without errors.
    assert result.returncode == 0, f"STDERR: {result.stderr}"
    
    # Verify that output images exist.
    expected_files = [
        outdir / "seg2cat_1666_1_color.png",
        outdir / "seg2cat_1666_1_label.png"
    ]
    for file in expected_files:
        assert file.exists(), f"Output file missing: {file}"
        # Optionally, check that image dimensions are as expected.
        img = PIL.Image.open(file)
        assert img.size[0] > 0 and img.size[1] > 0

def test_script_execution_error_invalid_network(tmp_path):
    outdir = tmp_path / "bad_net"
    outdir.mkdir()
    cmd = [
        "python", "applications/generate_samples.py",
        "--network", "invalid_path.pkl",
        "--outdir", str(outdir),
        "--random_seed", "1",
        "--input_id", "1234",
        "--cfg", "seg2cat"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode != 0
    assert "Error" in result.stderr or result.stderr != ""

    
def test_script_creates_missing_output_directory(tmp_path):
    outdir = tmp_path / "auto_created"
    # Do not manually create outdir
    cmd = [
        "python", "applications/generate_samples.py",
        "--network", "checkpoints/pix2pix3d_seg2cat.pkl",
        "--outdir", str(outdir),
        "--random_seed", "1",
        "--input_id", "555",
        "--cfg", "seg2cat"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0
    assert outdir.exists()