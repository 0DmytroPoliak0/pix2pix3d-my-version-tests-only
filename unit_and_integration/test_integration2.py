# tests/test_integration.py
import sys
sys.path.append('.')

import subprocess
import os
from pathlib import Path
import PIL.Image

def test_generate_samples_integration(tmp_path):
    # Use a temporary directory for outputs
    outdir = tmp_path / "output"
    outdir.mkdir()
    
    # Set command arguments (you can adapt these to your environment)
    command = [
        "python", "applications/generate_samples.py",
        "--network", "checkpoints/pix2pix3d_seg2cat.pkl",
        "--outdir", str(outdir),
        "--random_seed", "1",
        "--input_id", "0",
        "--cfg", "seg2cat"
    ]
    
    process = subprocess.run(command, capture_output=True, text=True)
    # Assert the process finished successfully
    assert process.returncode == 0, f"Process failed: {process.stderr}"
    
    # Check that the expected output file exists
    output_file = outdir / "seg2cat_0_1_color.png"
    assert output_file.exists(), "Output color image not found"
    # Optionally, load the image and check its dimensions
    img = PIL.Image.open(output_file)
    assert img.size == (512, 512), "Unexpected image dimensions"