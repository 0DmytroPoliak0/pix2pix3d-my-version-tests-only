# tests/performance/test_mesh_performance.py
"""
Performance Testing for Mesh Extraction in pix2pix3D

This test measures the execution time of the 3D mesh extraction pipeline.
It creates a dummy sample input image on the fly, runs the mesh extraction,
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
def test_mesh_extraction_execution_time(tmp_path):
    """
    Test that mesh extraction completes within 45 seconds.
    """
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
    print(f"[Mesh Performance] Extraction time: {elapsed:.2f} seconds")
    assert result.returncode == 0, f"Mesh extraction failed: {result.stderr}"
    assert elapsed < 45, f"Mesh extraction took too long: {elapsed:.2f} seconds"
    
    # Check that at least one mesh file (.ply) exists.
    mesh_files = list(outdir.glob("*.ply"))
    assert mesh_files, "No mesh output file found."