# test_usecase.py

import sys
sys.path.append('.')

import subprocess
from pathlib import Path
import PIL.Image
import pytest

def test_generate_with_id_1666(tmp_path):
    outdir = tmp_path / "output"
    outdir.mkdir()

    cmd = [
        "python", "applications/generate_samples.py",
        "--network", "checkpoints/pix2pix3d_seg2cat.pkl",
        "--outdir", str(outdir),
        "--random_seed", "1",
        "--input_id", "555",
        "--cfg", "seg2cat"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"Script failed: {result.stderr}"

    expected_files = [
        outdir / "seg2cat_555_1_color.png",
        outdir / "seg2cat_555_1_label.png"
    ]

    for file in expected_files:
        assert file.exists(), f"Missing file: {file}"
        img = PIL.Image.open(file)
        assert img.size[0] > 0 and img.size[1] > 0

def test_script_with_invalid_network_path(tmp_path):
    outdir = tmp_path / "output"
    outdir.mkdir()

    cmd = [
        "python", "applications/generate_samples.py",
        "--network", "invalid/path/to/model.pkl",
        "--outdir", str(outdir),
        "--random_seed", "1",
        "--input_id", "1000",
        "--cfg", "seg2cat"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode != 0, "Expected script to fail with invalid network path"

def test_multiple_inputs_generate_images(tmp_path):
    for input_id in [3, 4, 5]:
        outdir = tmp_path / f"out_{input_id}"
        outdir.mkdir()
        cmd = [
            "python", "applications/generate_samples.py",
            "--network", "checkpoints/pix2pix3d_seg2cat.pkl",
            "--outdir", str(outdir),
            "--random_seed", "1",
            "--input_id", str(input_id),
            "--cfg", "seg2cat"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0
        assert (outdir / f"seg2cat_{input_id}_1_color.png").exists()