# test_usecase2.py

import sys
sys.path.append('.')

import subprocess
from pathlib import Path
import PIL.Image
import pytest

def test_output_image_is_valid_png(tmp_path):
    outdir = tmp_path / "out"
    outdir.mkdir()

    cmd = [
        "python", "applications/generate_samples.py",
        "--network", "checkpoints/pix2pix3d_seg2cat.pkl",
        "--outdir", str(outdir),
        "--random_seed", "1",
        "--input_id", "5",
        "--cfg", "seg2cat"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"Script failed: {result.stderr}"

    output_file = outdir / "seg2cat_5_1_color.png"
    assert output_file.exists(), "Output file does not exist"

    with open(output_file, "rb") as f:
        magic_number = f.read(8)
        assert magic_number == b"\x89PNG\r\n\x1a\n", "File is not a valid PNG"

def test_label_image_dimensions_match_color(tmp_path):
    outdir = tmp_path / "match_dims"
    outdir.mkdir()
    cmd = [
        "python", "applications/generate_samples.py",
        "--network", "checkpoints/pix2pix3d_seg2cat.pkl",
        "--outdir", str(outdir),
        "--random_seed", "1",
        "--input_id", "888",
        "--cfg", "seg2cat"
    ]
    subprocess.run(cmd)
    color = PIL.Image.open(outdir / "seg2cat_888_1_color.png")
    label = PIL.Image.open(outdir / "seg2cat_888_1_label.png")
    assert color.size == label.size
