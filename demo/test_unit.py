# test_unit.py

import pytest
from pathlib import Path
from PIL import Image

# Dummy helpers to simulate unit testing targets
def generate_filename(input_id, seed, cfg, type="color"):
    return f"{cfg}_{input_id}_{seed}_{type}.png"

def is_valid_image_size(image: Image.Image, expected_size=(512, 512)):
    return image.size == expected_size

def check_config_valid(cfg):
    return cfg in ["seg2cat", "seg2edge", "cat2seg"]

def construct_output_path(base, filename):
    return Path(base) / filename

def is_supported_extension(filepath):
    return Path(filepath).suffix.lower() in [".png", ".jpg", ".jpeg"]

# Unit tests

def test_generate_filename_format():
    filename = generate_filename(1666, 1, "seg2cat")
    assert filename == "seg2cat_1666_1_color.png"

def test_image_size_match(tmp_path):
    img_path = tmp_path / "img.png"
    Image.new("RGB", (512, 512)).save(img_path)
    img = Image.open(img_path)
    assert is_valid_image_size(img)

def test_config_validation_passes():
    assert check_config_valid("cat2seg")

def test_output_path_construction():
    path = construct_output_path("results", "img.png")
    assert str(path) == "results/img.png"

def test_supported_image_extension():
    assert is_supported_extension("sample.jpeg")
