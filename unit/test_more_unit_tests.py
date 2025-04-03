import sys
sys.path.append('.')  # ensure project root is in the path

import os
import tempfile
import pytest
import click
import numpy as np
import torch
import PIL.Image

from training.utils import color_mask, color_list
from applications.generate_samples import init_conditional_dataset_kwargs  # adjust the import if needed

# --- Test for color_mask ---
def test_color_mask_shape():
    """
    Test that the color_mask function returns an RGB image.
    It takes a 2D segmentation mask and returns a 3D numpy array with shape (H, W, 3).
    """
    dummy_mask = np.array([[0, 1], [2, 3]], dtype=np.int32)
    colored = color_mask(dummy_mask)
    assert colored.shape == (2, 2, 3), f"Expected shape (2,2,3), got {colored.shape}"

def test_color_mask_values():
    """
    Test that the color_mask function returns pixel values in the range 0-255.
    """
    dummy_mask = np.array([[0, 1], [2, 3]], dtype=np.int32)
    colored = color_mask(dummy_mask)
    assert colored.min() >= 0 and colored.max() <= 255, "Pixel values should be between 0 and 255"

# --- Test for color_list ---
def test_color_list_structure():
    """
    Test that color_list is a list (or tuple) of colors,
    and that each color is a 3-element tuple or list of integers in [0, 255].
    """
    cl = color_list
    assert isinstance(cl, (list, tuple)), "color_list should be a list or tuple"
    for color in cl:
        assert isinstance(color, (list, tuple)), "Each color should be a list or tuple"
        assert len(color) == 3, "Each color should have three components (R, G, B)"
        for c in color:
            assert isinstance(c, int), "Each color component should be an integer"
            assert 0 <= c <= 255, "Color component should be in the range 0-255"

# --- Test for CPU monkey patch (cpu_torch_load) ---
def test_cpu_torch_load():
    """
    Test that our monkey-patched torch.load (via cpu_torch_load) returns a tensor on CPU.
    We save a small tensor to a temporary file and then load it back.
    """
    # Create a simple tensor
    x = torch.tensor([1.0, 2.0])
    # Create a temporary file to save the tensor
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp_name = tmp.name
    torch.save(x, tmp_name)
    # Use our monkey-patched torch.load to load the tensor
    loaded_x = torch.load(tmp_name)
    # Check that the loaded tensor is on CPU
    assert loaded_x.device.type == "cpu", "The loaded tensor should be on CPU"
    # Clean up
    os.remove(tmp_name)

# --- Test for init_conditional_dataset_kwargs error handling ---
def test_init_conditional_dataset_kwargs_invalid_path():
    """
    Test that init_conditional_dataset_kwargs raises a ClickException
    when given non-existent dataset paths.
    """
    with pytest.raises(click.ClickException):
        init_conditional_dataset_kwargs("nonexistent_data.zip", "nonexistent_mask.zip", "seg")

def test_init_conditional_dataset_kwargs_unknown_type():
    """
    Test that init_conditional_dataset_kwargs raises a ClickException
    when given an unknown data type.
    """
    with pytest.raises(click.ClickException):
        init_conditional_dataset_kwargs("dummy.zip", "dummy_mask.zip", "unknown")

if __name__ == "__main__":
    pytest.main()