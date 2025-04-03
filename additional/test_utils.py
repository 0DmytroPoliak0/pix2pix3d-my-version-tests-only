import sys
sys.path.append('.')

import numpy as np
import pytest
from training.utils import color_mask

def test_color_mask_shape():
    # Create a simple segmentation mask (2x2)
    dummy_mask = np.array([[0, 1], [2, 3]], dtype=np.int32)
    colored = color_mask(dummy_mask)
    # Expected output shape (2,2,3)
    assert colored.shape == (2, 2, 3), f"Expected shape (2,2,3), got {colored.shape}"

def test_color_mask_values():
    dummy_mask = np.array([[0, 1], [2, 3]], dtype=np.int32)
    colored = color_mask(dummy_mask)
    # All values should be in the range 0-255.
    assert colored.min() >= 0 and colored.max() <= 255, "Pixel values out of range (0-255)"
    