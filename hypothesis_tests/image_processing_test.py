import sys
sys.path.append('.')

import numpy as np
import cv2
import pytest
from hypothesis import given, strategies as st
from hypothesis.extra.numpy import arrays

# Example image processing function: applies a Gaussian blur.
def apply_gaussian_blur(image, kernel_size=5):
    # Ensure kernel size is odd.
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

# Hypothesis test for the Gaussian blur function.
@given(
    # Generate random images: height and width between 10 and 100, 3 channels (RGB).
    image=arrays(
        dtype=np.uint8,
        shape=st.tuples(st.integers(10, 100), st.integers(10, 100), st.just(3))
    ),
    kernel_size=st.integers(min_value=3, max_value=15)
)
def test_gaussian_blur_properties(image, kernel_size):
    blurred = apply_gaussian_blur(image, kernel_size)
    
    # Check that the blurred image retains the original shape.
    assert blurred.shape == image.shape, "Blurred image should have the same dimensions as the original."
    
    # Ensure all pixel values are within the valid range.
    assert blurred.min() >= 0 and blurred.max() <= 255, "Pixel values must remain between 0 and 255."
    
    # Verify that the blurring reduces the overall variance (i.e., smooths the image).
    original_variance = np.var(image)
    blurred_variance = np.var(blurred)
    assert blurred_variance <= original_variance, (
        f"Blurred variance ({blurred_variance}) should be less than or equal to original variance ({original_variance})."
    )

# To run this test, simply execute: pytest <this_file.py>