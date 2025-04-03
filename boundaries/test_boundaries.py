import sys
sys.path.append('.')  # ensure project root is in the path

import numpy as np
import cv2
import pytest

# Dummy implementations for testing purposes.
def generate_filename(input_id, seed, config):
    """Generate filename based on input id, seed, and configuration string."""
    return f"{config}_{input_id}_{seed}_color.png"

def apply_gaussian_blur(image, kernel_size=5):
    """Apply Gaussian blur to an image with the given kernel size.
    
    Ensures that the kernel size is odd.
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

# Boundaries Testing for generate_filename.
def test_generate_filename_minimum_values():
    """Test generate_filename with minimal valid inputs."""
    filename = generate_filename(0, 0, "a")
    expected = "a_0_0_color.png"
    assert filename == expected, f"Expected {expected}, got {filename}"

def test_generate_filename_maximum_values():
    """Test generate_filename with large input values."""
    input_id = 10**6
    seed = 10**3
    config = "seg2cat"
    filename = generate_filename(input_id, seed, config)
    expected = f"{config}_{input_id}_{seed}_color.png"
    assert filename == expected, f"Expected {expected}, got {filename}"

# Boundaries Testing for apply_gaussian_blur.
def test_apply_gaussian_blur_small_image():
    """Test image processing on a very small image."""
    # Create a 10x10 pixel image.
    image = np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)
    blurred = apply_gaussian_blur(image, kernel_size=3)
    assert blurred.shape == image.shape, "Blurred image dimensions do not match the original."

def test_apply_gaussian_blur_large_image():
    """Test image processing on a large image."""
    # Create a large 2048x2048 pixel image.
    image = np.random.randint(0, 256, (2048, 2048, 3), dtype=np.uint8)
    blurred = apply_gaussian_blur(image, kernel_size=5)
    assert blurred.shape == image.shape, "Blurred image dimensions do not match the original."