import numpy as np
import pytest
from hypothesis import given, assume, strategies as st
from hypothesis.extra.numpy import arrays

# A simplified dummy SSIM function for demonstration purposes.
def dummy_ssim(img1, img2):
    """
    For identical images, returns 1.0.
    Otherwise, computes a rough similarity based on the mean absolute difference.
    """
    if np.array_equal(img1, img2):
        return 1.0
    # Compute mean absolute difference (scaled to 1.0)
    diff = np.mean(np.abs(img1.astype(np.float32) - img2.astype(np.float32)))
    return max(0.0, 1.0 - diff / 255)

# Strategy to generate a valid shape for images.
shape_strategy = st.tuples(
    st.integers(min_value=10, max_value=50),  # height
    st.integers(min_value=10, max_value=50),  # width
    st.just(3)  # channels (RGB)
)

@given(
    shape=shape_strategy,
    data=st.data(),
)
def test_dummy_ssim_identity(shape, data):
    # Generate a random image with the given shape.
    img = data.draw(arrays(dtype=np.uint8, shape=shape))
    ssim_value = dummy_ssim(img, img)
    assert abs(ssim_value - 1.0) < 1e-6, f"SSIM of identical images should be 1.0, got {ssim_value}"

@given(
    shape=shape_strategy,
    data=st.data(),
)
def test_dummy_ssim_with_noise(shape, data):
    # Generate an image and a noise array with the same shape.
    img = data.draw(arrays(dtype=np.uint8, shape=shape))
    noise = data.draw(arrays(dtype=np.uint8, shape=shape))
    
    # Skip cases where noise is all zeros or constant.
    assume(not np.all(noise == 0))
    assume(np.std(noise) > 0)
    
    # Create a noisy image by adding 0.5 * noise (as float) to the original image.
    noisy_img = np.clip(img.astype(np.float32) + 0.5 * noise.astype(np.float32), 0, 255).astype(np.uint8)
    
    # Compute SSIM between the original and the noisy image.
    ssim_value = dummy_ssim(img, noisy_img)
    
    # For a genuinely noisy image, SSIM should drop below 1.0.
    assert 0.0 <= ssim_value < 1.0, f"SSIM should drop below 1.0 for noisy images, got {ssim_value}"