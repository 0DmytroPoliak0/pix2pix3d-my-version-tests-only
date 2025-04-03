import sys
sys.path.append('.')  # ensure project root is in the path


import time
import numpy as np
import concurrent.futures
import pytest
from hypothesis import given, settings, strategies as st

# Dummy simulation of an image generation function.
def simulate_image_generation(config, seed, image_size=(512, 512)):
    """
    Simulate image generation.
    The function uses the seed to determine a random delay (simulating processing time)
    and returns a dummy image (a random numpy array) of the specified size.
    """
    np.random.seed(seed)
    # Simulate processing delay between 5ms and 20ms.
    delay = np.random.uniform(0.005, 0.02)
    time.sleep(delay)
    # Return a dummy RGB image.
    return np.random.randint(0, 256, (image_size[0], image_size[1], 3), dtype=np.uint8)

@given(
    num_iterations=st.integers(min_value=50, max_value=200),
    config=st.sampled_from(["seg2cat", "seg2face", "edge2car"]),
    seed_start=st.integers(min_value=1, max_value=1000)
)
@settings(deadline=None)  # Disable deadline since the test might run a bit longer.
def test_concurrent_image_generation(num_iterations, config, seed_start):
    """
    Advanced stress test:
    - Run a number of concurrent image generation simulations.
    - Measure the average processing time per image.
    - Assert that the average time is below a predefined threshold.
    """
    def task(seed):
        return simulate_image_generation(config, seed)
    
    seeds = list(range(seed_start, seed_start + num_iterations))
    start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        # Execute image generation tasks concurrently.
        results = list(executor.map(task, seeds))
    total_time = time.time() - start_time
    avg_time = total_time / num_iterations
    
    # Set a threshold for average processing time per image (e.g., 0.03 seconds).
    threshold = 0.03
    assert avg_time < threshold, (
        f"Average processing time per image is too high: {avg_time:.4f}s (threshold: {threshold}s)"
    )