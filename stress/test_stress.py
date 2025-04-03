import sys
sys.path.append('.')  # ensure project root is in the path


import time
import numpy as np
import cv2
import pytest
import concurrent.futures

# Dummy implementation of Gaussian blur (same as in boundaries tests).
def apply_gaussian_blur(image, kernel_size=5):
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def test_stress_apply_gaussian_blur():
    """
    Stress test for the Gaussian blur function.
    Repeatedly apply the blur on a 512x512 image to simulate heavy load.
    """
    image = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    iterations = 1000  # Number of iterations to simulate stress.
    
    start_time = time.time()
    for _ in range(iterations):
        _ = apply_gaussian_blur(image, kernel_size=5)
    end_time = time.time()
    
    total_time = end_time - start_time
    threshold = 10  # Expect 1000 iterations to complete in less than 10 seconds.
    assert total_time < threshold, f"Stress test exceeded threshold: {total_time:.2f} seconds (limit: {threshold}s)"

@pytest.mark.skip(reason="Optional stress test - run only when needed")
def test_stress_concurrent_execution():
    """
    Concurrent stress test: Launch multiple threads running the blur function concurrently.
    This test is skipped by default.
    """
    def blur_task():
        image = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
        return apply_gaussian_blur(image, kernel_size=5)
    
    iterations = 100
    start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(blur_task) for _ in range(iterations)]
        concurrent.futures.wait(futures)
    end_time = time.time()
    total_time = end_time - start_time
    threshold = 5  # Adjust threshold as needed.
    assert total_time < threshold, f"Concurrent stress test took too long: {total_time:.2f} seconds (limit: {threshold}s)"