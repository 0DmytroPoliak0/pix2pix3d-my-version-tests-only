# tests/test_performance.py
import sys
sys.path.append('.')

import time
import torch
import pytest

def dummy_mapping(z, pose, condition):
    # Simulate a delay for mapping function
    time.sleep(0.1)
    return torch.randn(1, 25, 256)

def test_mapping_performance():
    z = torch.randn(1, 512)
    pose = torch.zeros(1, 25)
    condition = {'mask': torch.zeros(1,1,128,128), 'pose': pose}
    start_time = time.time()
    _ = dummy_mapping(z, pose, condition)
    elapsed = time.time() - start_time
    # Assert that the mapping function finishes within 0.5 seconds
    assert elapsed < 0.5, f"Mapping function too slow: {elapsed:.2f} seconds"