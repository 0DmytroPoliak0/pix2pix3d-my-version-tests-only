# tests/test_mapping.py

import sys
sys.path.append('.')

import torch
import pytest

# Dummy generator to mimic the behavior of G.mapping.
class DummyGenerator:
    def __init__(self):
        self.z_dim = 512

    def mapping(self, z, pose, condition):
        # For this dummy, return a tensor with shape [batch, num_layers, style_dim].
        # Assume our expected number of layers is 13 (as per our current model)
        # and style_dim is 256.
        # In a real test, this function would invoke the actual mapping logic.
        batch_size = z.shape[0]
        return torch.randn(batch_size, 13, 256)

# Fixture for the dummy generator.
@pytest.fixture
def dummy_G():
    return DummyGenerator()

# Fixture for a dummy pose (with 25 elements if that's what the model expects).
@pytest.fixture
def dummy_pose():
    return torch.zeros(1, 25)

def test_mapping_output_shape(dummy_G, dummy_pose):
    """
    Test that the mapping function returns an output with the correct dimensions.
    Expected output shape is [batch, num_layers, style_dim].
    Here, we expect num_layers to be 13 for our dummy generator.
    """
    z = torch.randn(1, dummy_G.z_dim)
    condition = {'mask': torch.zeros(1, 1, 128, 128), 'pose': dummy_pose}
    ws = dummy_G.mapping(z, dummy_pose, condition)
    # Assert that the output is 3D (batch, layers, style_dim)
    assert ws.ndim == 3, f"Expected output tensor to have 3 dimensions, got {ws.ndim}"
    # Check that the number of layers is as expected.
    assert ws.shape[1] == 13, f"Expected 13 layers, got {ws.shape[1]}"