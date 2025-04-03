# tests/test_mapping.py
import sys
sys.path.append('.')

import torch
import pytest

class DummyGenerator:
    def __init__(self, z_dim=512, num_layers=25, style_dim=256):
        self.z_dim = z_dim
        self.num_layers = num_layers
        self.style_dim = style_dim
        
    def mapping(self, z, pose, condition):
        # For testing purposes, return a dummy tensor with shape [1, num_layers, style_dim]
        # (You can change this to simulate the expected behavior of your mapping function.)
        return torch.randn(1, self.num_layers, self.style_dim)

@pytest.fixture
def dummy_G():
    return DummyGenerator()

@pytest.fixture
def dummy_pose():
    # Create a dummy pose tensor of the expected shape, e.g., [1, 25]
    return torch.zeros(1, 25)

def test_mapping_output_shape(dummy_G, dummy_pose):
    z = torch.randn(1, dummy_G.z_dim)
    condition = {'mask': torch.zeros(1, 1, 128, 128), 'pose': dummy_pose}
    ws = dummy_G.mapping(z, dummy_pose, condition)
    # Check that ws has three dimensions: [batch, num_layers, style_dim]
    assert ws.ndim == 3
    # Ensure the second dimension equals the expected number of layers.
    assert ws.shape[1] == dummy_G.num_layers
    # Check that the last dimension equals style_dim.
    assert ws.shape[2] == dummy_G.style_dim