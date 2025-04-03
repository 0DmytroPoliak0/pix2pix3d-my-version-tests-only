# tests/test_synthesis.py
import sys
sys.path.append('.')

import torch
import pytest

class DummyGeneratorWithSynthesis:
    def __init__(self, z_dim=512, num_layers=25, style_dim=256):
        self.z_dim = z_dim
        self.num_layers = num_layers
        self.style_dim = style_dim
    
    def mapping(self, z, pose, condition):
        return torch.randn(1, self.num_layers, self.style_dim)
    
    def synthesis(self, ws, pose, noise_mode='const', neural_rendering_resolution=128):
        # For testing, simulate a dummy output with expected image shape, e.g., [1, 3, 128, 128]
        image = torch.randn(1, 3, neural_rendering_resolution, neural_rendering_resolution)
        # And simulate a semantic output, e.g., with 6 classes for seg2cat:
        semantic = torch.randn(1, 6, neural_rendering_resolution, neural_rendering_resolution)
        return {'image': image, 'semantic': semantic}

@pytest.fixture
def dummy_G_synthesis():
    return DummyGeneratorWithSynthesis()

@pytest.fixture
def dummy_pose():
    return torch.zeros(1, 25)

def test_synthesis_output(dummy_G_synthesis, dummy_pose):
    z = torch.randn(1, dummy_G_synthesis.z_dim)
    condition = {'mask': torch.zeros(1, 1, 128, 128), 'pose': dummy_pose}
    ws = dummy_G_synthesis.mapping(z, dummy_pose, condition)
    out = dummy_G_synthesis.synthesis(ws, dummy_pose, neural_rendering_resolution=128)
    # Check image output shape
    assert out['image'].shape == (1, 3, 128, 128)
    # Check semantic output shape (assuming 6 classes for seg2cat)
    assert out['semantic'].shape == (1, 6, 128, 128)