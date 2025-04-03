import sys
sys.path.append('.')

import pytest
import torch

# Dummy generator for testing
class DummyGenerator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Set the latent dimension to match your model's z_dim (adjust if necessary)
        self.z_dim = 512

    def mapping(self, z, pose, cond):
        # Return a dummy style vector; adjust the dimensions as needed.
        # For example, if your network expects a style vector of shape (1, 13, 256)
        return torch.zeros(1, 13, 256)

    def synthesis(self, ws, pose, noise_mode, neural_rendering_resolution):
        # Create dummy outputs: a color image and semantic tensor.
        # Adjust the number of channels according to your configuration.
        dummy_img = torch.zeros(1, 3, neural_rendering_resolution, neural_rendering_resolution)
        # For seg2face or seg2cat, semantic might have 6 channels, for example.
        dummy_sem = torch.zeros(1, 6, neural_rendering_resolution, neural_rendering_resolution)
        return {"image": dummy_img, "semantic": dummy_sem}

@pytest.fixture
def dummy_G():
    return DummyGenerator()

@pytest.fixture
def dummy_pose():
    # Create a dummy pose vector. In your code, custom input poses have shape (1, 25).
    return torch.zeros(1, 25)