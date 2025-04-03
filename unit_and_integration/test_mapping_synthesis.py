import sys
sys.path.append('.')

import torch
import numpy as np

def test_mapping_output_shape(dummy_G, dummy_pose):
    # Here dummy_G is a fixture that loads a small dummy network for testing.
    # For demonstration, assume dummy_G.mapping takes (z, pose, condition)
    # and returns a tensor of shape [1, 13, 256] (since l=13).
    z = torch.randn(1, dummy_G.z_dim)
    condition = {'mask': torch.zeros(1, 1, 128, 128), 'pose': dummy_pose}
    ws = dummy_G.mapping(z, dummy_pose, condition)
    # Assert that ws has the expected dimensions: [batch, num_layers, style_dim]
    assert ws.ndim == 3
    assert ws.shape[1] == 13  # Updated expectation: 13 style vectors, not 25.

def test_synthesis_output(dummy_G, dummy_pose):
    z = torch.randn(1, dummy_G.z_dim)
    condition = {'mask': torch.zeros(1,1,128,128), 'pose': dummy_pose}
    ws = dummy_G.mapping(z, dummy_pose, condition)
    out = dummy_G.synthesis(ws, dummy_pose, noise_mode='const', neural_rendering_resolution=128)
    # Check that out['image'] is a tensor of expected shape, e.g., [1, 3, H, W]
    assert out['image'].ndim == 4
    assert out['image'].shape[1] == 3