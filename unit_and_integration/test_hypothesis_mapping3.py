# tests/test_hypothesis_mapping3.py

import sys
sys.path.append('.')

import torch
import pytest
from hypothesis import given, settings, HealthCheck, strategies as st

# We define dummy_G and dummy_pose as module-scoped fixtures so that they persist
# for the whole module instead of being recreated per test example.
@pytest.fixture(scope="module")
def dummy_G():
    # This DummyGenerator mimics the behavior of the real generator's mapping function.
    # For the purposes of testing, we assume:
    # - z_dim is the dimensionality of the input latent code.
    # - mapping() returns a tensor of shape [batch, num_layers, style_dim].
    # In our test, we expect 25 layers and a style dimension of 256.
    class DummyGenerator:
        def __init__(self):
            self.z_dim = 512
        def mapping(self, z, pose, condition):
            # For testing, we simply return a random tensor with shape [1, 25, 256].
            return torch.randn(1, 25, 256)
    return DummyGenerator()

@pytest.fixture(scope="module")
def dummy_pose():
    # Create a dummy pose tensor of shape [1, 25]
    return torch.zeros(1, 25)

# Use settings to suppress the health check regarding function-scoped fixtures.
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(seed=st.integers(min_value=0, max_value=100))
def test_mapping_output_shape_hypothesis(dummy_G, dummy_pose, seed):
    """
    Test that the mapping function of the dummy generator returns a tensor
    with the expected shape: [batch, num_layers, style_dim]. We use Hypothesis
    to provide a random seed and verify consistency over many random examples.
    """
    torch.manual_seed(seed)
    z = torch.randn(1, dummy_G.z_dim)
    # Define a dummy condition dictionary containing a zero mask and the dummy pose.
    condition = {'mask': torch.zeros(1, 1, 128, 128), 'pose': dummy_pose}
    ws = dummy_G.mapping(z, dummy_pose, condition)
    
    # Check that the output has three dimensions
    assert ws.ndim == 3, f"Expected output tensor to have 3 dimensions, got {ws.ndim}"
    # Check that the second dimension (number of layers) is 25
    assert ws.shape[1] == 25, f"Expected 25 layers, got {ws.shape[1]}"
    # Optionally, check that the third dimension (style dimension) is 256
    assert ws.shape[2] == 256, f"Expected style dimension of 256, got {ws.shape[2]}"