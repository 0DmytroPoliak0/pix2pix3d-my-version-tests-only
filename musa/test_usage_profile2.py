# tests/test_usage_profile.py
import sys
sys.path.append('.')

import numpy as np
import pytest

def get_operational_profile():
    # Define probabilities for each category
    # e.g., cars: 0.5, faces: 0.3, cats: 0.2
    return {'edge2car': 0.5, 'seg2face': 0.3, 'seg2cat': 0.2}

def test_operational_profile_sum():
    profile = get_operational_profile()
    total = sum(profile.values())
    # Check that the total probability sums to 1 (or is very close, accounting for float precision)
    np.testing.assert_almost_equal(total, 1.0, decimal=5)

def sample_test_case():
    profile = get_operational_profile()
    categories = list(profile.keys())
    probabilities = list(profile.values())
    # Sample a test case based on the operational profile
    return np.random.choice(categories, p=probabilities)

def test_sample_test_case():
    # Run the sampling multiple times to ensure all categories are eventually chosen
    samples = [sample_test_case() for _ in range(100)]
    assert "edge2car" in samples
    assert "seg2face" in samples
    assert "seg2cat" in samples