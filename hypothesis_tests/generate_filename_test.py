import sys
sys.path.append('.')

import pytest
from hypothesis import given, strategies as st

# Example implementation of the generate_filename function.
def generate_filename(input_id, seed, config):
    # In our actual project, this function is defined elsewhere.
    # Here is a simplified version for illustration.
    return f"{config}_{input_id}_{seed}_color.png"

@given(
    input_id=st.integers(min_value=0, max_value=10000),
    seed=st.integers(min_value=0, max_value=1000),
    config=st.text(min_size=1, max_size=10)
)
def test_generate_filename(input_id, seed, config):
    filename = generate_filename(input_id, seed, config)
    # Verify that the filename starts with the config string,
    # contains the input_id and seed, and ends with '_color.png'
    assert filename.startswith(config), f"Filename {filename} does not start with config {config}"
    assert f"_{input_id}_" in filename, f"Filename {filename} does not contain input_id {input_id}"
    assert f"_{seed}_" in filename, f"Filename {filename} does not contain seed {seed}"
    assert filename.endswith("_color.png"), f"Filename {filename} does not end with '_color.png'"