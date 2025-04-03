# test_unit2.py

import pytest
import random
import os

# Dummy unit functions
def apply_seed(seed):
    random.seed(seed)
    return random.randint(0, 100000)

def validate_input_id(input_id):
    try:
        val = int(input_id)
        return val >= 0
    except ValueError:
        return False

def extract_base_name(filename):
    return os.path.splitext(os.path.basename(filename))[0]

def normalize_cfg(cfg):
    return cfg.strip().lower()

def is_env_ready(checkpoint_path):
    return os.path.exists(checkpoint_path)

# Unit tests

def test_apply_seed_reproducibility():
    val1 = apply_seed(42)
    val2 = apply_seed(42)
    assert val1 == val2

def test_input_id_validation_good():
    assert validate_input_id("123")

def test_input_id_validation_bad():
    assert not validate_input_id("abc")

def test_extract_basename():
    assert extract_base_name("/path/to/seg2cat_1.png") == "seg2cat_1"

def test_env_ready_fails_for_fake_path():
    assert not is_env_ready("nonexistent/file.pkl")
