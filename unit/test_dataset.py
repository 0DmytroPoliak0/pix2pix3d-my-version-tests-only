import sys
sys.path.append('.')

import os
import pytest
import numpy as np
import PIL.Image
from training.dataset import ImageFolderDataset
from pathlib import Path

# A sample test dataset: we create a temporary folder with a couple of dummy images.
@pytest.fixture
def dummy_dataset(tmp_path):
    # Create a dummy directory structure with one image file.
    image_dir = tmp_path / "dummy_images"
    image_dir.mkdir()
    img_path = image_dir / "dummy.png"
    # Create a 64x64 black image.
    PIL.Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(img_path)
    return image_dir

def test_image_folder_dataset_loading(dummy_dataset):
    # Test that ImageFolderDataset loads images correctly.
    ds = ImageFolderDataset(path=str(dummy_dataset))
    # Check that one image is loaded.
    assert len(ds) == 1
    image, label = ds[0]
    # Image shape: expecting 3 channels and dimensions matching the dummy image.
    assert image.shape[0] == 3
    assert image.shape[1] == 64
    assert image.shape[2] == 64