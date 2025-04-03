# tests/test_dataset.py
import sys
sys.path.append('.')

import pytest
import numpy as np
import PIL.Image
import torch
from training.dataset import ImageFolderDataset  # Ensure your PYTHONPATH is set correctly

def test_image_folder_dataset_loading(tmp_path):
    # Create a dummy directory with one image.
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    dummy_image = PIL.Image.new("RGB", (128, 128), color="red")
    dummy_image.save(image_dir / "dummy.png")
    
    # Instantiate the dataset
    dataset = ImageFolderDataset(path=str(image_dir), resolution=128)
    assert len(dataset) == 1
    image, label = dataset[0]
    # Check the image shape and type
    assert isinstance(image, np.ndarray)
    assert image.shape[0] == 3  # channels