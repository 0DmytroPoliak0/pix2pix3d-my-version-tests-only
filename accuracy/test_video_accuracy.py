import sys
sys.path.append('.')

import subprocess
import sys
import os
from pathlib import Path
import pytest

@pytest.mark.integration
def test_generated_video(tmp_path):
    """
    Integration test for video generation:
    1. Run the generate_video.py script using a known configuration.
    2. Check that the expected video file (e.g. edge2car_0_1_color.gif) is created.
    3. Verify that the file size is non-zero and the file exists.
    """
    outdir = tmp_path / "video_output"
    outdir.mkdir()
    
    cmd = [
        "python", "applications/generate_video.py",
        "--network", "checkpoints/pix2pix3d_edge2car.pkl",
        "--outdir", str(outdir),
        "--random_seed", "1",
        "--cfg", "edge2car",
        "--input", "tests/sample_inputs/edge2car_sample.png"  # ensure this exists or adjust as needed
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"Video generation failed: {result.stderr}"
    
    video_path = outdir / "edge2car_0_1_color.gif"
    assert video_path.exists(), "Generated video file not found."
    file_size = video_path.stat().st_size
    assert file_size > 0, "Generated video file is empty."
    print(f"Video generated successfully: {video_path} ({file_size} bytes)")