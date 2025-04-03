import sys
sys.path.append('.')

import random
import subprocess
from pathlib import Path
import PIL.Image
import numpy as np

# Define an operational profile: each tuple is (cfg, weight).
OP_profile = [
    ("seg2face", 0.6),
    ("seg2cat", 0.3),
    ("edge2car", 0.1)
]

def run_pipeline(cfg):
    outdir = Path("temp_output")  # use a temp output folder
    outdir.mkdir(exist_ok=True)
    cmd = [
        "python", "applications/generate_samples.py",
        "--network", f"checkpoints/pix2pix3d_{cfg}.pkl",
        "--outdir", str(outdir),
        "--random_seed", "1",
        "--input_id", "0",
        "--cfg", cfg
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode, outdir

def test_usage_profile():
    # Randomly select a configuration based on weights
    cfg = random.choices(
        [profile[0] for profile in OP_profile],
        weights=[profile[1] for profile in OP_profile],
        k=1
    )[0]
    retcode, outdir = run_pipeline(cfg)
    assert retcode == 0, f"Pipeline failed for {cfg}"
    # Check that at least one output file is created.
    files = list(outdir.glob(f"{cfg}_*_color.png"))
    assert len(files) > 0, f"No output files generated for {cfg}"