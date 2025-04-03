import sys
sys.path.append('.')

import subprocess
import sys
import os
from pathlib import Path
import pytest
import trimesh

@pytest.mark.integration
def test_extracted_mesh(tmp_path):
    """
    Integration test for 3D mesh extraction:
    
    1. Run the extract_mesh.py script using a known configuration.
    2. Check that the output mesh file (e.g. semantic_mesh.ply) is created.
    3. Load the mesh using trimesh and verify that it contains vertices and faces.
    """
    outdir = tmp_path / "mesh_output"
    outdir.mkdir()
    
    cmd = [
        "python", "applications/extract_mesh.py",
        "--network", "checkpoints/pix2pix3d_seg2cat.pkl",
        "--outdir", str(outdir),
        "--cfg", "seg2cat",
        "--input", "tests/sample_inputs/seg2cat_sample.png"  # ensure this exists or adjust as needed
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"Mesh extraction failed: {result.stderr}"
    
    mesh_path = outdir / "semantic_mesh.ply"
    assert mesh_path.exists(), "Extracted mesh file not found."
    
    # Load the mesh with trimesh.
    mesh = trimesh.load(str(mesh_path))
    assert mesh.vertices.size > 0, "Mesh has no vertices."
    assert mesh.faces.size > 0, "Mesh has no faces."
    print(f"Mesh extracted successfully: {mesh_path}")
    print(f"Vertices: {len(mesh.vertices)}, Faces: {len(mesh.faces)}")