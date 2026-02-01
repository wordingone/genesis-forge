"""
Verification test: Does the test script still produce good output?
Run with: .\venv311\Scripts\python.exe tests\diagnostics\verify_test_script_20260131.py

This replicates quick_verify.py but with timestamped output for comparison.
"""

import sys
sys.path.insert(0, r"B:\M\ArtificialArchitecture\spatial\trellis-forge")

import torch
import numpy as np
import trimesh
from pathlib import Path
from datetime import datetime

torch.set_grad_enabled(False)

def run_verify():
    from trellis2.pipelines import Trellis2ImageTo3DPipeline

    # Generate unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = f"verify_testscript_{timestamp}"

    print(f"=== Test Script Verification ===")
    print(f"Output name: {output_name}")
    print(f"Purpose: Verify test script STILL produces good output")
    print()

    print("Loading pipeline...")
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(
        r"B:\M\ArtificialArchitecture\spatial\trellis-forge\models\hub\TRELLIS.2-4B"
    )
    pipeline.cuda()

    print("\nRunning generation (NO explicit sampler params - uses 100% internal defaults)...")
    from PIL import Image
    image = Image.open(r"B:\M\ArtificialArchitecture\spatial\trellis-forge\reference\spiral_input.png")

    torch.manual_seed(42)
    np.random.seed(42)

    # KEY: No explicit sampler params - exactly like quick_verify.py
    out_mesh = pipeline.run(image, seed=42)
    if isinstance(out_mesh, list):
        out_mesh = out_mesh[0]

    # Export with DEFAULTS (texture_size=2048, decimation_target=300000)
    output_dir = Path(r"B:\M\ArtificialArchitecture\spatial\trellis-forge\outputs\benchmarks\verification")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{output_name}.glb"

    print(f"\nExporting with DEFAULT parameters (texture_size=2048, decimation_target=300000)...")
    print(f"Output: {output_path}")
    out_mesh.export(str(output_path), verbose=True)

    # Verify output
    print("\n" + "="*70)
    print("VERIFICATION RESULTS")
    print("="*70)

    final = trimesh.load(str(output_path))
    geom = list(final.geometry.values())[0]

    print(f"Vertices: {len(geom.vertices)}")
    print(f"Faces: {len(geom.faces)}")

    # File size
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"File size: {file_size_mb:.2f} MB")

    # Count degenerate faces
    v = geom.vertices[geom.faces]
    areas = np.linalg.norm(np.cross(v[:,1]-v[:,0], v[:,2]-v[:,0]), axis=1) / 2
    degenerate = (areas < 1e-10).sum()
    print(f"Degenerate faces (area < 1e-10): {degenerate}")

    # Also check with edge-based threshold (like o_voxel uses)
    v0 = geom.vertices[geom.faces[:, 0]]
    v1 = geom.vertices[geom.faces[:, 1]]
    v2 = geom.vertices[geom.faces[:, 2]]
    edge1 = np.linalg.norm(v1 - v0, axis=1)
    edge2 = np.linalg.norm(v2 - v1, axis=1)
    edge3 = np.linalg.norm(v0 - v2, axis=1)
    edge_degenerate = ((edge1 < 1e-5) | (edge2 < 1e-5) | (edge3 < 1e-5)).sum()
    print(f"Degenerate faces (edge < 1e-5): {edge_degenerate}")

    # Texture size
    if hasattr(geom, 'visual') and hasattr(geom.visual, 'material'):
        mat = geom.visual.material
        if hasattr(mat, 'baseColorTexture') and mat.baseColorTexture is not None:
            print(f"Texture size: {mat.baseColorTexture.size}")

    # Compare with reference files
    print(f"\n--- Comparison with reference files ---")

    # verify_defaults.glb (the "perfect" one)
    verify_defaults_path = Path(r"B:\M\ArtificialArchitecture\spatial\trellis-forge\outputs\benchmarks\trace_test\verify_defaults.glb")
    if verify_defaults_path.exists():
        vd = trimesh.load(str(verify_defaults_path))
        vd_geom = list(vd.geometry.values())[0]
        vd_v = vd_geom.vertices[vd_geom.faces]
        vd_areas = np.linalg.norm(np.cross(vd_v[:,1]-vd_v[:,0], vd_v[:,2]-vd_v[:,0]), axis=1) / 2
        vd_degen = (vd_areas < 1e-10).sum()
        print(f"verify_defaults.glb: {len(vd_geom.faces)} faces, {vd_degen} degenerate, {verify_defaults_path.stat().st_size / (1024*1024):.2f} MB")

    # Official reference
    official_path = Path(r"B:\M\ArtificialArchitecture\spatial\trellis-forge\reference\spiral_official.glb")
    if official_path.exists():
        off = trimesh.load(str(official_path))
        off_geom = list(off.geometry.values())[0]
        off_v = off_geom.vertices[off_geom.faces]
        off_areas = np.linalg.norm(np.cross(off_v[:,1]-off_v[:,0], off_v[:,2]-off_v[:,0]), axis=1) / 2
        off_degen = (off_areas < 1e-10).sum()
        print(f"spiral_official.glb: {len(off_geom.faces)} faces, {off_degen} degenerate")

    print(f"\nOutput saved to: {output_path}")
    print(f"\nVisually inspect this file to see if it has black bars or artifacts.")


if __name__ == "__main__":
    run_verify()
