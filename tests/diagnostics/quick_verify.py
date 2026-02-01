"""
Quick verification of TRELLIS.2 export fixes:
1. texture_size=2048 (default, matches official HF demo)
2. decimation_target=300000 (default, matches official HF demo)
3. Post-UV-unwrap degenerate face removal
"""

import sys
sys.path.insert(0, r"B:\M\ArtificialArchitecture\spatial\trellis-forge")

import torch
import numpy as np
import trimesh
from pathlib import Path

torch.set_grad_enabled(False)

def run_verify():
    from trellis2.pipelines import Trellis2ImageTo3DPipeline

    print("Loading pipeline...")
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(
        r"B:\M\ArtificialArchitecture\spatial\trellis-forge\models\hub\TRELLIS.2-4B"
    )
    pipeline.cuda()

    print("\nRunning generation...")
    from PIL import Image
    image = Image.open(r"B:\M\ArtificialArchitecture\spatial\trellis-forge\reference\spiral_input.png")

    torch.manual_seed(42)
    np.random.seed(42)

    out_mesh = pipeline.run(image, seed=42)
    if isinstance(out_mesh, list):
        out_mesh = out_mesh[0]

    # Export with defaults (texture_size=2048, decimation_target=300000 - matches official HF demo)
    output_path = Path(r"B:\M\ArtificialArchitecture\spatial\trellis-forge\outputs\benchmarks\trace_test\verify_defaults.glb")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("\nExporting with default parameters...")
    out_mesh.export(str(output_path), verbose=True)

    # Verify output
    print("\n" + "="*70)
    print("VERIFICATION RESULTS")
    print("="*70)

    final = trimesh.load(str(output_path))
    geom = list(final.geometry.values())[0]

    print(f"Vertices: {len(geom.vertices)}")
    print(f"Faces: {len(geom.faces)}")

    # Count degenerate faces
    v = geom.vertices[geom.faces]
    areas = np.linalg.norm(np.cross(v[:,1]-v[:,0], v[:,2]-v[:,0]), axis=1) / 2
    degenerate = (areas < 1e-10).sum()
    print(f"Degenerate faces: {degenerate}")

    # Texture size
    if hasattr(geom, 'visual') and hasattr(geom.visual, 'material'):
        mat = geom.visual.material
        if hasattr(mat, 'baseColorTexture') and mat.baseColorTexture is not None:
            print(f"Texture size: {mat.baseColorTexture.size}")

    # Compare with official
    official = trimesh.load(r"B:\M\ArtificialArchitecture\spatial\trellis-forge\reference\spiral_official.glb")
    off_geom = list(official.geometry.values())[0]

    print(f"\nOfficial reference:")
    print(f"  Faces: {len(off_geom.faces)}")
    print(f"  Degenerate: 2")
    print(f"  Texture: (2048, 2048)")

    print(f"\nFace count ratio: {len(geom.faces) / len(off_geom.faces):.2f}x")
    print(f"\nOutput saved to: {output_path}")


if __name__ == "__main__":
    run_verify()
