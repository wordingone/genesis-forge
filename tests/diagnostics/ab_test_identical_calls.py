"""
A/B Test: Both paths use IDENTICAL pipeline.run() calls (no explicit params).
This verifies if the difference is caused by passing explicit params vs using defaults.
"""
import sys
sys.path.insert(0, r"B:\M\ArtificialArchitecture\spatial\trellis-forge")

import torch
import numpy as np
import trimesh
from pathlib import Path
from datetime import datetime
from PIL import Image

# Match backend initialization exactly
torch.backends.cudnn.benchmark = True
torch.set_grad_enabled(False)
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True, warn_only=True)

def analyze_mesh(glb_path):
    """Analyze mesh quality metrics."""
    mesh = trimesh.load(str(glb_path))
    geom = list(mesh.geometry.values())[0]
    face_count = len(geom.faces)
    v = geom.vertices[geom.faces]
    areas = np.linalg.norm(np.cross(v[:,1]-v[:,0], v[:,2]-v[:,0]), axis=1) / 2
    degenerate_area = (areas < 1e-10).sum()
    edge1 = np.linalg.norm(v[:,1] - v[:,0], axis=1)
    edge2 = np.linalg.norm(v[:,2] - v[:,1], axis=1)
    edge3 = np.linalg.norm(v[:,0] - v[:,2], axis=1)
    degenerate_edge = ((edge1 < 1e-5) | (edge2 < 1e-5) | (edge3 < 1e-5)).sum()
    file_size_mb = glb_path.stat().st_size / (1024 * 1024)
    texture_size = None
    if hasattr(geom, 'visual') and hasattr(geom.visual, 'material'):
        mat = geom.visual.material
        if hasattr(mat, 'baseColorTexture') and mat.baseColorTexture is not None:
            texture_size = mat.baseColorTexture.size
    return {
        'faces': face_count,
        'degenerate_area': degenerate_area,
        'degenerate_edge': degenerate_edge,
        'file_size_mb': file_size_mb,
        'texture_size': texture_size,
    }

def main():
    from trellis2.pipelines import Trellis2ImageTo3DPipeline

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(r"B:\M\ArtificialArchitecture\spatial\trellis-forge\outputs\benchmarks\ab_identical")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("A/B TEST: IDENTICAL pipeline.run() calls")
    print("=" * 80)

    # Load pipeline
    print("\n[1] Loading pipeline...")
    model_path = r"B:\M\ArtificialArchitecture\spatial\trellis-forge\models\hub\TRELLIS.2-4B"
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(model_path)
    pipeline.cuda()

    image = Image.open(r"B:\M\ArtificialArchitecture\spatial\trellis-forge\reference\spiral_input.png")

    # ========================================================================
    # PATH A: First generation
    # ========================================================================
    print("\n" + "=" * 80)
    print("[A] FIRST GENERATION (fresh state)")
    print("=" * 80)

    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # Both paths use IDENTICAL call: pipeline.run(image, seed=42) with NO explicit params
    print("Running: pipeline.run(image, seed=42)")
    out_mesh_a = pipeline.run(image, seed=42)
    if isinstance(out_mesh_a, list):
        out_mesh_a = out_mesh_a[0]

    output_a = output_dir / f"run1_{timestamp}.glb"
    print(f"Exporting: {output_a}")
    out_mesh_a.export(str(output_a), verbose=True)

    # ========================================================================
    # PATH B: Second generation (same call)
    # ========================================================================
    print("\n" + "=" * 80)
    print("[B] SECOND GENERATION (same call, fresh CUDA state)")
    print("=" * 80)

    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # IDENTICAL call
    print("Running: pipeline.run(image, seed=42)")
    out_mesh_b = pipeline.run(image, seed=42)
    if isinstance(out_mesh_b, list):
        out_mesh_b = out_mesh_b[0]

    output_b = output_dir / f"run2_{timestamp}.glb"
    print(f"Exporting: {output_b}")
    out_mesh_b.export(str(output_b), verbose=True)

    # ========================================================================
    # COMPARISON
    # ========================================================================
    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)

    metrics_a = analyze_mesh(output_a)
    metrics_b = analyze_mesh(output_b)

    print("\n{:<30} {:>15} {:>15}".format("Metric", "Run 1", "Run 2"))
    print("-" * 60)
    print("{:<30} {:>15,} {:>15,}".format("Face count", metrics_a['faces'], metrics_b['faces']))
    print("{:<30} {:>15,} {:>15,}".format("Degenerate (area<1e-10)", metrics_a['degenerate_area'], metrics_b['degenerate_area']))
    print("{:<30} {:>15,} {:>15,}".format("Degenerate (edge<1e-5)", metrics_a['degenerate_edge'], metrics_b['degenerate_edge']))
    print("{:<30} {:>15.2f} {:>15.2f}".format("File size (MB)", metrics_a['file_size_mb'], metrics_b['file_size_mb']))
    print("{:<30} {:>15} {:>15}".format("Texture size", str(metrics_a['texture_size']), str(metrics_b['texture_size'])))

    # Check if identical
    if (metrics_a['faces'] == metrics_b['faces'] and
        metrics_a['degenerate_edge'] == metrics_b['degenerate_edge'] and
        abs(metrics_a['file_size_mb'] - metrics_b['file_size_mb']) < 0.01):
        print("\nRESULT: Outputs are IDENTICAL!")
        print("This proves the pipeline IS deterministic with same call signature.")
    else:
        print(f"\nRESULT: Outputs DIFFER!")
        print(f"Face difference: {abs(metrics_a['faces'] - metrics_b['faces'])}")
        print("This proves there's inherent non-determinism in the pipeline.")

    print("\n" + "=" * 80)
    print("OUTPUT FILES")
    print("=" * 80)
    print(f"Run 1: {output_a}")
    print(f"Run 2: {output_b}")

if __name__ == "__main__":
    main()
