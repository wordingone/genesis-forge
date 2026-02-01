"""
A/B Test: Compare test script path vs GUI path EXACTLY.
Runs both in the same Python session to eliminate environmental differences.

This will definitively show if the difference is in:
1. Parameter passing (sampler params)
2. Export params
3. Something else (CUDA state, memory, etc.)
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

    # Face count
    face_count = len(geom.faces)

    # Degenerate faces (area < 1e-10)
    v = geom.vertices[geom.faces]
    areas = np.linalg.norm(np.cross(v[:,1]-v[:,0], v[:,2]-v[:,0]), axis=1) / 2
    degenerate_area = (areas < 1e-10).sum()

    # Degenerate faces (edge < 1e-5)
    edge1 = np.linalg.norm(v[:,1] - v[:,0], axis=1)
    edge2 = np.linalg.norm(v[:,2] - v[:,1], axis=1)
    edge3 = np.linalg.norm(v[:,0] - v[:,2], axis=1)
    degenerate_edge = ((edge1 < 1e-5) | (edge2 < 1e-5) | (edge3 < 1e-5)).sum()

    # File size
    file_size_mb = glb_path.stat().st_size / (1024 * 1024)

    # Texture size
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
    output_dir = Path(r"B:\M\ArtificialArchitecture\spatial\trellis-forge\outputs\benchmarks\ab_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("A/B TEST: Test Script Path vs GUI Path")
    print("=" * 80)

    # Load pipeline
    print("\n[1] Loading pipeline...")
    model_path = r"B:\M\ArtificialArchitecture\spatial\trellis-forge\models\hub\TRELLIS.2-4B"
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(model_path)
    pipeline.cuda()

    # Load image
    image = Image.open(r"B:\M\ArtificialArchitecture\spatial\trellis-forge\reference\spiral_input.png")

    # ========================================================================
    # PATH A: Test Script Path (NO explicit params)
    # ========================================================================
    print("\n" + "=" * 80)
    print("[A] TEST SCRIPT PATH (NO explicit params)")
    print("=" * 80)

    # Clear CUDA cache
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # Set seeds BEFORE (like test script does)
    torch.manual_seed(42)
    np.random.seed(42)

    # Run with NO explicit sampler params
    print("Running pipeline.run(image, seed=42) with NO sampler params...")
    out_mesh_a = pipeline.run(image, seed=42)
    if isinstance(out_mesh_a, list):
        out_mesh_a = out_mesh_a[0]

    # Export with ONLY verbose (all other params use defaults)
    output_a = output_dir / f"path_A_testscript_{timestamp}.glb"
    print(f"Exporting with DEFAULTS: {output_a}")
    out_mesh_a.export(str(output_a), verbose=True)

    # ========================================================================
    # PATH B: GUI Path (EXPLICIT params, same values as defaults)
    # ========================================================================
    print("\n" + "=" * 80)
    print("[B] GUI PATH (EXPLICIT params)")
    print("=" * 80)

    # Clear CUDA cache (fresh state like GUI would have)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # Set seed only via pipeline.run (like GUI does)
    # NOTE: GUI does NOT do torch.manual_seed or np.random.seed before pipeline.run

    # GUI params (same values as pipeline.json defaults)
    sparse_params = {
        "steps": 12,
        "guidance_strength": 7.5,
        "guidance_rescale": 0.7,
        "rescale_t": 5.0,
        # NOTE: guidance_interval NOT passed
    }
    shape_params = {
        "steps": 12,
        "guidance_strength": 7.5,
        "guidance_rescale": 0.5,
        "rescale_t": 3.0,
        # NOTE: guidance_interval NOT passed
    }
    tex_params = {
        "steps": 12,
        "guidance_strength": 1.0,
        "guidance_rescale": 0.0,
        "rescale_t": 3.0,
        # NOTE: guidance_interval NOT passed
    }

    print("Running pipeline.run(image, seed=42, sparse_structure_sampler_params=..., ...)")
    print(f"  sparse_params: {sparse_params}")
    print(f"  shape_params: {shape_params}")
    print(f"  tex_params: {tex_params}")

    out_mesh_b = pipeline.run(
        image,
        seed=42,
        sparse_structure_sampler_params=sparse_params,
        shape_slat_sampler_params=shape_params,
        tex_slat_sampler_params=tex_params,
        pipeline_type="1024_cascade",
    )
    if isinstance(out_mesh_b, list):
        out_mesh_b = out_mesh_b[0]

    # Export with EXPLICIT params (same values as defaults)
    output_b = output_dir / f"path_B_guipath_{timestamp}.glb"
    print(f"Exporting with EXPLICIT params: {output_b}")
    out_mesh_b.export(
        str(output_b),
        texture_size=2048,
        decimation_target=300000,
        remesh=True,
        remesh_band=1.0,
        remesh_project=0.0,
        verbose=True,
    )

    # ========================================================================
    # COMPARISON
    # ========================================================================
    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)

    metrics_a = analyze_mesh(output_a)
    metrics_b = analyze_mesh(output_b)

    print("\n{:<30} {:>15} {:>15}".format("Metric", "Path A (Test)", "Path B (GUI)"))
    print("-" * 60)
    print("{:<30} {:>15,} {:>15,}".format("Face count", metrics_a['faces'], metrics_b['faces']))
    print("{:<30} {:>15,} {:>15,}".format("Degenerate (area<1e-10)", metrics_a['degenerate_area'], metrics_b['degenerate_area']))
    print("{:<30} {:>15,} {:>15,}".format("Degenerate (edge<1e-5)", metrics_a['degenerate_edge'], metrics_b['degenerate_edge']))
    print("{:<30} {:>15.2f} {:>15.2f}".format("File size (MB)", metrics_a['file_size_mb'], metrics_b['file_size_mb']))
    print("{:<30} {:>15} {:>15}".format("Texture size", str(metrics_a['texture_size']), str(metrics_b['texture_size'])))

    print("\n" + "=" * 80)
    print("OUTPUT FILES")
    print("=" * 80)
    print(f"Path A (Test Script): {output_a}")
    print(f"Path B (GUI):         {output_b}")
    print("\nVisually compare these files to see if there's a quality difference.")

    # Check if files are identical in size
    size_diff = abs(metrics_a['file_size_mb'] - metrics_b['file_size_mb'])
    if size_diff < 0.01:
        print("\nFiles are nearly identical in size - likely same output.")
    else:
        print(f"\nFiles differ by {size_diff:.2f} MB - investigate visual differences.")

if __name__ == "__main__":
    main()
