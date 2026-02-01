"""Test voxel coverage - check if vertices sample from occupied regions."""
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from pathlib import Path
TRELLIS_ROOT = Path(__file__).parent

import sys
sys.path.insert(0, str(TRELLIS_ROOT))

import torch
import numpy as np
from PIL import Image

# Ensure venv is used
assert "venv311" in sys.executable, f"Wrong Python: {sys.executable}"

def analyze_coverage(mesh, name="mesh"):
    """Analyze how well vertices are covered by occupied voxels."""
    attr_volume = mesh.attrs.cuda()
    coords_vol = mesh.coords.cuda()
    vertices = mesh.vertices.cuda()

    grid_size = round(1.0 / mesh.voxel_size)
    aabb = torch.tensor([[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], device='cuda', dtype=torch.float32)
    voxel_size = (aabb[1] - aabb[0]) / grid_size

    # Build occupancy grid
    coords_int = coords_vol.long()
    occupancy = torch.zeros(grid_size, grid_size, grid_size, device='cuda', dtype=torch.bool)
    valid_coords = (coords_int >= 0).all(dim=1) & (coords_int < grid_size).all(dim=1)
    valid_int = coords_int[valid_coords]
    occupancy[valid_int[:, 0], valid_int[:, 1], valid_int[:, 2]] = True

    # Convert vertices to voxel coords
    vert_voxel = (vertices - aabb[0]) / voxel_size
    vert_voxel_int = vert_voxel.long().clamp(0, grid_size - 1)

    # Check coverage levels
    # Level 0: vertex's nearest voxel is occupied
    level_0 = occupancy[vert_voxel_int[:, 0], vert_voxel_int[:, 1], vert_voxel_int[:, 2]]

    # Level 1+: count occupied corners in trilinear cell
    corner_counts = torch.zeros(vertices.shape[0], device='cuda', dtype=torch.int32)
    for dx in [0, 1]:
        for dy in [0, 1]:
            for dz in [0, 1]:
                corner = (vert_voxel_int + torch.tensor([dx, dy, dz], device='cuda')).clamp(0, grid_size - 1)
                corner_occupied = occupancy[corner[:, 0], corner[:, 1], corner[:, 2]]
                corner_counts += corner_occupied.int()

    print(f"\n{name} Coverage Analysis:")
    print(f"  Total vertices: {vertices.shape[0]}")
    print(f"  Occupied voxels: {occupancy.sum().item()} ({100*occupancy.sum().item()/(grid_size**3):.3f}%)")
    print(f"  Vertices with nearest voxel occupied: {level_0.sum().item()} ({100*level_0.sum().item()/vertices.shape[0]:.1f}%)")

    for i in range(9):
        count = (corner_counts == i).sum().item()
        pct = 100 * count / vertices.shape[0]
        bar = "#" * int(pct / 2)
        print(f"  {i} corners occupied: {count:>8} ({pct:>5.1f}%) {bar}")

    # Key metric: vertices with partial coverage (1-7 corners) - these cause artifacts
    partial = ((corner_counts > 0) & (corner_counts < 8)).sum().item()
    print(f"\n  PARTIAL COVERAGE (1-7 corners): {partial} ({100*partial/vertices.shape[0]:.1f}%)")
    print(f"  These vertices cause texture discontinuities!")

    return corner_counts


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, default='reference/spiral_input.png')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    print("=" * 70)
    print("VOXEL COVERAGE TEST")
    print("=" * 70)

    # Load pipeline
    print("\nLoading pipeline...")
    from trellis2.pipelines.trellis2_image_to_3d import Trellis2ImageTo3DPipeline
    from PIL import Image as PILImage

    MODEL_PATH = TRELLIS_ROOT / 'models' / 'hub' / 'TRELLIS.2-4B'
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(str(MODEL_PATH))
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    pipeline.configure_vram_mode(vram_gb)
    pipeline.to("cuda")

    # Parameters
    SPARSE_PARAMS = {"steps": 12, "guidance_strength": 7.5, "guidance_rescale": 0.7, "rescale_t": 5.0}
    SHAPE_PARAMS = {"steps": 12, "guidance_strength": 7.5, "guidance_rescale": 0.5, "rescale_t": 3.0}
    TEX_PARAMS = {"steps": 12, "guidance_strength": 1.0, "guidance_rescale": 0.0, "rescale_t": 3.0}

    # Load and preprocess image
    print(f"\nProcessing: {args.image}")
    raw_image = PILImage.open(args.image).convert("RGBA")
    preprocessed = pipeline.preprocess_image(raw_image)

    torch.manual_seed(args.seed)

    # Get conditioning
    cond_512 = pipeline.get_cond([preprocessed], 512)
    cond_1024 = pipeline.get_cond([preprocessed], 1024)

    # Run pipeline
    print("\nGenerating mesh...")
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=True):
        coords = pipeline.sample_sparse_structure(
            cond_512, 32, num_samples=1, sampler_params=SPARSE_PARAMS,
        )
        shape_slat, hr_resolution = pipeline.sample_shape_slat_cascade(
            cond_512, cond_1024,
            pipeline.models['shape_slat_flow_model_512'],
            pipeline.models['shape_slat_flow_model_1024'],
            512, 1024, coords, sampler_params=SHAPE_PARAMS,
        )
        tex_slat = pipeline.sample_tex_slat(
            cond_1024, pipeline.models['tex_slat_flow_model_1024'],
            shape_slat, sampler_params=TEX_PARAMS,
        )

    torch.cuda.empty_cache()

    # Decode
    out_meshes = pipeline.decode_latent(shape_slat, tex_slat, hr_resolution)
    mesh = out_meshes[0]

    # Analyze BEFORE any processing
    print("\n" + "=" * 70)
    print("BEFORE REMESHING (Original decoded mesh)")
    print("=" * 70)
    analyze_coverage(mesh, "Original")

    # Now let's see what happens after remeshing/simplification (what to_glb does)
    print("\n" + "=" * 70)
    print("AFTER REMESHING (What to_glb processes)")
    print("=" * 70)

    # Simulate the remeshing that to_glb does
    import cumesh

    vertices = mesh.vertices.cuda()
    faces = mesh.faces.cuda()

    mesh_obj = cumesh.CuMesh()
    mesh_obj.init(vertices, faces)
    mesh_obj.fill_holes(max_hole_perimeter=3e-2)

    # Build BVH on original
    verts_filled, faces_filled = mesh_obj.read()
    bvh = cumesh.cuBVH(verts_filled, faces_filled)

    # Remesh
    grid_size = round(1.0 / mesh.voxel_size)
    aabb = torch.tensor([[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], device='cuda', dtype=torch.float32)
    center = aabb.mean(dim=0)
    scale = (aabb[1] - aabb[0]).max().item()

    mesh_obj.init(*cumesh.remeshing.remesh_narrow_band_dc(
        verts_filled, faces_filled,
        center=center,
        scale=(grid_size + 3 * 1) / grid_size * scale,
        resolution=grid_size,
        band=1,
        project_back=0,
        verbose=False,
        bvh=bvh,
    ))

    # Simplify
    mesh_obj.simplify(500000, verbose=False)
    simp_verts, simp_faces = mesh_obj.read()

    print(f"\nRemeshed mesh: {simp_verts.shape[0]} vertices, {simp_faces.shape[0]} faces")

    # Create a fake mesh object to analyze coverage
    class FakeMesh:
        def __init__(self, verts, coords, attrs, voxel_size):
            self.vertices = verts
            self.coords = coords
            self.attrs = attrs
            self.voxel_size = voxel_size

    fake_mesh = FakeMesh(simp_verts, mesh.coords, mesh.attrs, mesh.voxel_size)
    analyze_coverage(fake_mesh, "Remeshed+Simplified")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("""
The key insight is that vertices with PARTIAL coverage (1-7 occupied corners)
will produce interpolated values that don't match their true neighbors in
UV space. This creates the triangular texture artifacts.

Possible solutions:
1. Densify the voxel grid before sampling (fill in missing neighbors)
2. Use nearest-neighbor sampling for partially covered vertices
3. Project vertices to fully-covered regions before sampling
4. Accept some quality loss and improve the 2D inpainting
""")


if __name__ == "__main__":
    main()
