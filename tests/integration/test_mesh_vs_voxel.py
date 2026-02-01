"""Test if mesh vertex positions correctly map to voxel coordinates."""
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from pathlib import Path
TRELLIS_ROOT = Path(__file__).parent
MODELS_DIR = TRELLIS_ROOT / "models"
os.environ['HF_HOME'] = str(MODELS_DIR)
os.environ['HF_HUB_CACHE'] = str(MODELS_DIR / "hub")
os.environ['TORCH_HOME'] = str(MODELS_DIR / "torch_hub")
os.environ['HF_HUB_OFFLINE'] = '1'

import sys
sys.path.insert(0, str(TRELLIS_ROOT))

import torch
import numpy as np
from PIL import Image
from flex_gemm.ops.grid_sample import grid_sample_3d

# Ensure venv is used
assert "venv311" in sys.executable, f"Wrong Python: {sys.executable}"

def main():
    print("=" * 60)
    print("MESH VERTEX vs VOXEL COORDINATE TEST")
    print("=" * 60)

    # Load the pipeline and run to get real data
    print("\n1. Loading pipeline...")
    from trellis2.pipelines.trellis2_image_to_3d import Trellis2ImageTo3DPipeline

    MODEL_PATH = MODELS_DIR / "hub" / "TRELLIS.2-4B"
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(str(MODEL_PATH))

    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    pipeline.configure_vram_mode(vram_gb)
    pipeline.to("cuda")
    print(f"   Pipeline loaded")

    # Load test image
    print("\n2. Loading test image...")
    image_path = TRELLIS_ROOT / "reference" / "spiral_input.png"
    image = Image.open(str(image_path)).convert("RGBA")
    print(f"   Image: {image.size}")

    # Run pipeline
    print("\n3. Running generation (seed 42)...")
    result = pipeline.run(
        image,
        seed=42,
        return_latent=True,
    )

    out_mesh_list, (shape_slat, tex_slat, res) = result
    out_mesh = out_mesh_list[0]

    vertices = out_mesh.vertices.cuda()  # [V, 3]
    attrs = out_mesh.attrs.cuda()  # [N, C]
    coords = out_mesh.coords.cuda()  # [N, 3]
    layout = out_mesh.layout
    origin = out_mesh.origin.cuda()  # [3]
    voxel_size = out_mesh.voxel_size

    print(f"\n4. Data shapes:")
    print(f"   Vertices: {vertices.shape}")
    print(f"   Voxel coords: {coords.shape}")
    print(f"   Voxel attrs: {attrs.shape}")

    print(f"\n5. Coordinate ranges:")
    print(f"   Mesh vertices X: [{vertices[:, 0].min():.4f}, {vertices[:, 0].max():.4f}]")
    print(f"   Mesh vertices Y: [{vertices[:, 1].min():.4f}, {vertices[:, 1].max():.4f}]")
    print(f"   Mesh vertices Z: [{vertices[:, 2].min():.4f}, {vertices[:, 2].max():.4f}]")
    print(f"   Voxel coords X: [{coords[:, 0].min()}, {coords[:, 0].max()}]")
    print(f"   Voxel coords Y: [{coords[:, 1].min()}, {coords[:, 1].max()}]")
    print(f"   Voxel coords Z: [{coords[:, 2].min()}, {coords[:, 2].max()}]")

    # Convert voxel coords to world coords for comparison
    voxel_world = coords.float() * voxel_size + origin  # [N, 3]
    print(f"\n6. Voxel positions in world coords:")
    print(f"   Voxel world X: [{voxel_world[:, 0].min():.4f}, {voxel_world[:, 0].max():.4f}]")
    print(f"   Voxel world Y: [{voxel_world[:, 1].min():.4f}, {voxel_world[:, 1].max():.4f}]")
    print(f"   Voxel world Z: [{voxel_world[:, 2].min():.4f}, {voxel_world[:, 2].max():.4f}]")

    # Check overlap between mesh vertices and voxel positions
    print("\n" + "=" * 60)
    print("SPATIAL OVERLAP ANALYSIS")
    print("=" * 60)

    mesh_min = vertices.min(0).values
    mesh_max = vertices.max(0).values
    voxel_min = voxel_world.min(0).values
    voxel_max = voxel_world.max(0).values

    print(f"\n   Mesh AABB: [{mesh_min.tolist()}] to [{mesh_max.tolist()}]")
    print(f"   Voxel AABB: [{voxel_min.tolist()}] to [{voxel_max.tolist()}]")

    # Check if mesh is inside voxel volume
    inside = ((mesh_min >= voxel_min).all() and (mesh_max <= voxel_max).all())
    print(f"   Mesh inside voxel volume: {inside}")

    # Sample at mesh vertices and check if we get valid data
    print("\n" + "=" * 60)
    print("MESH VERTEX SAMPLING TEST")
    print("=" * 60)

    n_test = 1000
    test_indices = torch.randperm(vertices.shape[0])[:n_test]
    test_vertices = vertices[test_indices]  # [n_test, 3]

    # Convert to voxel coordinates using postprocess.py method
    aabb = torch.tensor([[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], device='cuda', dtype=torch.float32)
    grid_size = torch.tensor([1024, 1024, 1024], device='cuda', dtype=torch.int32)
    voxel_size_pp = (aabb[1] - aabb[0]) / grid_size.float()

    grid = ((test_vertices - aabb[0]) / voxel_size_pp).reshape(1, -1, 3)

    print(f"\n   Testing {n_test} mesh vertices...")
    print(f"   Grid coord range: [{grid.min():.1f}, {grid.max():.1f}]")

    # Check how many are out of bounds
    oob_low = (grid < 0).any(dim=-1).sum().item()
    oob_high = (grid >= 1024).any(dim=-1).sum().item()
    print(f"   Out of bounds (any dim < 0): {oob_low}")
    print(f"   Out of bounds (any dim >= 1024): {oob_high}")

    # Sample attributes
    shape = torch.Size([1, attrs.shape[1], *grid_size.tolist()])
    coords_4d = torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=-1)  # [N, 4]

    sampled = grid_sample_3d(attrs, coords_4d, shape, grid, mode='trilinear')[0]  # [n_test, C]
    sampled_rgb = sampled[:, layout['base_color']]

    print(f"\n   Sampled RGB stats:")
    print(f"   R: min={sampled_rgb[:, 0].min():.4f}, max={sampled_rgb[:, 0].max():.4f}, mean={sampled_rgb[:, 0].mean():.4f}")
    print(f"   G: min={sampled_rgb[:, 1].min():.4f}, max={sampled_rgb[:, 1].max():.4f}, mean={sampled_rgb[:, 1].mean():.4f}")
    print(f"   B: min={sampled_rgb[:, 2].min():.4f}, max={sampled_rgb[:, 2].max():.4f}, mean={sampled_rgb[:, 2].mean():.4f}")

    # Check how many samples are zero/black (failed lookups)
    zero_samples = (sampled.abs().sum(dim=1) < 0.01).sum().item()
    print(f"\n   Zero/near-zero samples: {zero_samples} / {n_test} ({100*zero_samples/n_test:.1f}%)")

    # Compare with voxel attribute statistics
    voxel_rgb = attrs[:, layout['base_color']]
    print(f"\n   Original voxel RGB stats:")
    print(f"   R: min={voxel_rgb[:, 0].min():.4f}, max={voxel_rgb[:, 0].max():.4f}, mean={voxel_rgb[:, 0].mean():.4f}")
    print(f"   G: min={voxel_rgb[:, 1].min():.4f}, max={voxel_rgb[:, 1].max():.4f}, mean={voxel_rgb[:, 1].mean():.4f}")
    print(f"   B: min={voxel_rgb[:, 2].min():.4f}, max={voxel_rgb[:, 2].max():.4f}, mean={voxel_rgb[:, 2].mean():.4f}")

    # Check if sampled means match voxel means
    print(f"\n   Mean comparison (sampled vs voxel):")
    for ch, name in [(0, 'R'), (1, 'G'), (2, 'B')]:
        sampled_mean = sampled_rgb[:, ch].mean().item()
        voxel_mean = voxel_rgb[:, ch].mean().item()
        diff = abs(sampled_mean - voxel_mean)
        status = "OK" if diff < 0.05 else f"DRIFT ({diff:.4f})"
        print(f"   {name}: Sampled={sampled_mean:.4f}, Voxel={voxel_mean:.4f}, Diff={diff:.4f} {status}")

    # Find the nearest voxel for each mesh vertex and check if colors match
    print("\n" + "=" * 60)
    print("NEAREST NEIGHBOR VERIFICATION")
    print("=" * 60)

    # For a subset, find the nearest voxel and compare
    n_verify = 100
    verify_indices = torch.randperm(vertices.shape[0])[:n_verify]
    verify_vertices = vertices[verify_indices]

    # Convert vertices to voxel coords
    verify_voxel = (verify_vertices - origin) / voxel_size  # [n_verify, 3]

    # Round to nearest integer
    verify_voxel_int = verify_voxel.round().long()  # [n_verify, 3]

    # Find exact matches in coords
    matches = 0
    misses = 0
    for i in range(n_verify):
        target = verify_voxel_int[i]
        # Check if this coordinate exists in coords
        mask = (coords == target).all(dim=1)
        if mask.any():
            matches += 1
        else:
            misses += 1

    print(f"\n   Nearest voxel lookup ({n_verify} vertices):")
    print(f"   Exact matches (voxel exists at vertex location): {matches}")
    print(f"   Misses (no voxel at vertex location): {misses}")
    print(f"   Miss rate: {100*misses/n_verify:.1f}%")

    # If we have misses, check how far vertices are from nearest voxel
    if misses > 0:
        print("\n   Distance to nearest voxel for misses:")

        # Build a simple lookup for coords
        # For efficiency, just check a random subset
        distances = []
        for i in range(min(50, n_verify)):
            target = verify_voxel_int[i]
            mask = (coords == target).all(dim=1)
            if not mask.any():
                # Find nearest
                diff = (coords.float() - target.float()).abs().sum(dim=1)
                min_dist = diff.min().item()
                distances.append(min_dist)

        if distances:
            print(f"   Min distance: {min(distances):.1f} voxels")
            print(f"   Max distance: {max(distances):.1f} voxels")
            print(f"   Mean distance: {np.mean(distances):.1f} voxels")

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
