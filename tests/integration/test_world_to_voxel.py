"""Test world-to-voxel coordinate transformation in the texture baking pipeline."""
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
    print("WORLD-TO-VOXEL COORDINATE TRANSFORMATION TEST")
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

    print(f"\n4. Mesh data:")
    print(f"   Vertices: {out_mesh.vertices.shape}")
    print(f"   Faces: {out_mesh.faces.shape}")
    print(f"   Voxel origin: {out_mesh.origin}")
    print(f"   Voxel size: {out_mesh.voxel_size}")

    vertices = out_mesh.vertices.cuda()
    attrs = out_mesh.attrs.cuda()
    coords = out_mesh.coords.cuda()  # [N, 3]
    layout = out_mesh.layout

    print(f"\n5. Voxel data:")
    print(f"   Attrs shape: {attrs.shape}")
    print(f"   Coords shape: {coords.shape}")
    print(f"   Coords range X: [{coords[:, 0].min()}, {coords[:, 0].max()}]")
    print(f"   Coords range Y: [{coords[:, 1].min()}, {coords[:, 1].max()}]")
    print(f"   Coords range Z: [{coords[:, 2].min()}, {coords[:, 2].max()}]")

    # Test: Sample at known voxel coordinates
    print("\n" + "=" * 60)
    print("TEST 1: Sample at known voxel coordinates")
    print("=" * 60)

    # Pick some random voxel indices
    n_test = 10
    test_indices = torch.randperm(coords.shape[0])[:n_test]

    # Get the coordinates and attributes at those indices
    test_coords = coords[test_indices]  # [n_test, 3]
    test_attrs = attrs[test_indices]  # [n_test, C]

    print(f"\n   Testing {n_test} known voxel positions...")

    # Prepare for grid_sample_3d
    grid_size = torch.tensor([1024, 1024, 1024], device='cuda', dtype=torch.int32)
    shape = torch.Size([1, attrs.shape[1], *grid_size.tolist()])
    coords_4d = torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=-1)  # [N, 4]

    # Sample at exact voxel coordinates
    grid = test_coords.float().reshape(1, -1, 3)  # [1, n_test, 3]
    sampled = grid_sample_3d(attrs, coords_4d, shape, grid, mode='nearest')[0]  # [n_test, C]

    # Compare
    base_color_slice = layout['base_color']
    original_rgb = test_attrs[:, base_color_slice]
    sampled_rgb = sampled[:, base_color_slice]

    print(f"\n   Comparing original vs sampled RGB at voxel coordinates:")
    for i in range(min(5, n_test)):
        coord = test_coords[i].tolist()
        orig = original_rgb[i].tolist()
        samp = sampled_rgb[i].tolist()
        diff = (original_rgb[i] - sampled_rgb[i]).abs().max().item()
        status = "MATCH" if diff < 0.01 else f"MISMATCH (diff={diff:.4f})"
        print(f"   [{i}] @ ({coord[0]:.0f},{coord[1]:.0f},{coord[2]:.0f}): "
              f"Orig=({orig[0]:.3f},{orig[1]:.3f},{orig[2]:.3f}) "
              f"Samp=({samp[0]:.3f},{samp[1]:.3f},{samp[2]:.3f}) {status}")

    # Test: Convert voxel coords to world coords and back
    print("\n" + "=" * 60)
    print("TEST 2: World-to-voxel coordinate conversion")
    print("=" * 60)

    origin = out_mesh.origin.cuda()
    voxel_size = out_mesh.voxel_size

    # Convert voxel coords to world coords
    world_pos = test_coords.float() * voxel_size + origin  # [n_test, 3]

    # Convert back to voxel coords
    recovered_voxel = (world_pos - origin) / voxel_size

    print(f"\n   Testing round-trip conversion:")
    for i in range(min(5, n_test)):
        orig = test_coords[i].tolist()
        world = world_pos[i].tolist()
        recov = recovered_voxel[i].tolist()
        diff = (test_coords[i].float() - recovered_voxel[i]).abs().max().item()
        status = "MATCH" if diff < 0.01 else f"MISMATCH (diff={diff:.4f})"
        print(f"   [{i}] Voxel({orig[0]:.0f},{orig[1]:.0f},{orig[2]:.0f}) -> "
              f"World({world[0]:.4f},{world[1]:.4f},{world[2]:.4f}) -> "
              f"Voxel({recov[0]:.1f},{recov[1]:.1f},{recov[2]:.1f}) {status}")

    # Test: Sample at world positions
    print("\n" + "=" * 60)
    print("TEST 3: Sample at world positions (via voxel conversion)")
    print("=" * 60)

    # Sample at the world positions
    grid_world = recovered_voxel.reshape(1, -1, 3)  # [1, n_test, 3]
    sampled_world = grid_sample_3d(attrs, coords_4d, shape, grid_world, mode='nearest')[0]  # [n_test, C]

    sampled_world_rgb = sampled_world[:, base_color_slice]

    print(f"\n   Comparing original vs sampled-via-world RGB:")
    for i in range(min(5, n_test)):
        coord = test_coords[i].tolist()
        orig = original_rgb[i].tolist()
        samp = sampled_world_rgb[i].tolist()
        diff = (original_rgb[i] - sampled_world_rgb[i]).abs().max().item()
        status = "MATCH" if diff < 0.01 else f"MISMATCH (diff={diff:.4f})"
        print(f"   [{i}] @ ({coord[0]:.0f},{coord[1]:.0f},{coord[2]:.0f}): "
              f"Orig=({orig[0]:.3f},{orig[1]:.3f},{orig[2]:.3f}) "
              f"Samp=({samp[0]:.3f},{samp[1]:.3f},{samp[2]:.3f}) {status}")

    # Test: Compare official postprocess.py grid calculation
    print("\n" + "=" * 60)
    print("TEST 4: Official postprocess.py grid calculation")
    print("=" * 60)

    # This is how postprocess.py calculates it
    aabb = torch.tensor([[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], device='cuda', dtype=torch.float32)
    grid_size_pp = torch.tensor([1024, 1024, 1024], device='cuda', dtype=torch.int32)
    voxel_size_pp = (aabb[1] - aabb[0]) / grid_size_pp.float()

    print(f"   AABB: {aabb[0].tolist()} to {aabb[1].tolist()}")
    print(f"   Grid size: {grid_size_pp.tolist()}")
    print(f"   Voxel size (postprocess.py): {voxel_size_pp.tolist()}")
    print(f"   Voxel size (mesh): {out_mesh.voxel_size}")
    print(f"   Origin (mesh): {origin.tolist()}")

    # Check if they're the same
    voxel_size_match = abs(voxel_size_pp[0].item() - out_mesh.voxel_size) < 1e-6
    print(f"   Voxel size match: {voxel_size_match}")

    # Calculate grid coords using postprocess.py method
    grid_pp = ((world_pos - aabb[0]) / voxel_size_pp).reshape(1, -1, 3)

    print(f"\n   Comparing grid coord methods:")
    for i in range(min(5, n_test)):
        coord = test_coords[i].tolist()
        my_grid = recovered_voxel[i].tolist()
        pp_grid = grid_pp[0, i].tolist()
        diff = (recovered_voxel[i] - grid_pp[0, i]).abs().max().item()
        status = "SAME" if diff < 0.01 else f"DIFFERENT (diff={diff:.4f})"
        print(f"   [{i}] Voxel({coord[0]:.0f},{coord[1]:.0f},{coord[2]:.0f}): "
              f"My=({my_grid[0]:.1f},{my_grid[1]:.1f},{my_grid[2]:.1f}) "
              f"PP=({pp_grid[0]:.1f},{pp_grid[1]:.1f},{pp_grid[2]:.1f}) {status}")

    # Sample using postprocess.py grid coords
    sampled_pp = grid_sample_3d(attrs, coords_4d, shape, grid_pp, mode='nearest')[0]  # [n_test, C]
    sampled_pp_rgb = sampled_pp[:, base_color_slice]

    print(f"\n   Comparing original vs sampled-via-postprocess-grid RGB:")
    for i in range(min(5, n_test)):
        coord = test_coords[i].tolist()
        orig = original_rgb[i].tolist()
        samp = sampled_pp_rgb[i].tolist()
        diff = (original_rgb[i] - sampled_pp_rgb[i]).abs().max().item()
        status = "MATCH" if diff < 0.01 else f"MISMATCH (diff={diff:.4f})"
        print(f"   [{i}] @ ({coord[0]:.0f},{coord[1]:.0f},{coord[2]:.0f}): "
              f"Orig=({orig[0]:.3f},{orig[1]:.3f},{orig[2]:.3f}) "
              f"Samp=({samp[0]:.3f},{samp[1]:.3f},{samp[2]:.3f}) {status}")

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
