"""Compare attribute sampling between our pipeline and expected output.

This script focuses on identifying the ROOT CAUSE of texture artifacts by:
1. Checking attr_volume statistics
2. Checking grid coordinate ranges
3. Checking sampled attribute statistics
4. Comparing with expected patterns
"""
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

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, default='reference/spiral_input.png')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    output_dir = TRELLIS_ROOT / 'outputs' / 'attr_comparison'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ATTRIBUTE SAMPLING COMPARISON")
    print("=" * 70)
    print(f"Input: {args.image}")
    print(f"Output dir: {output_dir}")

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
    SEED = args.seed
    SPARSE_PARAMS = {"steps": 12, "guidance_strength": 7.5, "guidance_rescale": 0.7, "rescale_t": 5.0}
    SHAPE_PARAMS = {"steps": 12, "guidance_strength": 7.5, "guidance_rescale": 0.5, "rescale_t": 3.0}
    TEX_PARAMS = {"steps": 12, "guidance_strength": 1.0, "guidance_rescale": 0.0, "rescale_t": 3.0}

    # Load and preprocess image
    print("\nPreprocessing image...")
    raw_image = PILImage.open(args.image).convert("RGBA")
    preprocessed = pipeline.preprocess_image(raw_image)
    preprocessed.save(str(output_dir / "00_preprocessed.png"))

    torch.manual_seed(SEED)

    # Get conditioning
    print("\nGetting image conditioning...")
    cond_512 = pipeline.get_cond([preprocessed], 512)
    cond_1024 = pipeline.get_cond([preprocessed], 1024)

    # Run pipeline stages
    print("\nRunning pipeline stages...")
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=True):
        coords = pipeline.sample_sparse_structure(
            cond_512, 32, num_samples=1, sampler_params=SPARSE_PARAMS,
        )
        print(f"  Sparse coords: {coords.shape}")

        shape_slat, hr_resolution = pipeline.sample_shape_slat_cascade(
            cond_512, cond_1024,
            pipeline.models['shape_slat_flow_model_512'],
            pipeline.models['shape_slat_flow_model_1024'],
            512, 1024, coords, sampler_params=SHAPE_PARAMS,
        )
        print(f"  Shape SLat: {shape_slat.feats.shape}, HR res: {hr_resolution}")

        tex_slat = pipeline.sample_tex_slat(
            cond_1024, pipeline.models['tex_slat_flow_model_1024'],
            shape_slat, sampler_params=TEX_PARAMS,
        )
        print(f"  Tex SLat: {tex_slat.feats.shape}")

    torch.cuda.empty_cache()

    # Decode mesh
    print("\nDecoding mesh...")
    out_meshes = pipeline.decode_latent(shape_slat, tex_slat, hr_resolution)
    mesh = out_meshes[0]

    print("\n" + "=" * 70)
    print("ATTRIBUTE VOLUME ANALYSIS")
    print("=" * 70)

    attr_volume = mesh.attrs.cuda()
    coords_vol = mesh.coords.cuda()

    print(f"\nAttr volume shape: {attr_volume.shape}")
    print(f"  - Number of voxels: {attr_volume.shape[0]}")
    print(f"  - Channels per voxel: {attr_volume.shape[1]}")

    print(f"\nAttr volume statistics:")
    print(f"  - Min: {attr_volume.min().item():.6f}")
    print(f"  - Max: {attr_volume.max().item():.6f}")
    print(f"  - Mean: {attr_volume.mean().item():.6f}")
    print(f"  - Std: {attr_volume.std().item():.6f}")

    # Check for NaN/Inf
    nan_count = torch.isnan(attr_volume).sum().item()
    inf_count = torch.isinf(attr_volume).sum().item()
    print(f"  - NaN count: {nan_count}")
    print(f"  - Inf count: {inf_count}")

    # Check per-channel statistics (PBR layout: RGB, Metallic, Roughness, Alpha)
    print(f"\nPer-channel analysis:")
    channel_names = ['R', 'G', 'B', 'Metallic', 'Roughness', 'Alpha']
    for i, name in enumerate(channel_names):
        if i < attr_volume.shape[1]:
            ch = attr_volume[:, i]
            print(f"  {name}: min={ch.min().item():.4f}, max={ch.max().item():.4f}, mean={ch.mean().item():.4f}")

    print(f"\nCoords volume shape: {coords_vol.shape}")
    print(f"Coords statistics:")
    print(f"  - X: min={coords_vol[:,0].min().item():.0f}, max={coords_vol[:,0].max().item():.0f}")
    print(f"  - Y: min={coords_vol[:,1].min().item():.0f}, max={coords_vol[:,1].max().item():.0f}")
    print(f"  - Z: min={coords_vol[:,2].min().item():.0f}, max={coords_vol[:,2].max().item():.0f}")

    # Grid size and voxel size
    grid_size = round(1.0 / mesh.voxel_size)
    aabb = torch.tensor([[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], device='cuda', dtype=torch.float32)
    grid_size_t = torch.tensor([grid_size, grid_size, grid_size], device='cuda')
    voxel_size = (aabb[1] - aabb[0]) / grid_size_t.float()

    print(f"\nGrid configuration:")
    print(f"  - Grid size: {grid_size}")
    print(f"  - Voxel size: {voxel_size.tolist()}")
    print(f"  - AABB: {aabb.tolist()}")

    # Check if coords are within expected range
    expected_max_coord = grid_size - 1
    out_of_range = (coords_vol < 0).any(dim=1) | (coords_vol > expected_max_coord).any(dim=1)
    print(f"  - Coords out of [0, {expected_max_coord}]: {out_of_range.sum().item()}")

    print("\n" + "=" * 70)
    print("MESH VERTEX ANALYSIS")
    print("=" * 70)

    vertices = mesh.vertices.cuda()
    faces = mesh.faces.cuda()

    print(f"\nVertices: {vertices.shape[0]}")
    print(f"  - X: min={vertices[:,0].min().item():.6f}, max={vertices[:,0].max().item():.6f}")
    print(f"  - Y: min={vertices[:,1].min().item():.6f}, max={vertices[:,1].max().item():.6f}")
    print(f"  - Z: min={vertices[:,2].min().item():.6f}, max={vertices[:,2].max().item():.6f}")

    # Check if vertices are within AABB
    in_aabb = ((vertices >= aabb[0]) & (vertices <= aabb[1])).all(dim=1)
    print(f"  - Vertices in AABB: {in_aabb.sum().item()} / {vertices.shape[0]}")

    if not in_aabb.all():
        out_verts = vertices[~in_aabb]
        print(f"  - Out-of-AABB vertices sample:")
        for i in range(min(5, out_verts.shape[0])):
            print(f"      {out_verts[i].tolist()}")

    # Convert vertex positions to voxel coordinates to check sampling positions
    vert_voxel = (vertices - aabb[0]) / voxel_size
    print(f"\nVertex voxel coords:")
    print(f"  - X: min={vert_voxel[:,0].min().item():.2f}, max={vert_voxel[:,0].max().item():.2f}")
    print(f"  - Y: min={vert_voxel[:,1].min().item():.2f}, max={vert_voxel[:,1].max().item():.2f}")
    print(f"  - Z: min={vert_voxel[:,2].min().item():.2f}, max={vert_voxel[:,2].max().item():.2f}")

    # Check sparsity coverage
    print("\n" + "=" * 70)
    print("SPARSITY COVERAGE ANALYSIS")
    print("=" * 70)

    # Create dense occupancy grid
    coords_int = coords_vol.long()
    occupancy = torch.zeros(grid_size, grid_size, grid_size, device='cuda', dtype=torch.bool)
    valid_coords = (coords_int >= 0).all(dim=1) & (coords_int < grid_size).all(dim=1)
    valid_int = coords_int[valid_coords]
    occupancy[valid_int[:, 0], valid_int[:, 1], valid_int[:, 2]] = True

    total_voxels = grid_size ** 3
    occupied_voxels = occupancy.sum().item()
    print(f"Total voxels in grid: {total_voxels}")
    print(f"Occupied voxels: {occupied_voxels} ({100*occupied_voxels/total_voxels:.2f}%)")

    # Check which voxels the mesh vertices would sample from
    vert_voxel_int = vert_voxel.long().clamp(0, grid_size - 1)
    verts_in_occupied = occupancy[vert_voxel_int[:, 0], vert_voxel_int[:, 1], vert_voxel_int[:, 2]]
    print(f"Vertices sampling from occupied voxels: {verts_in_occupied.sum().item()} / {vertices.shape[0]}")

    # Trilinear sampling uses 8 neighbors - check how many have all 8 neighbors
    # For each vertex, check if all 8 corner voxels are occupied
    corners = []
    for dx in [0, 1]:
        for dy in [0, 1]:
            for dz in [0, 1]:
                corner = (vert_voxel_int + torch.tensor([dx, dy, dz], device='cuda')).clamp(0, grid_size - 1)
                corner_occupied = occupancy[corner[:, 0], corner[:, 1], corner[:, 2]]
                corners.append(corner_occupied)

    all_corners_occupied = torch.stack(corners, dim=1).all(dim=1)
    print(f"Vertices with all 8 trilinear corners occupied: {all_corners_occupied.sum().item()} / {vertices.shape[0]}")

    some_corners_occupied = torch.stack(corners, dim=1).any(dim=1)
    print(f"Vertices with at least 1 corner occupied: {some_corners_occupied.sum().item()} / {vertices.shape[0]}")

    # Vertices with NO corners occupied will sample zeros
    no_corners = ~some_corners_occupied
    if no_corners.sum() > 0:
        print(f"\n  WARNING: {no_corners.sum().item()} vertices have NO occupied corners!")
        print(f"  These will sample BLACK (zero) values")

    print("\n" + "=" * 70)
    print("DIAGNOSIS COMPLETE")
    print("=" * 70)

    # Save a visualization of the occupancy grid (middle slice)
    mid = grid_size // 2
    slice_xy = occupancy[:, :, mid].float().cpu().numpy()
    slice_xz = occupancy[:, mid, :].float().cpu().numpy()
    slice_yz = occupancy[mid, :, :].float().cpu().numpy()

    Image.fromarray((slice_xy * 255).astype(np.uint8)).save(output_dir / "occupancy_xy.png")
    Image.fromarray((slice_xz * 255).astype(np.uint8)).save(output_dir / "occupancy_xz.png")
    Image.fromarray((slice_yz * 255).astype(np.uint8)).save(output_dir / "occupancy_yz.png")
    print(f"\nOccupancy slices saved to {output_dir}")


if __name__ == "__main__":
    main()
