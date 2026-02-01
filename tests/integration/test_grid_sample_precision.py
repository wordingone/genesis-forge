"""Compare flex_gemm grid_sample_3d with PyTorch F.grid_sample for numerical accuracy."""
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from pathlib import Path
TRELLIS_ROOT = Path(__file__).parent

import sys
sys.path.insert(0, str(TRELLIS_ROOT))

import torch
import torch.nn.functional as F
from flex_gemm.ops.grid_sample import grid_sample_3d

# Ensure venv is used
assert "venv311" in sys.executable, f"Wrong Python: {sys.executable}"


def create_dense_volume(sparse_attrs, sparse_coords, shape):
    """Convert sparse tensor to dense for PyTorch grid_sample comparison."""
    B, C, W, H, D = shape
    dense = torch.zeros(B, C, W, H, D, dtype=sparse_attrs.dtype, device=sparse_attrs.device)

    # sparse_coords is [N, 4] with [batch, x, y, z]
    batch = sparse_coords[:, 0].long()
    x = sparse_coords[:, 1].long()
    y = sparse_coords[:, 2].long()
    z = sparse_coords[:, 3].long()

    # Assign values
    for i in range(sparse_coords.shape[0]):
        dense[batch[i], :, x[i], y[i], z[i]] = sparse_attrs[i]

    return dense


def pytorch_grid_sample(dense_volume, grid, mode='bilinear'):
    """
    PyTorch reference implementation of 3D grid sampling.

    Args:
        dense_volume: [B, C, W, H, D] dense tensor
        grid: [B, L, 3] grid coordinates in VOXEL space (0 to size-1)
        mode: 'bilinear' or 'nearest'

    Returns:
        [L, C] sampled values
    """
    B, C, W, H, D = dense_volume.shape
    _, L, _ = grid.shape

    # Convert voxel coords to normalized coords for F.grid_sample
    # F.grid_sample expects [-1, 1] range
    # grid is in voxel space: [0, size-1]
    # normalized = 2 * (voxel / (size - 1)) - 1

    grid_norm = grid.clone()
    grid_norm[..., 0] = 2 * grid[..., 0] / (W - 1) - 1
    grid_norm[..., 1] = 2 * grid[..., 1] / (H - 1) - 1
    grid_norm[..., 2] = 2 * grid[..., 2] / (D - 1) - 1

    # F.grid_sample expects [B, N, H_out, W_out, 3] for 3D
    # Our grid is [B, L, 3], reshape to [B, L, 1, 1, 3]
    grid_5d = grid_norm.reshape(B, L, 1, 1, 3)

    # Note: F.grid_sample uses (z, y, x) ordering for 3D
    # We need to reverse the coordinate order
    grid_5d = grid_5d[..., [2, 1, 0]]  # x,y,z -> z,y,x

    # Sample
    sampled = F.grid_sample(
        dense_volume, grid_5d,
        mode=mode,
        padding_mode='border',
        align_corners=True
    )

    # Output is [B, C, L, 1, 1], reshape to [L, C]
    return sampled[0, :, :, 0, 0].T


def main():
    print("=" * 60)
    print("GRID SAMPLE PRECISION TEST")
    print("=" * 60)

    device = 'cuda'

    # Create a test case with known values
    W, H, D = 16, 16, 16
    C = 3  # RGB
    N = 100  # Number of sparse voxels

    # Create sparse data at random locations with random colors
    torch.manual_seed(42)

    # Random coordinates
    coords_xyz = torch.randint(0, W, (N, 3), device=device, dtype=torch.int32)
    coords = torch.cat([torch.zeros(N, 1, device=device, dtype=torch.int32), coords_xyz], dim=1)  # [N, 4]

    # Random colors (RGB)
    attrs = torch.rand(N, C, device=device, dtype=torch.float32)

    print(f"\nSparse tensor: {N} voxels with {C} channels in {W}x{H}x{D} grid")

    # Create dense volume for PyTorch comparison
    shape = (1, C, W, H, D)
    dense = create_dense_volume(attrs, coords, shape)

    # Create query points
    n_queries = 1000
    # Sample at random locations including fractional coordinates for interpolation
    grid = torch.rand(1, n_queries, 3, device=device) * (W - 1)  # [0, W-1]

    print(f"Querying {n_queries} points")
    print(f"Grid range: [{grid.min():.2f}, {grid.max():.2f}]")

    # Sample with flex_gemm
    print("\n" + "=" * 60)
    print("FLEX_GEMM GRID_SAMPLE_3D (TRILINEAR)")
    print("=" * 60)

    result_flex_raw = grid_sample_3d(
        attrs, coords,
        shape=torch.Size(shape),
        grid=grid,
        mode='trilinear'
    )  # [1, L, C]

    # Squeeze batch dimension to get [L, C]
    result_flex = result_flex_raw[0] if result_flex_raw.dim() == 3 else result_flex_raw

    print(f"Result shape (raw): {result_flex_raw.shape}")
    print(f"Result shape (squeezed): {result_flex.shape}")
    print(f"Result range: [{result_flex.min():.4f}, {result_flex.max():.4f}]")
    print(f"Result mean: {result_flex.mean():.4f}")

    # Sample with PyTorch (trilinear = bilinear in 3D)
    print("\n" + "=" * 60)
    print("PYTORCH F.GRID_SAMPLE (BILINEAR)")
    print("=" * 60)

    result_torch = pytorch_grid_sample(dense, grid, mode='bilinear')

    print(f"Result shape: {result_torch.shape}")
    print(f"Result range: [{result_torch.min():.4f}, {result_torch.max():.4f}]")
    print(f"Result mean: {result_torch.mean():.4f}")

    # Compare
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)

    diff = (result_flex - result_torch).abs()
    print(f"Max absolute difference: {diff.max():.6f}")
    print(f"Mean absolute difference: {diff.mean():.6f}")
    print(f"Std of difference: {diff.std():.6f}")

    # Check where large differences occur
    large_diff_mask = diff.max(dim=1).values > 0.01
    n_large_diff = large_diff_mask.sum().item()
    print(f"Points with diff > 0.01: {n_large_diff} / {n_queries} ({100*n_large_diff/n_queries:.1f}%)")

    # Analyze where differences occur
    if n_large_diff > 0:
        print("\nLarge difference analysis:")
        large_diff_indices = large_diff_mask.nonzero().squeeze(-1)
        if large_diff_indices.dim() == 0:
            large_diff_indices = large_diff_indices.unsqueeze(0)
        for idx in large_diff_indices[:5].tolist():  # Show first 5
            coord = grid[0, idx].tolist()
            flex_val = result_flex[idx].tolist()
            torch_val = result_torch[idx].tolist()
            diff_val = diff[idx].tolist()
            print(f"  Query ({coord[0]:.2f}, {coord[1]:.2f}, {coord[2]:.2f}):")
            print(f"    flex_gemm: {flex_val}")
            print(f"    pytorch:   {torch_val}")
            print(f"    diff:      {diff_val}")

    # Test nearest mode
    print("\n" + "=" * 60)
    print("NEAREST MODE COMPARISON")
    print("=" * 60)

    result_flex_nearest_raw = grid_sample_3d(
        attrs, coords,
        shape=torch.Size(shape),
        grid=grid,
        mode='nearest'
    )
    result_flex_nearest = result_flex_nearest_raw[0] if result_flex_nearest_raw.dim() == 3 else result_flex_nearest_raw

    result_torch_nearest = pytorch_grid_sample(dense, grid, mode='nearest')

    diff_nearest = (result_flex_nearest - result_torch_nearest).abs()
    print(f"Max absolute difference (nearest): {diff_nearest.max():.6f}")
    print(f"Mean absolute difference (nearest): {diff_nearest.mean():.6f}")

    # The key question: does flex_gemm handle sparse-to-dense differently?
    print("\n" + "=" * 60)
    print("SPARSE BOUNDARY TEST")
    print("=" * 60)

    # Query at exact sparse voxel locations
    exact_grid = coords_xyz.float().unsqueeze(0)  # [1, N, 3]

    result_flex_exact_raw = grid_sample_3d(
        attrs, coords,
        shape=torch.Size(shape),
        grid=exact_grid,
        mode='trilinear'
    )
    result_flex_exact = result_flex_exact_raw[0] if result_flex_exact_raw.dim() == 3 else result_flex_exact_raw

    result_torch_exact = pytorch_grid_sample(dense, exact_grid, mode='bilinear')

    print(f"Query at exact voxel locations ({N} queries)")

    # Compare with original attrs
    diff_from_orig = (result_flex_exact - attrs).abs()
    print(f"flex_gemm diff from original: max={diff_from_orig.max():.6f}, mean={diff_from_orig.mean():.6f}")

    diff_torch_from_orig = (result_torch_exact - attrs).abs()
    print(f"pytorch diff from original: max={diff_torch_from_orig.max():.6f}, mean={diff_torch_from_orig.mean():.6f}")

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
