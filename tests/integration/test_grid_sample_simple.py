"""Simple test of flex_gemm grid_sample_3d precision."""
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from pathlib import Path
TRELLIS_ROOT = Path(__file__).parent

import sys
sys.path.insert(0, str(TRELLIS_ROOT))

import torch
from flex_gemm.ops.grid_sample import grid_sample_3d

# Ensure venv is used
assert "venv311" in sys.executable, f"Wrong Python: {sys.executable}"

print("=" * 60)
print("SIMPLE GRID SAMPLE PRECISION TEST")
print("=" * 60)

device = 'cuda'

# Create a simple 8x8x8 sparse tensor with known values
W, H, D = 8, 8, 8
C = 3  # RGB

# Place voxels with known colors
voxels = [
    # (x, y, z, R, G, B)
    (0, 0, 0, 1.0, 0.0, 0.0),  # Red at origin
    (7, 0, 0, 0.0, 1.0, 0.0),  # Green at x-max
    (0, 7, 0, 0.0, 0.0, 1.0),  # Blue at y-max
    (0, 0, 7, 1.0, 1.0, 0.0),  # Yellow at z-max
    (4, 4, 4, 1.0, 1.0, 1.0),  # White at center
    (3, 3, 3, 0.5, 0.5, 0.5),  # Gray near center
    (5, 5, 5, 0.5, 0.5, 0.5),  # Gray near center
]

N = len(voxels)
coords = torch.tensor([[0, x, y, z] for x, y, z, r, g, b in voxels], dtype=torch.int32, device=device)
attrs = torch.tensor([[r, g, b] for x, y, z, r, g, b in voxels], dtype=torch.float32, device=device)

print(f"\nSparse tensor: {N} voxels")
for i, (x, y, z, r, g, b) in enumerate(voxels):
    print(f"  [{i}] ({x},{y},{z}): RGB=({r:.1f}, {g:.1f}, {b:.1f})")

shape = torch.Size([1, C, W, H, D])

# Test 1: Query at exact voxel locations - should return exact colors
print("\n" + "=" * 60)
print("TEST 1: Query at exact voxel locations")
print("=" * 60)

grid_exact = torch.tensor([[[x, y, z] for x, y, z, r, g, b in voxels]], dtype=torch.float32, device=device)
result = grid_sample_3d(attrs, coords, shape, grid_exact, mode='nearest')  # [1, N, 3]
result = result[0]  # [N, 3]

print(f"\nResult shape after squeeze: {result.shape}")
print("\nNearest mode:")
for i, (x, y, z, r, g, b) in enumerate(voxels):
    expected = [r, g, b]
    got = result[i].cpu().tolist()
    diff = max(abs(got[j] - expected[j]) for j in range(3))
    status = "MATCH" if diff < 0.001 else f"DIFF={diff:.4f}"
    print(f"  ({x},{y},{z}): Expected=({r:.1f},{g:.1f},{b:.1f}), Got=({got[0]:.2f},{got[1]:.2f},{got[2]:.2f}) {status}")

result_tri = grid_sample_3d(attrs, coords, shape, grid_exact, mode='trilinear')[0]

print("\nTrilinear mode:")
for i, (x, y, z, r, g, b) in enumerate(voxels):
    expected = [r, g, b]
    got = result_tri[i].cpu().tolist()
    diff = max(abs(got[j] - expected[j]) for j in range(3))
    status = "MATCH" if diff < 0.001 else f"DIFF={diff:.4f}"
    print(f"  ({x},{y},{z}): Expected=({r:.1f},{g:.1f},{b:.1f}), Got=({got[0]:.2f},{got[1]:.2f},{got[2]:.2f}) {status}")

# Test 2: Query between center voxels - should interpolate
print("\n" + "=" * 60)
print("TEST 2: Query between voxels (trilinear interpolation)")
print("=" * 60)

# Query at midpoint between center (4,4,4) white and (3,3,3) gray
mid_point = torch.tensor([[[3.5, 3.5, 3.5]]], dtype=torch.float32, device=device)
result_mid = grid_sample_3d(attrs, coords, shape, mid_point, mode='trilinear')[0, 0].cpu().tolist()

print(f"Query at (3.5, 3.5, 3.5) - between gray(0.5,0.5,0.5) and white(1,1,1)")
print(f"  Result: ({result_mid[0]:.3f}, {result_mid[1]:.3f}, {result_mid[2]:.3f})")
print(f"  Note: Sparse interpolation may differ from dense - only interpolates between existing voxels")

# Test 3: Query outside occupied voxels
print("\n" + "=" * 60)
print("TEST 3: Query at empty location")
print("=" * 60)

# Query at (1,1,1) which has no voxel - should return zero or interpolated from neighbors
empty_point = torch.tensor([[[1.0, 1.0, 1.0]]], dtype=torch.float32, device=device)
result_empty = grid_sample_3d(attrs, coords, shape, empty_point, mode='trilinear')[0, 0].cpu().tolist()

print(f"Query at (1,1,1) - empty location with only (0,0,0) red voxel nearby")
print(f"  Result: ({result_empty[0]:.3f}, {result_empty[1]:.3f}, {result_empty[2]:.3f})")
print(f"  Note: flex_gemm sparse sampling returns weighted average of existing voxels")

# Test 4: Check for numerical precision issues near boundaries
print("\n" + "=" * 60)
print("TEST 4: Boundary precision test")
print("=" * 60)

# Query at slight offsets from exact voxel locations
offsets = [0.0, 0.001, 0.01, 0.1, 0.5]
base = torch.tensor([[[4.0, 4.0, 4.0]]], dtype=torch.float32, device=device)  # White voxel

print("Querying around (4,4,4) white voxel:")
for offset in offsets:
    point = base + offset
    result_off = grid_sample_3d(attrs, coords, shape, point, mode='trilinear')[0, 0].cpu().tolist()
    print(f"  (4+{offset:.3f}, 4+{offset:.3f}, 4+{offset:.3f}): ({result_off[0]:.4f}, {result_off[1]:.4f}, {result_off[2]:.4f})")

# Test 5: Check if out-of-bounds returns zero
print("\n" + "=" * 60)
print("TEST 5: Out-of-bounds query")
print("=" * 60)

oob_points = [
    [-1.0, -1.0, -1.0],  # Below min
    [8.0, 8.0, 8.0],      # Above max
    [100.0, 0.0, 0.0],    # Way outside
]

for pt in oob_points:
    query = torch.tensor([[pt]], dtype=torch.float32, device=device)
    result_oob = grid_sample_3d(attrs, coords, shape, query, mode='trilinear')[0, 0].cpu().tolist()
    print(f"  ({pt[0]:.1f}, {pt[1]:.1f}, {pt[2]:.1f}): ({result_oob[0]:.4f}, {result_oob[1]:.4f}, {result_oob[2]:.4f})")

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)
