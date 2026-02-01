"""Test grid_sample_3d coordinate handling to find the axis mismatch."""
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
from flex_gemm.ops.grid_sample import grid_sample_3d

# Ensure venv is used
assert "venv311" in sys.executable, f"Wrong Python: {sys.executable}"

def main():
    print("=" * 60)
    print("GRID_SAMPLE_3D COORDINATE TEST")
    print("=" * 60)

    # Create a simple sparse tensor with known values
    # We'll put distinct colors at specific coordinates
    device = 'cuda'

    # Simple 8x8x8 grid
    W, H, D = 8, 8, 8
    C = 3  # RGB

    # Create sparse coordinates with distinct colors at each location
    coords_list = []
    feats_list = []

    # Place voxels at corners with distinct colors
    test_cases = [
        # (x, y, z, R, G, B)
        (0, 0, 0, 1.0, 0.0, 0.0),  # Origin = pure RED
        (7, 0, 0, 0.0, 1.0, 0.0),  # X-max = pure GREEN
        (0, 7, 0, 0.0, 0.0, 1.0),  # Y-max = pure BLUE
        (0, 0, 7, 1.0, 1.0, 0.0),  # Z-max = YELLOW
        (4, 4, 4, 1.0, 0.0, 1.0),  # Center = MAGENTA
    ]

    for x, y, z, r, g, b in test_cases:
        coords_list.append([0, x, y, z])  # batch=0, x, y, z
        feats_list.append([r, g, b])

    coords = torch.tensor(coords_list, dtype=torch.int32, device=device)  # [N, 4]
    feats = torch.tensor(feats_list, dtype=torch.float32, device=device)  # [N, C]

    print(f"\nSparse tensor:")
    print(f"  Coords shape: {coords.shape}")
    print(f"  Feats shape: {feats.shape}")
    print(f"  Grid size: {W}x{H}x{D}")

    for i, (x, y, z, r, g, b) in enumerate(test_cases):
        print(f"  [{i}] @ ({x},{y},{z}): RGB=({r:.1f}, {g:.1f}, {b:.1f})")

    # Now query at exact coordinates
    print("\n" + "=" * 60)
    print("QUERY TEST: Sample at exact voxel coordinates")
    print("=" * 60)

    shape = torch.Size([1, C, W, H, D])

    for x, y, z, expected_r, expected_g, expected_b in test_cases:
        # grid is [B, L, 3] with (x, y, z) ordering
        grid = torch.tensor([[[x, y, z]]], dtype=torch.float32, device=device)

        result = grid_sample_3d(feats, coords, shape, grid, mode='nearest')
        r, g, b = result[0, 0].tolist()

        match = (abs(r - expected_r) < 0.01 and
                 abs(g - expected_g) < 0.01 and
                 abs(b - expected_b) < 0.01)
        status = "MATCH" if match else "MISMATCH"

        print(f"  Query ({x},{y},{z}): Got RGB=({r:.2f}, {g:.2f}, {b:.2f}), "
              f"Expected=({expected_r:.1f}, {expected_g:.1f}, {expected_b:.1f}) {status}")

    # Test permutations to find the correct mapping
    print("\n" + "=" * 60)
    print("PERMUTATION TEST: Try different axis orderings")
    print("=" * 60)

    # If (x,y,z) doesn't work, try all permutations
    import itertools
    for perm in itertools.permutations([0, 1, 2]):
        axis_names = ['X', 'Y', 'Z']
        perm_name = f"({axis_names[perm[0]]}, {axis_names[perm[1]]}, {axis_names[perm[2]]})"

        all_match = True
        for x, y, z, expected_r, expected_g, expected_b in test_cases:
            # Apply permutation to query coordinates
            orig = [x, y, z]
            permuted = [orig[perm[0]], orig[perm[1]], orig[perm[2]]]

            grid = torch.tensor([[permuted]], dtype=torch.float32, device=device)
            result = grid_sample_3d(feats, coords, shape, grid, mode='nearest')
            r, g, b = result[0, 0].tolist()

            if abs(r - expected_r) > 0.01 or abs(g - expected_g) > 0.01 or abs(b - expected_b) > 0.01:
                all_match = False
                break

        status = "ALL MATCH" if all_match else ""
        if all_match:
            print(f"  {perm_name}: {status}")

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
