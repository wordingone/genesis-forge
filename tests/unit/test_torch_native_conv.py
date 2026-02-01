"""Test the torch_native sparse convolution backend."""
import os
import sys

# Set environment BEFORE any imports
os.environ['SPARSE_CONV_BACKEND'] = 'torch_native'
sys.path.insert(0, 'o_voxel')

import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Import after env is set
from trellis2.modules.sparse import SparseTensor, config
from trellis2.modules.sparse.conv import SparseConv3d

print(f"Conv backend: {config.CONV}")

# Create a simple sparse tensor for testing
# Simulating a small version of what TRELLIS.2 produces
N = 1000  # Number of sparse voxels
C = 64    # Channels
spatial_shape = (64, 64, 64)

# Random sparse coordinates - using int32 as expected by SparseTensor
coords = torch.randint(0, 64, (N, 4), dtype=torch.int32)
coords[:, 0] = 0  # Batch 0

# Random features
feats = torch.randn(N, C, dtype=torch.float32)

print(f"\nCreating SparseTensor with {N} voxels, {C} channels")

# With torch_native backend, SparseTensor uses dict storage
st = SparseTensor(feats=feats, coords=coords, shape=torch.Size([1, C]))
print(f"SparseTensor created: feats {st.feats.shape}, coords {st.coords.shape}")

# Create conv layer
print("\nCreating SparseConv3d layer...")
conv = SparseConv3d(
    in_channels=C,
    out_channels=C,
    kernel_size=3,
    stride=1,
    dilation=1,
    padding=None,
    bias=True
)
conv = conv.cuda()

# Move sparse tensor to GPU
st_cuda = SparseTensor(
    feats=feats.cuda(),
    coords=coords.cuda(),
    shape=torch.Size([1, C])
)

# Run forward pass
print("\nRunning forward pass...")
try:
    output = conv(st_cuda)
    print(f"SUCCESS! Output feats shape: {output.feats.shape}")
    print(f"Output coords shape: {output.coords.shape}")
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test with larger tensor to verify int64 handling
print("\n" + "="*60)
print("Testing with larger tensor (simulating 1024_cascade scale)")
print("="*60)

# Simulate ~1M voxels (closer to real 1024_cascade ~17M)
N_large = 1_000_000
coords_large = torch.randint(0, 256, (N_large, 4), dtype=torch.int32)
coords_large[:, 0] = 0  # Batch 0
feats_large = torch.randn(N_large, C, dtype=torch.float32)

print(f"Creating SparseTensor with {N_large:,} voxels...")
st_large = SparseTensor(
    feats=feats_large.cuda(),
    coords=coords_large.cuda(),
    shape=torch.Size([1, C])
)

print(f"Running forward pass on {N_large:,} voxels...")
try:
    import time
    start = time.time()
    output_large = conv(st_large)
    elapsed = time.time() - start
    print(f"SUCCESS! Processed {N_large:,} voxels in {elapsed:.2f}s")
    print(f"Output feats shape: {output_large.feats.shape}")
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test with 17M voxels (actual 1024_cascade size)
print("\n" + "="*60)
print("Testing with 17M voxels (actual 1024_cascade size)")
print("="*60)

N_17m = 17_762_320
try:
    coords_17m = torch.randint(0, 512, (N_17m, 4), dtype=torch.int32)
    coords_17m[:, 0] = 0
    feats_17m = torch.randn(N_17m, C, dtype=torch.float16, device='cuda')  # Use fp16 to save memory

    print(f"Creating SparseTensor with {N_17m:,} voxels...")
    st_17m = SparseTensor(
        feats=feats_17m,
        coords=coords_17m.cuda(),
        shape=torch.Size([1, C])
    )

    # Use fp16 weights
    conv_fp16 = conv.half()

    print(f"Running forward pass on {N_17m:,} voxels...")
    start = time.time()
    output_17m = conv_fp16(st_17m)
    elapsed = time.time() - start
    print(f"SUCCESS! Processed {N_17m:,} voxels in {elapsed:.2f}s")
    print(f"Output feats shape: {output_17m.feats.shape}")
except torch.cuda.OutOfMemoryError as e:
    print(f"CUDA OOM (expected for 24GB VRAM): {e}")
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("Test complete!")
print("="*60)
