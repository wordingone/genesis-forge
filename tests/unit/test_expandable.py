"""Test expandable_segments support on Windows PyTorch."""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import warnings
import sys

print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.version.cuda}")
print(f"Platform: {sys.platform}")
print(f"Allocator backend: {torch.cuda.get_allocator_backend()}")

# Check if expandable_segments config was accepted
try:
    alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    print(f"PYTORCH_CUDA_ALLOC_CONF: {alloc_conf}")
except Exception as e:
    print(f"Error reading env: {e}")

# Try to allocate - this will trigger the allocator init
print("\nAttempting CUDA allocation with expandable_segments:True...")
try:
    t = torch.zeros(1, device="cuda")
    print(f"Basic allocation: OK")

    # Check memory stats to see if expandable segments is active
    stats = torch.cuda.memory_stats()
    print(f"Allocator active: {stats.get('active.all.current', 'N/A')}")

    # Try a larger allocation pattern that would benefit from expandable segments
    tensors = []
    for i in range(10):
        tensors.append(torch.randn(1000, 1000, device="cuda"))
    for i in range(0, 10, 2):
        del tensors[i]
        tensors[i] = None
    torch.cuda.empty_cache()

    # Allocate a large block that would fail with fragmentation
    big = torch.randn(5000, 5000, device="cuda")
    print(f"Large allocation after fragmentation: OK ({big.numel() * 4 / 1024**2:.1f} MB)")
    del big, tensors
    torch.cuda.empty_cache()

    print("\nexpandable_segments appears to be WORKING")
except RuntimeError as e:
    if "not supported" in str(e).lower():
        print(f"\nexpandable_segments NOT SUPPORTED: {e}")
    else:
        print(f"\nRuntime error: {e}")
except Exception as e:
    print(f"\nUnexpected error: {e}")

# Also check what the allocator config actually shows
print("\n--- Allocator diagnostics ---")
try:
    snapshot = torch.cuda.memory._snapshot()
    if snapshot:
        print(f"Memory snapshot available: {type(snapshot)}")
        if hasattr(snapshot, 'keys'):
            print(f"Snapshot keys: {list(snapshot.keys())[:10]}")
except Exception as e:
    print(f"Snapshot error: {e}")

# Check if cuMemCreate etc. are available via ctypes
print("\n--- CUDA VMM API availability ---")
try:
    import ctypes
    try:
        nvcuda = ctypes.WinDLL("nvcuda.dll")
        vmm_funcs = [
            "cuMemCreate", "cuMemMap", "cuMemAddressReserve",
            "cuMemSetAccess", "cuMemUnmap", "cuMemRelease",
            "cuMemAddressFree", "cuMemGetAllocationGranularity"
        ]
        for func_name in vmm_funcs:
            try:
                func = getattr(nvcuda, func_name)
                print(f"  {func_name}: AVAILABLE (addr={ctypes.cast(func, ctypes.c_void_p).value:#x})")
            except AttributeError:
                print(f"  {func_name}: NOT FOUND")
    except OSError as e:
        print(f"  Cannot load nvcuda.dll: {e}")
except Exception as e:
    print(f"  ctypes check failed: {e}")
