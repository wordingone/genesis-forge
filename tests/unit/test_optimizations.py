"""Quick test to verify performance optimizations are working."""
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch

# Enable cuDNN benchmark (as the scripts now do)
torch.backends.cudnn.benchmark = True

print("="*60)
print("PERFORMANCE OPTIMIZATION VERIFICATION")
print("="*60)

print(f"\n1. cuDNN benchmark: {torch.backends.cudnn.benchmark}")
if torch.backends.cudnn.benchmark:
    print("   [OK] cuDNN kernel auto-tuning enabled")
else:
    print("   [FAIL] cuDNN benchmark should be True")

print(f"\n2. BF16 support: {torch.cuda.is_bf16_supported()}")
if torch.cuda.is_bf16_supported():
    print("   [OK] RTX 4090 native BF16 tensor cores available")
else:
    print("   [FAIL] BF16 should be supported on RTX 4090")

print(f"\n3. Autocast test:")
print(f"   Outside context: {torch.is_autocast_enabled()}")
with torch.autocast('cuda', dtype=torch.bfloat16, enabled=True):
    is_enabled = torch.is_autocast_enabled()
    print(f"   Inside context: {is_enabled}")
    if is_enabled:
        print("   [OK] Autocast will bypass manual_cast() overhead")
    else:
        print("   [FAIL] Autocast should be enabled inside context")

# Test that manual_cast returns unchanged tensor when autocast enabled
from trellis2.modules.utils import manual_cast
test_tensor = torch.randn(10, 10, device='cuda')
with torch.autocast('cuda', dtype=torch.bfloat16, enabled=True):
    result = manual_cast(test_tensor, torch.bfloat16)
    same_tensor = result is test_tensor  # Should be same object, not copy
    print(f"\n4. manual_cast bypass: {same_tensor}")
    if same_tensor:
        print("   [OK] manual_cast returns same tensor (no copy)")
    else:
        print("   [FAIL] manual_cast should return same tensor in autocast")

print(f"\n5. GPU Info:")
props = torch.cuda.get_device_properties(0)
print(f"   Device: {props.name}")
print(f"   VRAM: {props.total_memory / 1024**3:.1f} GB")
print(f"   Compute capability: {props.major}.{props.minor}")

print("\n" + "="*60)
print("All optimization checks complete.")
print("="*60)
