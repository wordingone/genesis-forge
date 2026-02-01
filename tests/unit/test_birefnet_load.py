"""Test BiRefNet loads correctly in offline mode."""
import os
os.environ['HF_HOME'] = os.path.abspath('./models')
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import sys
sys.path.insert(0, 'o_voxel')
sys.path.insert(0, 'cumesh_source')

from trellis2.pipelines.rembg.BiRefNet import BiRefNet
import torch
from PIL import Image

print("[TEST] Loading BiRefNet in offline mode...")
model = BiRefNet("ZhengPeng7/BiRefNet")
print(f"[TEST] Model type: {type(model.model).__name__}")
print(f"[TEST] Parameters: {sum(p.numel() for p in model.model.parameters()) / 1e6:.1f}M")

# Move to CUDA
model.to("cuda")
print("[TEST] Model moved to CUDA")

# Test with a dummy image
img = Image.new('RGB', (512, 512), color=(128, 128, 128))
result = model(img)
print(f"[TEST] Output: mode={result.mode}, size={result.size}")
if result.mode == 'RGBA':
    import numpy as np
    alpha = np.array(result)[:,:,3]
    print(f"[TEST] Alpha channel: min={alpha.min()}, max={alpha.max()}, mean={alpha.mean():.1f}")
print("[TEST] BiRefNet working correctly!")

# Test with actual input image
input_path = 'reference/test_input.jpg.JPG'
if os.path.exists(input_path):
    real_img = Image.open(input_path).convert('RGB')
    print(f"\n[TEST] Testing with real input: {real_img.size}")
    result = model(real_img)
    alpha = np.array(result)[:,:,3]
    print(f"[TEST] Alpha: min={alpha.min()}, max={alpha.max()}, mean={alpha.mean():.1f}")
    # Count foreground pixels
    fg_pct = (alpha > 128).sum() / alpha.size * 100
    print(f"[TEST] Foreground: {fg_pct:.1f}% of pixels")
    # Save mask for inspection
    os.makedirs('outputs', exist_ok=True)
    mask_img = Image.fromarray(alpha)
    mask_img.save('outputs/birefnet_mask.png')
    result.save('outputs/birefnet_result.png')
    print("[TEST] Saved mask to outputs/birefnet_mask.png")
    print("[TEST] Saved result to outputs/birefnet_result.png")

model.cpu()
torch.cuda.empty_cache()
print("[TEST] Cleanup done")
