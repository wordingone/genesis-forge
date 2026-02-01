"""Test 1024 flow model performance directly."""
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
import time

# Enable optimizations
torch.backends.cudnn.benchmark = True

print("Loading pipeline...")
from trellis2.pipelines.trellis2_image_to_3d import Trellis2ImageTo3DPipeline
from trellis2.modules.sparse import SparseTensor

MODEL_PATH = MODELS_DIR / "hub" / "TRELLIS.2-4B"
pipeline = Trellis2ImageTo3DPipeline.from_pretrained(str(MODEL_PATH))

# Configure high VRAM mode
vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
pipeline.configure_vram_mode(vram_gb)
pipeline.to("cuda")

# Get the 1024 flow model
flow_model_1024 = pipeline.models['shape_slat_flow_model_1024']
print(f"1024 model dtype: {next(flow_model_1024.parameters()).dtype}")
print(f"1024 model device: {next(flow_model_1024.parameters()).device}")

# Create test input (simulating ~30k tokens)
n_tokens = 30000
coords = torch.randint(0, 64, (n_tokens, 4), device='cuda').int()
coords[:, 0] = 0  # batch index

# Test WITHOUT autocast
print(f"\n--- Testing WITHOUT autocast ---")
noise = SparseTensor(
    feats=torch.randn(n_tokens, 32, device='cuda'),
    coords=coords,
)
cond = torch.randn(1, 49, 1024, device='cuda')

# Warmup
with torch.no_grad():
    t = torch.tensor([500.0], device='cuda')
    _ = flow_model_1024(noise, t, cond)
torch.cuda.synchronize()

# Time
times = []
for i in range(3):
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        t = torch.tensor([500.0], device='cuda')
        _ = flow_model_1024(noise, t, cond)
    torch.cuda.synchronize()
    t1 = time.time()
    times.append(t1 - t0)
    print(f"  Run {i+1}: {times[-1]:.2f}s")

print(f"  Average (no autocast): {sum(times)/len(times):.2f}s")

# Test WITH autocast
print(f"\n--- Testing WITH autocast (BF16) ---")
with torch.autocast('cuda', dtype=torch.bfloat16, enabled=True):
    # Warmup
    with torch.no_grad():
        t = torch.tensor([500.0], device='cuda')
        _ = flow_model_1024(noise, t, cond)
    torch.cuda.synchronize()

    # Time
    times = []
    for i in range(3):
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            t = torch.tensor([500.0], device='cuda')
            _ = flow_model_1024(noise, t, cond)
        torch.cuda.synchronize()
        t1 = time.time()
        times.append(t1 - t0)
        print(f"  Run {i+1}: {times[-1]:.2f}s")

    print(f"  Average (with autocast): {sum(times)/len(times):.2f}s")

print("\nDone!")
