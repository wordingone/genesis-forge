"""Test full sampler performance to match run_generation_test exactly."""
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
from PIL import Image

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

# Load and preprocess image (same as run_generation_test)
INPUT_IMAGE = TRELLIS_ROOT / "reference" / "test_input.jpg.JPG"
raw_image = Image.open(str(INPUT_IMAGE)).convert("RGBA")
preprocessed = pipeline.preprocess_image(raw_image)

# Get conditioning (same as run_generation_test)
cond_512 = pipeline.get_cond([preprocessed], 512)
cond_1024 = pipeline.get_cond([preprocessed], 1024)

print(f"cond_512 shape: {cond_512['cond'].shape}")
print(f"cond_1024 shape: {cond_1024['cond'].shape}")

# Sample sparse structure first
torch.manual_seed(42)
coords = pipeline.sample_sparse_structure(
    cond_512, 32,
    num_samples=1,
    sampler_params={"steps": 12, "guidance_strength": 7.5, "guidance_rescale": 0.7, "rescale_t": 5.0},
)
print(f"Sparse coords: {coords.shape[0]}")

# Now test the LR shape sampling - this should be fast (~5s total)
print("\n--- Testing LR Shape Sampling (512) ---")
flow_model_lr = pipeline.models['shape_slat_flow_model_512']
noise = SparseTensor(
    feats=torch.randn(coords.shape[0], flow_model_lr.in_channels).to("cuda"),
    coords=coords,
)

t0 = time.time()
with torch.autocast('cuda', dtype=torch.bfloat16, enabled=True):
    slat_lr = pipeline.shape_slat_sampler.sample(
        flow_model_lr,
        noise,
        **cond_512,
        steps=12,
        guidance_strength=7.5,
        guidance_rescale=0.5,
        rescale_t=3.0,
        verbose=True,
        tqdm_desc="LR Shape SLat",
    ).samples
t1 = time.time()
print(f"LR Shape Sampling: {t1-t0:.1f}s")

# Denormalize
std = torch.tensor(pipeline.shape_slat_normalization['std'])[None].to(slat_lr.device)
mean = torch.tensor(pipeline.shape_slat_normalization['mean'])[None].to(slat_lr.device)
slat_lr_denorm = slat_lr * std + mean

# Upsample - this is where we disable autocast
print("\n--- Testing Upsample ---")
decoder = pipeline.models['shape_slat_decoder']
t0 = time.time()
with torch.autocast('cuda', enabled=False):
    if slat_lr_denorm.feats.dtype != torch.float32:
        slat_lr_denorm = slat_lr_denorm.replace(feats=slat_lr_denorm.feats.float())
    hr_coords = decoder.upsample(slat_lr_denorm, upsample_times=4)
t1 = time.time()
print(f"Upsample: {t1-t0:.1f}s")
print(f"HR coords: {hr_coords.shape[0]}")

# Quantize to get final coords
lr_resolution = 512
hr_resolution = 1024
quant_coords = torch.cat([
    hr_coords[:, :1],
    ((hr_coords[:, 1:] + 0.5) / lr_resolution * (hr_resolution // 16)).int(),
], dim=1)
final_coords = quant_coords.unique(dim=0)
print(f"Final tokens: {final_coords.shape[0]}")

# Now test HR shape sampling - this is what's slow
print("\n--- Testing HR Shape Sampling (1024) ---")
flow_model_hr = pipeline.models['shape_slat_flow_model_1024']
noise_hr = SparseTensor(
    feats=torch.randn(final_coords.shape[0], flow_model_hr.in_channels).to("cuda"),
    coords=final_coords,
)

print(f"Is autocast enabled? {torch.is_autocast_enabled()}")
print("Running with autocast enabled...")

t0 = time.time()
with torch.autocast('cuda', dtype=torch.bfloat16, enabled=True):
    print(f"  Autocast now: {torch.is_autocast_enabled()}")
    slat_hr = pipeline.shape_slat_sampler.sample(
        flow_model_hr,
        noise_hr,
        **cond_1024,
        steps=12,
        guidance_strength=7.5,
        guidance_rescale=0.5,
        rescale_t=3.0,
        verbose=True,
        tqdm_desc="HR Shape SLat",
    ).samples
t1 = time.time()
print(f"HR Shape Sampling: {t1-t0:.1f}s")

print("\nDone!")
