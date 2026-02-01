"""
TRELLIS.2 Performance Diagnostic Script
Pinpoints exactly where the 78-minute runtime is coming from.

Usage:
    .\venv311\Scripts\python.exe diagnose_performance.py
    .\venv311\Scripts\python.exe diagnose_performance.py --image path/to/image.jpg
    .\venv311\Scripts\python.exe diagnose_performance.py --list
"""
import os
import sys
import time
import json
import argparse
from pathlib import Path

# ===== CUDA ALLOCATOR CONFIG =====
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# ===== LOCAL MODEL CACHE =====
TRELLIS_ROOT = Path(__file__).parent
MODELS_DIR = TRELLIS_ROOT / "models"
os.environ['HF_HOME'] = str(MODELS_DIR)
os.environ['HF_HUB_CACHE'] = str(MODELS_DIR / "hub")
os.environ['HUGGINGFACE_HUB_CACHE'] = str(MODELS_DIR / "hub")
os.environ['TRANSFORMERS_CACHE'] = str(MODELS_DIR / "hub")
os.environ['TORCH_HOME'] = str(MODELS_DIR / "torch_hub")
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

sys.path.insert(0, str(TRELLIS_ROOT))
o_voxel_parent = TRELLIS_ROOT / "o_voxel"
if o_voxel_parent.exists():
    sys.path.insert(0, str(o_voxel_parent))

import torch
import numpy as np
from PIL import Image

# ===== PERFORMANCE OPTIMIZATION: Enable cuDNN benchmark =====
torch.backends.cudnn.benchmark = True


def list_reference_images():
    """List available reference images."""
    ref_dir = TRELLIS_ROOT / "reference"
    images = []
    for ext in ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG']:
        images.extend(ref_dir.glob(ext))
    return images


def run_diagnostics(INPUT_IMAGE: Path):
    """Run performance diagnostics with the specified input image."""
    print("="*70)
    print("TRELLIS.2 PERFORMANCE DIAGNOSTICS")
    print(f"Input Image: {INPUT_IMAGE}")
    print("="*70)

    # ===== TEST 1: CUDA and Backend Verification =====
    print("\n[TEST 1] CUDA and Backend Verification")
    print("-"*50)

    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    print(f"  CUDA version: {torch.version.cuda}")
    print(f"  cuDNN version: {torch.backends.cudnn.version()}")
    print(f"  cuDNN enabled: {torch.backends.cudnn.enabled}")
    print(f"  cuDNN benchmark: {torch.backends.cudnn.benchmark} {'[OPTIMIZED]' if torch.backends.cudnn.benchmark else '[WARNING: Should be True]'}")

    # Check device
    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    print(f"  GPU: {props.name}")
    print(f"  VRAM: {props.total_memory / 1024**3:.1f} GB")
    print(f"  Compute capability: {props.major}.{props.minor}")

    # Check BF16 support
    print(f"  BF16 supported: {torch.cuda.is_bf16_supported()}")

    # Test autocast functionality (critical for manual_cast bypass)
    print("\n  Testing autocast functionality:")
    print(f"    Autocast outside context: {torch.is_autocast_enabled()}")
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=True):
        print(f"    Autocast inside context: {torch.is_autocast_enabled()} {'[OPTIMIZED]' if torch.is_autocast_enabled() else '[ERROR]'}")

    # ===== TEST 2: Sparse Conv Backend =====
    print("\n[TEST 2] Sparse Convolution Backend")
    print("-"*50)

    try:
        import flex_gemm
        print(f"  flex_gemm imported: YES")
        print(f"  flex_gemm version: {getattr(flex_gemm, '__version__', 'unknown')}")

        # Check algorithm settings
        from trellis2.modules.sparse.conv import config as conv_config
        print(f"  SPCONV_ALGO: {conv_config.SPCONV_ALGO}")
        print(f"  FLEX_GEMM_ALGO: {conv_config.FLEX_GEMM_ALGO}")
        print(f"  FLEX_GEMM_HASHMAP_RATIO: {conv_config.FLEX_GEMM_HASHMAP_RATIO}")

        # Test flex_gemm functionality
        from flex_gemm.ops.spconv import sparse_submanifold_conv3d
        print(f"  sparse_submanifold_conv3d: AVAILABLE")
    except Exception as e:
        print(f"  flex_gemm: FAILED - {e}")

    # ===== TEST 3: Attention Backend =====
    print("\n[TEST 3] Attention Backend")
    print("-"*50)

    sparse_config = None
    try:
        from trellis2.modules.sparse import config as sparse_config
        print(f"  CONV backend: {sparse_config.CONV}")
        print(f"  ATTN backend: {sparse_config.ATTN}")
    except Exception as e:
        print(f"  Config load failed: {e}")

    try:
        import flash_attn
        print(f"  flash_attn imported: YES")
        print(f"  flash_attn version: {flash_attn.__version__}")

        # Test flash_attn functionality
        from flash_attn import flash_attn_varlen_qkvpacked_func
        print(f"  flash_attn_varlen_qkvpacked_func: AVAILABLE")

        # Quick functional test
        qkv = torch.randn(128, 3, 12, 64, dtype=torch.float16, device='cuda')
        cu_seqlens = torch.tensor([0, 128], dtype=torch.int32, device='cuda')
        out = flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens, 128)
        print(f"  flash_attn functional test: PASS")
    except Exception as e:
        print(f"  flash_attn: FAILED - {e}")

    # ===== TEST 4: Model Precision =====
    print("\n[TEST 4] Model Precision Analysis")
    print("-"*50)

    from trellis2.pipelines.trellis2_image_to_3d import Trellis2ImageTo3DPipeline
    MODEL_PATH = MODELS_DIR / "hub" / "TRELLIS.2-4B"

    print("  Loading pipeline (this will show backend messages)...")
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(str(MODEL_PATH))

    # Check model dtypes
    for name, model in pipeline.models.items():
        if model is not None:
            param = next(model.parameters(), None)
            if param is not None:
                print(f"  {name}: {param.dtype}")

    # ===== TEST 5: Token Count Analysis =====
    print("\n[TEST 5] Token Count Analysis (Key Metric)")
    print("-"*50)

    pipeline.to("cuda")

    # Load and preprocess test image
    raw_image = Image.open(str(INPUT_IMAGE)).convert("RGBA")
    preprocessed = pipeline.preprocess_image(raw_image)

    print(f"  Input image: {INPUT_IMAGE.name}")
    print(f"  Preprocessed size: {preprocessed.size}")

    # Get conditioning
    torch.manual_seed(42)
    cond_512 = pipeline.get_cond([preprocessed], 512)
    cond_1024 = pipeline.get_cond([preprocessed], 1024)

    # Sample sparse structure
    print("\n  Sampling sparse structure...")
    t0 = time.time()
    coords = pipeline.sample_sparse_structure(
        cond_512, 32,
        num_samples=1,
        sampler_params={"steps": 12, "guidance_strength": 7.5, "guidance_rescale": 0.7, "rescale_t": 5.0},
    )
    t1 = time.time()
    print(f"  Sparse structure time: {t1-t0:.2f}s")
    print(f"  Sparse coords shape: {coords.shape}")
    print(f"  Sparse coords count: {coords.shape[0]}")

    # Analyze expected token count after upsampling
    # This is the KEY metric - how many tokens will the 1024 model process?
    print("\n  Analyzing cascade token counts...")

    # Simulate what sample_shape_slat_cascade does for token counting
    from trellis2.modules.sparse import SparseTensor

    # Create initial noise for LR pass
    noise_lr = SparseTensor(
        feats=torch.randn(coords.shape[0], 32).to("cuda"),
        coords=coords,
    )
    print(f"  LR (512) tokens: {coords.shape[0]}")

    # Move decoder to GPU temporarily to run upsample
    pipeline.models['shape_slat_decoder'].to("cuda")
    pipeline.models['shape_slat_decoder'].low_vram = True

    # Sample LR shape slat (quick - just 12 steps at 512)
    print("\n  Running LR shape sampling (512 resolution)...")
    t0 = time.time()
    flow_model_lr = pipeline.models['shape_slat_flow_model_512']
    flow_model_lr.to("cuda")

    slat_lr = pipeline.shape_slat_sampler.sample(
        flow_model_lr,
        noise_lr,
        **cond_512,
        steps=12,
        guidance_strength=7.5,
        guidance_rescale=0.5,
        rescale_t=3.0,
        verbose=True,
        tqdm_desc="LR Shape SLat",
    ).samples

    flow_model_lr.cpu()
    t1 = time.time()
    print(f"  LR shape sampling time: {t1-t0:.2f}s")

    # Denormalize
    std = torch.tensor(pipeline.shape_slat_normalization['std'])[None].to(slat_lr.device)
    mean = torch.tensor(pipeline.shape_slat_normalization['mean'])[None].to(slat_lr.device)
    slat_lr_denorm = slat_lr * std + mean

    # Upsample to get HR coords
    print("\n  Upsampling to 1024 resolution...")
    t0 = time.time()
    hr_coords = pipeline.models['shape_slat_decoder'].upsample(slat_lr_denorm, upsample_times=4)
    t1 = time.time()
    print(f"  Upsample time: {t1-t0:.2f}s")
    print(f"  Raw HR coords count: {hr_coords.shape[0]}")

    pipeline.models['shape_slat_decoder'].cpu()

    # Quantize and deduplicate (this is what the cascade does)
    lr_resolution = 512
    hr_resolution = 1024
    quant_coords = torch.cat([
        hr_coords[:, :1],
        ((hr_coords[:, 1:] + 0.5) / lr_resolution * (hr_resolution // 16)).int(),
    ], dim=1)
    unique_coords = quant_coords.unique(dim=0)
    final_token_count = unique_coords.shape[0]

    print(f"\n  === TOKEN COUNT ANALYSIS ===")
    print(f"  Initial sparse coords: {coords.shape[0]}")
    print(f"  After 4x upsample: {hr_coords.shape[0]}")
    print(f"  After quantization: {final_token_count}")
    print(f"  Expansion factor: {final_token_count / coords.shape[0]:.1f}x")

    # Compare to expected
    print(f"\n  For reference:")
    print(f"    512³ dense would be: {512**3 // 16**3:,} = {512**3 // 16**3} tokens (32³ latent)")
    print(f"    1024³ dense would be: {1024**3 // 16**3:,} = {1024**3 // 16**3} tokens (64³ latent)")
    print(f"    Our token count: {final_token_count:,}")
    print(f"    Sparsity ratio: {final_token_count / (1024**3 // 16**3) * 100:.2f}% of dense")

    # ===== TEST 6: Single Step Timing =====
    print("\n[TEST 6] Single DiT Forward Pass Timing")
    print("-"*50)

    # Create noise for a single step test
    test_coords = unique_coords[:min(10000, final_token_count)]  # Limit for quick test
    noise_hr = SparseTensor(
        feats=torch.randn(test_coords.shape[0], 32).to("cuda"),
        coords=test_coords.to("cuda"),
    )

    flow_model_hr = pipeline.models['shape_slat_flow_model_1024']
    flow_model_hr.to("cuda")

    # Warmup
    with torch.no_grad():
        t = torch.tensor([500.0], device='cuda')
        _ = flow_model_hr(noise_hr, t, cond_1024['cond'])

    torch.cuda.synchronize()

    # Time single forward pass
    print(f"  Testing with {test_coords.shape[0]} tokens...")
    times = []
    for i in range(5):
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            t = torch.tensor([500.0], device='cuda')
            _ = flow_model_hr(noise_hr, t, cond_1024['cond'])
        torch.cuda.synchronize()
        t1 = time.time()
        times.append(t1 - t0)
        print(f"    Run {i+1}: {times[-1]*1000:.1f}ms")

    avg_time = np.mean(times[1:])  # Skip first (potential warmup)
    print(f"\n  Average forward pass: {avg_time*1000:.1f}ms")
    print(f"  With {final_token_count} tokens (scaled): {avg_time * (final_token_count / test_coords.shape[0])*1000:.1f}ms")
    print(f"  12 steps x 2 (CFG): {avg_time * (final_token_count / test_coords.shape[0]) * 12 * 2:.1f}s")

    flow_model_hr.cpu()

    # ===== TEST 7: Memory Fragmentation =====
    print("\n[TEST 7] Memory Analysis")
    print("-"*50)

    print(f"  Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"  Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    print(f"  Max allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

    # Check expandable_segments
    allocator_settings = torch.cuda.memory.get_allocator_backend()
    print(f"  Allocator backend: {allocator_settings}")

    # ===== TEST 8: Profiling a Full Stage =====
    print("\n[TEST 8] Profile Full 512 Shape Sampling")
    print("-"*50)

    # This is the fastest stage - if this is slow, we have a fundamental issue
    pipeline.models['shape_slat_flow_model_512'].to("cuda")

    noise_512 = SparseTensor(
        feats=torch.randn(coords.shape[0], 32).to("cuda"),
        coords=coords,
    )

    print(f"  Profiling 12 steps with {coords.shape[0]} tokens...")

    # Enable profiling
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True,
    ) as prof:
        torch.cuda.synchronize()
        t0 = time.time()
        slat_test = pipeline.shape_slat_sampler.sample(
            pipeline.models['shape_slat_flow_model_512'],
            noise_512,
            **cond_512,
            steps=12,
            guidance_strength=7.5,
            guidance_rescale=0.5,
            rescale_t=3.0,
            verbose=False,
        ).samples
        torch.cuda.synchronize()
        t1 = time.time()

    print(f"  Total time: {t1-t0:.2f}s")
    print(f"  Time per step: {(t1-t0)/12*1000:.1f}ms")

    # Print top operations
    print("\n  Top 10 CUDA operations by time:")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    # Save full profile
    profile_path = TRELLIS_ROOT / "outputs" / "generation_test" / "profile.json"
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    prof.export_chrome_trace(str(profile_path))
    print(f"\n  Full profile saved to: {profile_path}")

    pipeline.models['shape_slat_flow_model_512'].cpu()

    # ===== SUMMARY =====
    print("\n" + "="*70)
    print("DIAGNOSTIC SUMMARY")
    print("="*70)

    attn_backend = sparse_config.ATTN if sparse_config else "unknown"
    conv_backend = sparse_config.CONV if sparse_config else "unknown"

    print(f"""
Key Findings:
  - Token count after cascade: {final_token_count:,}
  - Sparsity: {final_token_count / (1024**3 // 16**3) * 100:.2f}% of dense 1024³
  - Attention backend: {attn_backend}
  - Conv backend: {conv_backend}
  - Model dtype: {next(pipeline.models['shape_slat_flow_model_1024'].parameters()).dtype}

If token count is >50k, the cascade is not exploiting sparsity properly.
If single forward pass is >500ms, there's a compute bottleneck.
If attention backend is not flash_attn, that's a major slowdown.
Check profile.json in Chrome (chrome://tracing) for detailed breakdown.
""")

    torch.cuda.empty_cache()
    print("Diagnostics complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TRELLIS.2 Performance Diagnostics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Use default test image
    .\\venv311\\Scripts\\python.exe diagnose_performance.py

    # Use specific image
    .\\venv311\\Scripts\\python.exe diagnose_performance.py --image reference/multi-material-ball.png

    # List available reference images
    .\\venv311\\Scripts\\python.exe diagnose_performance.py --list
        """
    )
    parser.add_argument(
        "--image", "-i",
        type=str,
        default=None,
        help="Path to input image. Can be absolute or relative to trellis-forge directory. "
             "Default: reference/test_input.jpg.JPG"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available reference images and exit"
    )

    args = parser.parse_args()

    if args.list:
        print("\nAvailable reference images:")
        print("-" * 40)
        for img in list_reference_images():
            print(f"  {img.relative_to(TRELLIS_ROOT)}")
        print("\nUsage: --image reference/<filename>")
        sys.exit(0)

    # Resolve input image path
    if args.image is None:
        INPUT_IMAGE = TRELLIS_ROOT / "reference" / "test_input.jpg.JPG"
        if not INPUT_IMAGE.exists():
            INPUT_IMAGE = TRELLIS_ROOT / "reference" / "test_input.jpg"
    else:
        INPUT_IMAGE = Path(args.image)
        if not INPUT_IMAGE.is_absolute() and not INPUT_IMAGE.exists():
            INPUT_IMAGE = TRELLIS_ROOT / args.image

    assert INPUT_IMAGE.exists(), f"Input image not found: {INPUT_IMAGE}"

    run_diagnostics(INPUT_IMAGE)
