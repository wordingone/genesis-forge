"""
TRELLIS.2 Generation Test Script
Bypasses FastAPI, directly invokes the pipeline with full per-stage instrumentation.
Outputs: GLB model, preprocessed image, resource report JSON.

Usage:
    .\venv311\Scripts\python.exe run_generation_test.py --image path/to/image.jpg
    .\venv311\Scripts\python.exe run_generation_test.py --image reference/test_input.jpg.JPG
    .\venv311\Scripts\python.exe run_generation_test.py  # Uses default test image
"""
import os
import sys
import json
import time
import argparse
from pathlib import Path

# ===== CUDA ALLOCATOR CONFIG - Must be set BEFORE any torch imports =====
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

# Add paths
sys.path.insert(0, str(TRELLIS_ROOT))
o_voxel_parent = TRELLIS_ROOT / "o_voxel"
if o_voxel_parent.exists():
    sys.path.insert(0, str(o_voxel_parent))

import torch
import psutil
from PIL import Image

# ===== PERFORMANCE OPTIMIZATION: Enable cuDNN benchmark =====
torch.backends.cudnn.benchmark = True


# ===== Resource Monitor =====
class ResourceMonitor:
    """Tracks GPU and system memory per-stage."""

    def __init__(self):
        self.stages = []
        self._current = None

    def start_stage(self, name: str):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        self._current = {
            "name": name,
            "start_time": time.time(),
            "gpu_allocated_before_mb": torch.cuda.memory_allocated() / 1024**2,
            "gpu_reserved_before_mb": torch.cuda.memory_reserved() / 1024**2,
            "ram_before_mb": psutil.Process().memory_info().rss / 1024**2,
            "system_ram_percent_before": psutil.virtual_memory().percent,
        }
        print(f"\n{'='*60}")
        print(f"  STAGE: {name}")
        print(f"  GPU Allocated: {self._current['gpu_allocated_before_mb']:.0f} MB")
        print(f"  System RAM: {self._current['system_ram_percent_before']:.1f}%")
        print(f"{'='*60}")

    def end_stage(self):
        torch.cuda.synchronize()
        c = self._current
        c["end_time"] = time.time()
        c["elapsed_s"] = c["end_time"] - c["start_time"]
        c["gpu_allocated_after_mb"] = torch.cuda.memory_allocated() / 1024**2
        c["gpu_reserved_after_mb"] = torch.cuda.memory_reserved() / 1024**2
        c["gpu_peak_allocated_mb"] = torch.cuda.max_memory_allocated() / 1024**2
        c["gpu_peak_reserved_mb"] = torch.cuda.max_memory_reserved() / 1024**2
        c["ram_after_mb"] = psutil.Process().memory_info().rss / 1024**2
        c["system_ram_percent_after"] = psutil.virtual_memory().percent

        print(f"  -> Elapsed: {c['elapsed_s']:.1f}s")
        print(f"  -> GPU Peak Allocated: {c['gpu_peak_allocated_mb']:.0f} MB")
        print(f"  -> GPU Peak Reserved: {c['gpu_peak_reserved_mb']:.0f} MB")
        print(f"  -> Process RAM: {c['ram_after_mb']:.0f} MB")

        self.stages.append(c)
        self._current = None

    def summary(self) -> dict:
        total_time = sum(s["elapsed_s"] for s in self.stages)
        overall_peak_gpu = max(s["gpu_peak_allocated_mb"] for s in self.stages) if self.stages else 0
        overall_peak_reserved = max(s["gpu_peak_reserved_mb"] for s in self.stages) if self.stages else 0
        peak_ram = max(s["ram_after_mb"] for s in self.stages) if self.stages else 0

        return {
            "total_time_s": total_time,
            "overall_peak_gpu_allocated_mb": overall_peak_gpu,
            "overall_peak_gpu_reserved_mb": overall_peak_reserved,
            "peak_process_ram_mb": peak_ram,
            "stages": [
                {
                    "name": s["name"],
                    "elapsed_s": round(s["elapsed_s"], 2),
                    "gpu_peak_allocated_mb": round(s["gpu_peak_allocated_mb"], 1),
                    "gpu_peak_reserved_mb": round(s["gpu_peak_reserved_mb"], 1),
                    "ram_after_mb": round(s["ram_after_mb"], 1),
                }
                for s in self.stages
            ],
        }


def main(input_image: Path = None, seed: int = 42, output_name: str = None):
    """
    Run TRELLIS.2 generation test.

    Args:
        input_image: Path to input image. If None, uses default test image.
        seed: Random seed for reproducibility.
        output_name: Custom output filename (without extension). If None, uses 'generated_output'.
    """
    OUTPUT_DIR = TRELLIS_ROOT / "outputs" / "generation_test"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Resolve input image path
    if input_image is None:
        INPUT_IMAGE = TRELLIS_ROOT / "reference" / "test_input.jpg.JPG"
    else:
        INPUT_IMAGE = Path(input_image)
        # If relative path, try resolving from TRELLIS_ROOT
        if not INPUT_IMAGE.is_absolute() and not INPUT_IMAGE.exists():
            INPUT_IMAGE = TRELLIS_ROOT / input_image

    MODEL_PATH = MODELS_DIR / "hub" / "TRELLIS.2-4B"

    assert INPUT_IMAGE.exists(), f"Input image not found: {INPUT_IMAGE}"
    assert MODEL_PATH.exists(), f"Model not found: {MODEL_PATH}"

    print(f"\n{'='*60}")
    print(f"  TRELLIS.2 Generation Test")
    print(f"  Input: {INPUT_IMAGE}")
    print(f"  Seed: {seed}")
    print(f"{'='*60}")

    monitor = ResourceMonitor()

    # ===== Stage 0: Pipeline Loading =====
    monitor.start_stage("Pipeline Loading")
    from trellis2.pipelines.trellis2_image_to_3d import Trellis2ImageTo3DPipeline
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(str(MODEL_PATH))

    # Configure VRAM mode based on GPU memory (disables low_vram for RTX 4090)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    pipeline.configure_vram_mode(vram_gb)

    pipeline.to("cuda")
    monitor.end_stage()

    # ===== Stage 1: Image Preprocessing =====
    monitor.start_stage("Image Preprocessing")
    raw_image = Image.open(str(INPUT_IMAGE)).convert("RGBA")
    preprocessed = pipeline.preprocess_image(raw_image)
    preprocessed.save(str(OUTPUT_DIR / "preprocessed.png"))
    print(f"  Preprocessed image size: {preprocessed.size}")
    monitor.end_stage()

    # ===== Official parameters =====
    SEED = seed
    PIPELINE_TYPE = "1024_cascade"
    SPARSE_PARAMS = {
        "steps": 12,
        "guidance_strength": 7.5,
        "guidance_rescale": 0.7,
        "rescale_t": 5.0,
    }
    SHAPE_PARAMS = {
        "steps": 12,
        "guidance_strength": 7.5,
        "guidance_rescale": 0.5,
        "rescale_t": 3.0,
    }
    TEX_PARAMS = {
        "steps": 12,
        "guidance_strength": 1.0,
        "guidance_rescale": 0.0,
        "rescale_t": 3.0,
    }

    torch.manual_seed(SEED)

    # ===== Stage 2: Image Conditioning =====
    monitor.start_stage("Image Conditioning (DinoV2)")
    cond_512 = pipeline.get_cond([preprocessed], 512)
    cond_1024 = pipeline.get_cond([preprocessed], 1024)
    monitor.end_stage()

    # ===== REMOVED AUTOCAST =====
    # Previous optimization used autocast(bfloat16) but this causes color mismatch
    # with official output. Official runs without autocast (default precision).
    # Color comparison:
    #   With autocast(bf16): R=154.8, G=170.1, B=184.7 (washed out)
    #   Official (no autocast): R=108.9, G=122.2, B=141.6 (richer colors)
    print(f"\n  Autocast enabled: {torch.is_autocast_enabled()} (should be False)")

    # ===== Stage 3: Sparse Structure Sampling =====
    monitor.start_stage("Stage 1: Sparse Structure Sampling")
    coords = pipeline.sample_sparse_structure(
        cond_512, 32,
        num_samples=1,
        sampler_params=SPARSE_PARAMS,
    )
    print(f"  Sparse structure coords: {coords.shape}")
    monitor.end_stage()

    # ===== Stage 4: Shape SLat Cascade (512->1024) =====
    monitor.start_stage("Stage 2: Shape SLat Cascade (512->1024)")
    shape_slat, hr_resolution = pipeline.sample_shape_slat_cascade(
        cond_512, cond_1024,
        pipeline.models['shape_slat_flow_model_512'],
        pipeline.models['shape_slat_flow_model_1024'],
        512, 1024,
        coords,
        sampler_params=SHAPE_PARAMS,
    )
    print(f"  Shape SLat features: {shape_slat.feats.shape}")
    print(f"  HR resolution: {hr_resolution}")
    monitor.end_stage()

    # ===== Stage 5: Texture SLat Sampling =====
    monitor.start_stage("Stage 3: Texture SLat Sampling")
    tex_slat = pipeline.sample_tex_slat(
        cond_1024,
        pipeline.models['tex_slat_flow_model_1024'],
        shape_slat,
        sampler_params=TEX_PARAMS,
    )
    print(f"  Tex SLat features: {tex_slat.feats.shape}")
    monitor.end_stage()

    torch.cuda.empty_cache()

    # Decoding happens outside autocast - mesh extraction may need FP32 precision
    # ===== Stage 6: Decode Shape (mesh extraction) =====
    monitor.start_stage("Decode Shape + Texture")
    out_meshes = pipeline.decode_latent(shape_slat, tex_slat, hr_resolution)
    mesh = out_meshes[0]
    print(f"  Mesh vertices: {mesh.vertices.shape}")
    print(f"  Mesh faces: {mesh.faces.shape}")
    print(f"  Voxel coords: {mesh.coords.shape}")
    print(f"  Attr volume: {mesh.attrs.shape}")
    monitor.end_stage()

    # ===== Stage 7: GLB Export =====
    monitor.start_stage("GLB Export (o_voxel.postprocess.to_glb)")
    import o_voxel.postprocess
    grid_size = round(1.0 / mesh.voxel_size)
    print(f"  Grid size: {grid_size}")
    print(f"  Decimation target: 500000")
    print(f"  Texture size: 2048")

    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=mesh.layout,
        grid_size=grid_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=500000,
        texture_size=2048,
        remesh=True,
        remesh_band=1,
        remesh_project=0,
        verbose=True,
        use_tqdm=True,
    )
    # Determine output filename
    if output_name:
        glb_filename = f"{output_name}.glb"
    else:
        glb_filename = f"generated_{INPUT_IMAGE.stem}.glb"
    glb_path = str(OUTPUT_DIR / glb_filename)
    glb.export(glb_path, extension_webp=True)
    glb_size_mb = os.path.getsize(glb_path) / 1024**2
    print(f"  GLB saved: {glb_path} ({glb_size_mb:.1f} MB)")
    monitor.end_stage()

    # ===== Summary =====
    report = monitor.summary()
    report["input"] = {
        "image_path": str(INPUT_IMAGE),
        "image_name": INPUT_IMAGE.name,
    }
    report["parameters"] = {
        "seed": SEED,
        "pipeline_type": PIPELINE_TYPE,
        "sparse_params": SPARSE_PARAMS,
        "shape_params": SHAPE_PARAMS,
        "tex_params": TEX_PARAMS,
        "decimation_target": 500000,
        "texture_size": 2048,
        "remesh": True,
        "remesh_band": 1,
        "remesh_project": 0,
    }
    report["output"] = {
        "glb_path": glb_path,
        "glb_size_mb": round(glb_size_mb, 2),
    }

    report_path = OUTPUT_DIR / "resource_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nResource report saved: {report_path}")

    # Print summary table
    print(f"\n{'='*70}")
    print(f"  TRELLIS.2 Generation Report")
    print(f"{'='*70}")
    print(f"  {'Stage':<45} {'Time':>8} {'GPU Peak':>10}")
    print(f"  {'-'*45} {'-'*8} {'-'*10}")
    for s in report["stages"]:
        print(f"  {s['name']:<45} {s['elapsed_s']:>7.1f}s {s['gpu_peak_allocated_mb']:>8.0f} MB")
    print(f"  {'-'*45} {'-'*8} {'-'*10}")
    print(f"  {'TOTAL':<45} {report['total_time_s']:>7.1f}s {report['overall_peak_gpu_allocated_mb']:>8.0f} MB peak")
    print(f"{'='*70}")

    # Threshold checks
    peak_gpu = report["overall_peak_gpu_allocated_mb"]
    total_time = report["total_time_s"]
    print(f"\n  Threshold checks:")
    print(f"    Peak GPU: {peak_gpu:.0f} MB {'PASS' if peak_gpu < 16000 else 'FAIL (>16 GB)'}")
    print(f"    Total time: {total_time:.0f}s {'PASS' if total_time < 300 else 'FAIL (>300s)'}")


def list_reference_images():
    """List available reference images."""
    ref_dir = TRELLIS_ROOT / "reference"
    images = []
    for ext in ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG']:
        images.extend(ref_dir.glob(ext))
    return images


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TRELLIS.2 Generation Test with full instrumentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Use default test image
    .\\venv311\\Scripts\\python.exe run_generation_test.py

    # Use specific image
    .\\venv311\\Scripts\\python.exe run_generation_test.py --image reference/multi-material-ball.png

    # Use absolute path
    .\\venv311\\Scripts\\python.exe run_generation_test.py --image C:\\path\\to\\image.jpg

    # Custom seed and output name
    .\\venv311\\Scripts\\python.exe run_generation_test.py --image my_image.jpg --seed 123 --output my_test

    # List available reference images
    .\\venv311\\Scripts\\python.exe run_generation_test.py --list
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
        "--seed", "-s",
        type=int,
        default=42,
        help="Random seed for reproducibility. Default: 42"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output filename (without extension). Default: generated_<input_name>"
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

    main(input_image=args.image, seed=args.seed, output_name=args.output)
