"""
Test TRELLIS.2 generation with hybrid precision (BF16 shape, FP32 texture).

Usage:
    .\venv311\Scripts\python.exe test_hybrid_precision.py --image path/to/image.jpg
    .\venv311\Scripts\python.exe test_hybrid_precision.py --image reference/multi-material-ball.png --seed 123
    .\venv311\Scripts\python.exe test_hybrid_precision.py  # Uses default test image
    .\venv311\Scripts\python.exe test_hybrid_precision.py --list  # List reference images
"""
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from pathlib import Path
TRELLIS_ROOT = Path(__file__).parent.parent.parent  # tests/canonical/ -> trellis-forge/
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
import argparse
import json

# Enable cuDNN benchmark
torch.backends.cudnn.benchmark = True

def run_test(seed: int = 42, input_image: Path = None):
    """
    Run TRELLIS.2 generation with hybrid precision.

    Args:
        seed: Random seed for reproducibility.
        input_image: Path to input image. If None, uses default test image.
    """
    # Resolve input image path
    # DEFAULT: Use spiral_input.png for consistent testing against spiral_official.glb benchmark
    if input_image is None:
        INPUT_IMAGE = TRELLIS_ROOT / "reference" / "spiral_input.png"
    else:
        INPUT_IMAGE = Path(input_image)
        # If relative path, try resolving from TRELLIS_ROOT
        if not INPUT_IMAGE.is_absolute() and not INPUT_IMAGE.exists():
            INPUT_IMAGE = TRELLIS_ROOT / input_image

    assert INPUT_IMAGE.exists(), f"Input image not found: {INPUT_IMAGE}"

    print(f"\n{'='*60}")
    print(f"Running TRELLIS.2 with HYBRID PRECISION (BF16 shape, FP32 tex)")
    print(f"Input: {INPUT_IMAGE}")
    print(f"Seed: {seed}")
    print(f"{'='*60}\n")

    print("Loading pipeline...")
    t0 = time.time()
    from trellis2.pipelines.trellis2_image_to_3d import Trellis2ImageTo3DPipeline

    MODEL_PATH = MODELS_DIR / "hub" / "TRELLIS.2-4B"
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(str(MODEL_PATH))

    # Configure high VRAM mode
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    pipeline.configure_vram_mode(vram_gb)
    pipeline.to("cuda")
    pipeline_load_time = time.time() - t0
    print(f"Pipeline loaded in {pipeline_load_time:.1f}s")

    # Load test image
    raw_image = Image.open(str(INPUT_IMAGE)).convert("RGBA")

    # Preprocess
    print("Preprocessing image...")
    t0 = time.time()
    preprocessed = pipeline.preprocess_image(raw_image)
    preprocess_time = time.time() - t0
    print(f"  Preprocessing: {preprocess_time:.1f}s")

    # Parameters matching official defaults
    sparse_params = {
        "steps": 12,
        "guidance_strength": 7.5,
        "guidance_rescale": 0.7,
        "rescale_t": 5.0,
    }
    shape_params = {
        "steps": 12,
        "guidance_strength": 7.5,
        "guidance_rescale": 0.5,
        "rescale_t": 3.0,
    }
    tex_params = {
        "steps": 12,
        "guidance_strength": 1.0,
        "guidance_rescale": 0.0,
        "rescale_t": 3.0,
    }

    # Run pipeline
    print("\n--- Running with HYBRID PRECISION ---")
    print("  Stage 1-2: BF16 (sparse structure + shape)")
    print("  Stage 3: FP32 (texture - color accuracy)")
    print()

    t_total = time.time()
    torch.manual_seed(seed)

    outputs = pipeline.run(
        preprocessed,
        seed=seed,
        preprocess_image=False,
        sparse_structure_sampler_params=sparse_params,
        shape_slat_sampler_params=shape_params,
        tex_slat_sampler_params=tex_params,
        pipeline_type='1024_cascade',
    )

    inference_time = time.time() - t_total
    print(f"\nTotal inference time: {inference_time:.1f}s")

    # Export
    output_dir = TRELLIS_ROOT / "outputs" / "hybrid_precision_test"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"generated_{INPUT_IMAGE.stem}_seed{seed}.glb"

    print(f"\nExporting to {output_path}...")
    t0 = time.time()
    import o_voxel.postprocess
    mesh = outputs[0]
    grid_size = round(1.0 / mesh.voxel_size)
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
        remesh=False,  # Official default - no remeshing
        remesh_band=1,
        remesh_project=0,
        verbose=True,
        use_tqdm=True,
    )
    glb.export(str(output_path), extension_webp=True)
    export_time = time.time() - t0
    print(f"Export time: {export_time:.1f}s")
    print(f"Exported: {output_path}")

    # Save timing report
    report = {
        "precision_mode": "hybrid_bf16_shape_fp32_tex",
        "input_image": str(INPUT_IMAGE),
        "input_name": INPUT_IMAGE.name,
        "seed": seed,
        "pipeline_load_time_s": pipeline_load_time,
        "preprocess_time_s": preprocess_time,
        "inference_time_s": inference_time,
        "export_time_s": export_time,
        "total_time_s": pipeline_load_time + preprocess_time + inference_time + export_time,
        "output_path": str(output_path),
    }
    report_path = output_dir / f"timing_report_{INPUT_IMAGE.stem}_seed{seed}.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"Timing report: {report_path}")

    return output_path, inference_time


def list_reference_images():
    """List available reference images."""
    ref_dir = TRELLIS_ROOT / "reference"
    images = []
    for ext in ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG']:
        images.extend(ref_dir.glob(ext))
    return images


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TRELLIS.2 Hybrid Precision Test (BF16 shape, FP32 texture)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Use default test image
    .\\venv311\\Scripts\\python.exe test_hybrid_precision.py

    # Use specific image
    .\\venv311\\Scripts\\python.exe test_hybrid_precision.py --image reference/multi-material-ball.png

    # Use absolute path with custom seed
    .\\venv311\\Scripts\\python.exe test_hybrid_precision.py --image C:\\path\\to\\image.jpg --seed 123

    # List available reference images
    .\\venv311\\Scripts\\python.exe test_hybrid_precision.py --list
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

    output_path, inference_time = run_test(seed=args.seed, input_image=args.image)

    print("\n" + "="*60)
    print("Test complete.")
    print(f"  Output: {output_path}")
    print(f"  Inference time: {inference_time:.1f}s")
    print("\nCompare visually with:")
    print("  Benchmark: reference/sample_2026-01-24T055452.643.glb")
    print("="*60)
