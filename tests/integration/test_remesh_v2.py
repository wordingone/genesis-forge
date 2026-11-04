"""Test TRELLIS.2 pipeline with BiRefNet preprocessing.

Strategy for Windows (no expandable_segments):
  1. Preprocess image with BiRefNet (separate GPU phase)
  2. Fully release BiRefNet GPU memory
  3. Run TRELLIS.2 pipeline with clean GPU state

IMPORTANT: Run with venv python:
  cd <repo>
  .\\venv311\\Scripts\\python.exe test_remesh_v2.py
"""
import os
import sys
import time
import gc
import argparse

# Add local source paths
sys.path.insert(0, 'o_voxel')
sys.path.insert(0, 'cumesh_source')

os.environ['HF_HOME'] = os.path.abspath('./models')
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
# Note: expandable_segments not supported on Windows; no custom CUDA alloc config

import torch
import numpy as np
from PIL import Image


def get_gpu_memory():
    """Get current GPU memory usage in GB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3
    return 0


def preprocess_image_standalone(image_path):
    """Run BiRefNet background removal as a standalone step.

    This preprocesses the image BEFORE loading the TRELLIS pipeline,
    so BiRefNet's GPU memory is fully released before decode phase.
    """
    from trellis2.pipelines.rembg.BiRefNet import BiRefNet

    print(f"\n[PREPROCESS] Loading BiRefNet...")
    t0 = time.time()
    rembg = BiRefNet("ZhengPeng7/BiRefNet")
    rembg.to("cuda")
    print(f"[PREPROCESS] BiRefNet loaded ({time.time()-t0:.1f}s), GPU: {get_gpu_memory():.2f} GB")

    # Load and preprocess
    image = Image.open(image_path).convert('RGB')
    print(f"[PREPROCESS] Input: {image.size}")

    # Resize if needed (matching pipeline's preprocess_image)
    max_size = max(image.size)
    scale = min(1, 1024 / max_size)
    if scale < 1:
        image = image.resize((int(image.width * scale), int(image.height * scale)), Image.Resampling.LANCZOS)
        print(f"[PREPROCESS] Resized to: {image.size}")

    # Run background removal
    result = rembg(image)
    print(f"[PREPROCESS] Background removed, GPU: {get_gpu_memory():.2f} GB")

    # Process alpha (matching pipeline's preprocess_image)
    output_np = np.array(result)
    alpha = output_np[:, :, 3]
    bbox = np.argwhere(alpha > 0.8 * 255)
    bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
    center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
    size = int(size * 1)
    bbox = center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2
    output = result.crop(bbox)
    output = np.array(output).astype(np.float32) / 255
    output = output[:, :, :3] * output[:, :, 3:4]
    output = Image.fromarray((output * 255).astype(np.uint8))

    # Fully release BiRefNet
    rembg.cpu()
    del rembg
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    print(f"[PREPROCESS] BiRefNet released, GPU: {get_gpu_memory():.2f} GB")

    return output


def main():
    parser = argparse.ArgumentParser(description='Test TRELLIS.2 pipeline v2')
    parser.add_argument('--resolution', type=str, default='1024',
                        choices=['512', '1024', '1024_cascade', '1536_cascade'],
                        help='Pipeline resolution (default: 1024)')
    parser.add_argument('--input', type=str, default='reference/test_input.jpg.JPG',
                        help='Input image path')
    parser.add_argument('--output', type=str, default='outputs/test_output.glb',
                        help='Output GLB path')
    parser.add_argument('--texture-size', type=int, default=2048,
                        help='Texture resolution (default: 2048)')
    parser.add_argument('--decimation', type=int, default=1000000,
                        help='Target face count (default: 1000000)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()

    print("=" * 80)
    print(f"TEST: TRELLIS.2 Pipeline v2 (resolution={args.resolution})")
    print("=" * 80)

    # Map simple resolution to pipeline_type
    pipeline_type = args.resolution
    if pipeline_type == '1024':
        pipeline_type = '1024_cascade'  # Match official default

    print(f"\n[CONFIG]")
    print(f"  Resolution: {args.resolution} -> pipeline_type={pipeline_type}")
    print(f"  Input: {args.input}")
    print(f"  Output: {args.output}")
    print(f"  Texture size: {args.texture_size}")
    print(f"  Decimation target: {args.decimation}")
    print(f"  Seed: {args.seed}")

    # PHASE 1: Preprocess image with BiRefNet (separate GPU phase)
    if not os.path.exists(args.input):
        print(f"[ERROR] Input image not found: {args.input}")
        return
    preprocessed = preprocess_image_standalone(args.input)
    print(f"[PREPROCESS] Preprocessed image: {preprocessed.size}, mode={preprocessed.mode}")

    # PHASE 2: Load and run TRELLIS.2 pipeline
    from trellis2.pipelines.trellis2_image_to_3d import Trellis2ImageTo3DPipeline
    print("\n[LOADING] Pipeline...")
    t0 = time.time()
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained('./models/hub/TRELLIS.2-4B')
    pipeline.cuda()
    print(f"[LOADED] Pipeline ready ({time.time()-t0:.1f}s)")
    print(f"[MEMORY] GPU: {get_gpu_memory():.2f} GB")

    # Run with pre-processed image (skip preprocessing in pipeline)
    print(f"\n[RUNNING] Generating mesh (pipeline_type={pipeline_type})...")
    t0 = time.time()
    outputs = pipeline.run(
        preprocessed,
        seed=args.seed,
        sparse_structure_sampler_params={
            "steps": 12,
            "guidance_strength": 7.5,
        },
        shape_slat_sampler_params={
            "steps": 12,
            "guidance_strength": 3.0,
        },
        tex_slat_sampler_params={
            "steps": 12,
        },
        pipeline_type=pipeline_type,
        max_num_tokens=50000,
        preprocess_image=False,  # Already preprocessed
    )
    mesh = outputs[0]
    gen_time = time.time() - t0

    print(f"\n[MESH INFO]")
    print(f"  Generation time: {gen_time:.1f}s")
    print(f"  Vertices: {mesh.vertices.shape}")
    print(f"  Faces: {mesh.faces.shape}")
    print(f"  Coords: {mesh.coords.shape}")
    print(f"  Attrs: {mesh.attrs.shape}")
    print(f"  Voxel shape: {mesh.voxel_shape}")
    print(f"[MEMORY] GPU: {get_gpu_memory():.2f} GB (peak: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB)")

    # Export
    os.makedirs(os.path.dirname(args.output) or 'outputs', exist_ok=True)
    print(f"\n[EXPORTING] Writing {args.output}...")
    t0 = time.time()
    mesh.export(
        args.output,
        decimation_target=args.decimation,
        texture_size=args.texture_size,
        remesh=True,
        remesh_band=1.0,
        remesh_project=0.0,
        verbose=True
    )
    export_time = time.time() - t0

    # Check output
    if os.path.exists(args.output):
        size_mb = os.path.getsize(args.output) / 1024**2
        print(f"\n[SUCCESS] Output: {args.output} ({size_mb:.2f} MB)")
    else:
        print(f"\n[ERROR] Output file not created!")
        return

    print(f"\n[TIMING SUMMARY]")
    print(f"  Generation: {gen_time:.1f}s")
    print(f"  Export: {export_time:.1f}s")
    print(f"  Total: {gen_time + export_time:.1f}s")
    print(f"[MEMORY] Peak GPU: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
    print(f"\nVisual inspection required:")
    print(f"  1. Open {args.output} in a 3D viewer")
    print(f"  2. Compare with reference/sample_2026-01-24T055452.643.glb")
    print(f"  3. Score: Shape, Texture, Color, PBR (must be 8+/10 each)")


if __name__ == '__main__':
    main()
