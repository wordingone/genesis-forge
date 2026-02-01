"""Test TRELLIS.2 generation WITHOUT autocast to verify if quality issues are autocast-related."""
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
import argparse

# Enable cuDNN benchmark
torch.backends.cudnn.benchmark = True

def run_test(use_autocast: bool = False, seed: int = 42):
    print(f"\n{'='*60}")
    print(f"Running TRELLIS.2 test with autocast={use_autocast}, seed={seed}")
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
    print(f"Pipeline loaded in {time.time()-t0:.1f}s")

    # Load test image
    INPUT_IMAGE = TRELLIS_ROOT / "reference" / "test_input.jpg.JPG"
    raw_image = Image.open(str(INPUT_IMAGE)).convert("RGBA")

    # Preprocess
    print("Preprocessing image...")
    preprocessed = pipeline.preprocess_image(raw_image)

    # Get conditioning
    print("Getting conditioning...")
    cond_512 = pipeline.get_cond([preprocessed], 512)
    cond_1024 = pipeline.get_cond([preprocessed], 1024)

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

    torch.manual_seed(seed)

    # Run sampling
    print(f"\n--- Sampling {'WITH' if use_autocast else 'WITHOUT'} autocast ---")

    t_total = time.time()

    # Stage 1: Sparse Structure
    print("\nStage 1: Sparse Structure...")
    t0 = time.time()
    if use_autocast:
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=True):
            coords = pipeline.sample_sparse_structure(cond_512, 32, 1, sparse_params)
    else:
        coords = pipeline.sample_sparse_structure(cond_512, 32, 1, sparse_params)
    print(f"  Coords: {coords.shape[0]} | Time: {time.time()-t0:.1f}s")

    # Stage 2: Shape SLat Cascade
    print("\nStage 2: Shape SLat Cascade...")
    t0 = time.time()
    if use_autocast:
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=True):
            shape_slat, res = pipeline.sample_shape_slat_cascade(
                cond_512, cond_1024,
                pipeline.models['shape_slat_flow_model_512'],
                pipeline.models['shape_slat_flow_model_1024'],
                512, 1024,
                coords, shape_params,
                49152
            )
    else:
        shape_slat, res = pipeline.sample_shape_slat_cascade(
            cond_512, cond_1024,
            pipeline.models['shape_slat_flow_model_512'],
            pipeline.models['shape_slat_flow_model_1024'],
            512, 1024,
            coords, shape_params,
            49152
        )
    print(f"  Shape slat feats: {shape_slat.feats.shape} | Resolution: {res} | Time: {time.time()-t0:.1f}s")

    # Stage 3: Texture SLat
    print("\nStage 3: Texture SLat...")
    t0 = time.time()
    if use_autocast:
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=True):
            tex_slat = pipeline.sample_tex_slat(
                cond_1024,
                pipeline.models['tex_slat_flow_model_1024'],
                shape_slat, tex_params
            )
    else:
        tex_slat = pipeline.sample_tex_slat(
            cond_1024,
            pipeline.models['tex_slat_flow_model_1024'],
            shape_slat, tex_params
        )
    print(f"  Tex slat feats: {tex_slat.feats.shape} | Time: {time.time()-t0:.1f}s")

    # Decode
    print("\nDecoding latents...")
    t0 = time.time()
    outputs = pipeline.decode_latent(shape_slat, tex_slat, res)
    print(f"  Mesh vertices: {outputs[0].vertices.shape[0]} | Time: {time.time()-t0:.1f}s")

    total_time = time.time() - t_total
    print(f"\nTotal inference time: {total_time:.1f}s")

    # Export
    output_suffix = "autocast" if use_autocast else "no_autocast"
    output_dir = TRELLIS_ROOT / "outputs" / "autocast_test"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"test_{output_suffix}_seed{seed}.glb"

    print(f"\nExporting to {output_path}...")
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
        remesh=True,
        remesh_band=1,
        remesh_project=0,
        verbose=True,
        use_tqdm=True,
    )
    glb.export(str(output_path), extension_webp=True)
    print(f"Exported: {output_path}")

    return output_path, total_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--autocast", action="store_true", help="Use autocast")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    run_test(use_autocast=args.autocast, seed=args.seed)

    print("\n" + "="*60)
    print("Test complete. Compare the output GLB visually with the benchmark:")
    print("  Benchmark: reference/sample_2026-01-24T055452.643.glb")
    print("="*60)
