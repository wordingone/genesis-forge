"""
Full TRELLIS.2 Pipeline Trace

Traces every component from image input to GLB output with detailed logging.
"""

import sys
import os
import time
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from datetime import datetime

# Add paths
sys.path.insert(0, r"B:\M\ArtificialArchitecture\spatial\trellis-forge")

# Disable gradients for inference
torch.set_grad_enabled(False)


def log_tensor(name, tensor, indent=0):
    """Log tensor statistics."""
    prefix = "  " * indent
    if tensor is None:
        print(f"{prefix}{name}: None")
        return

    if isinstance(tensor, (list, tuple)):
        print(f"{prefix}{name}: {type(tensor).__name__} len={len(tensor)}")
        for i, t in enumerate(tensor[:3]):
            log_tensor(f"[{i}]", t, indent + 1)
        if len(tensor) > 3:
            print(f"{prefix}  ... and {len(tensor) - 3} more")
        return

    if isinstance(tensor, dict):
        print(f"{prefix}{name}: dict keys={list(tensor.keys())[:5]}")
        for k, v in list(tensor.items())[:3]:
            log_tensor(f"['{k}']", v, indent + 1)
        return

    if not isinstance(tensor, torch.Tensor):
        if hasattr(tensor, 'feats'):
            # SparseTensor
            print(f"{prefix}{name}: SparseTensor")
            log_tensor("feats", tensor.feats, indent + 1)
            log_tensor("coords", tensor.coords, indent + 1)
            return
        print(f"{prefix}{name}: {type(tensor).__name__}")
        return

    print(f"{prefix}{name}: shape={list(tensor.shape)}, dtype={tensor.dtype}, device={tensor.device}")
    if tensor.numel() > 0 and tensor.numel() < 1e9:
        if tensor.dtype in [torch.float32, torch.float16, torch.bfloat16]:
            print(f"{prefix}  range=[{tensor.min().item():.4f}, {tensor.max().item():.4f}], mean={tensor.mean().item():.4f}")
        elif tensor.dtype in [torch.int32, torch.int64]:
            print(f"{prefix}  range=[{tensor.min().item()}, {tensor.max().item()}]")


def log_memory():
    """Log GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"  [GPU Memory] Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")


def run_traced_generation(image_path, seed=42, output_dir=None):
    """Run full pipeline with tracing."""
    print("="*70)
    print("TRELLIS.2 FULL PIPELINE TRACE")
    print("="*70)
    print(f"Image: {image_path}")
    print(f"Seed: {seed}")
    print(f"Time: {datetime.now().isoformat()}")
    log_memory()

    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ========================================================================
    # LOAD PIPELINE
    # ========================================================================
    print("\n" + "="*70)
    print("LOADING PIPELINE")
    print("="*70)
    start = time.time()

    from trellis2.pipelines import Trellis2ImageTo3DPipeline

    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(
        r"B:\M\ArtificialArchitecture\spatial\trellis-forge\models\hub\TRELLIS.2-4B"
    )
    pipeline.cuda()

    print(f"Pipeline loaded in {time.time() - start:.2f}s")
    print(f"Models: {list(pipeline.models.keys())}")
    print(f"Default pipeline type: {pipeline.default_pipeline_type}")
    print(f"Low VRAM mode: {pipeline.low_vram}")
    log_memory()

    # ========================================================================
    # PREPROCESS IMAGE
    # ========================================================================
    print("\n" + "="*70)
    print("STAGE 0: IMAGE PREPROCESSING")
    print("="*70)
    start = time.time()

    image = Image.open(image_path)
    print(f"Input image: size={image.size}, mode={image.mode}")

    processed_image = pipeline.preprocess_image(image)
    print(f"Processed image: size={processed_image.size}, mode={processed_image.mode}")
    print(f"Preprocessing time: {time.time() - start:.2f}s")
    log_memory()

    # ========================================================================
    # GET CONDITIONING
    # ========================================================================
    print("\n" + "="*70)
    print("STAGE 0.5: IMAGE CONDITIONING")
    print("="*70)
    start = time.time()

    torch.manual_seed(seed)
    cond_512 = pipeline.get_cond([processed_image], 512)
    cond_1024 = pipeline.get_cond([processed_image], 1024)

    print("Conditioning @ 512:")
    log_tensor("cond_512", cond_512, indent=1)
    print("Conditioning @ 1024:")
    log_tensor("cond_1024", cond_1024, indent=1)
    print(f"Conditioning time: {time.time() - start:.2f}s")
    log_memory()

    # ========================================================================
    # SPARSE STRUCTURE SAMPLING (Stage 1)
    # ========================================================================
    print("\n" + "="*70)
    print("STAGE 1: SPARSE STRUCTURE SAMPLING")
    print("="*70)
    start = time.time()

    ss_res = 32  # For 1024_cascade
    print(f"Sparse structure resolution: {ss_res}")

    coords = pipeline.sample_sparse_structure(
        cond_512, ss_res,
        num_samples=1,
        sampler_params={}
    )

    log_tensor("Output coords", coords)
    print(f"Num active voxels: {coords.shape[0]}")
    print(f"Sparse structure time: {time.time() - start:.2f}s")
    log_memory()

    # ========================================================================
    # SHAPE SLAT SAMPLING (Stage 2)
    # ========================================================================
    print("\n" + "="*70)
    print("STAGE 2: SHAPE SLAT SAMPLING (Cascade 512->1024)")
    print("="*70)
    start = time.time()

    shape_slat, res = pipeline.sample_shape_slat_cascade(
        cond_512, cond_1024,
        pipeline.models['shape_slat_flow_model_512'],
        pipeline.models['shape_slat_flow_model_1024'],
        512, 1024,
        coords,
        sampler_params={},
        max_num_tokens=49152
    )

    log_tensor("Shape SLat", shape_slat)
    print(f"Output resolution: {res}")
    print(f"Shape SLat time: {time.time() - start:.2f}s")
    log_memory()

    # ========================================================================
    # TEXTURE SLAT SAMPLING (Stage 3)
    # ========================================================================
    print("\n" + "="*70)
    print("STAGE 3: TEXTURE SLAT SAMPLING")
    print("="*70)
    start = time.time()

    tex_slat = pipeline.sample_tex_slat(
        cond_1024,
        pipeline.models['tex_slat_flow_model_1024'],
        shape_slat,
        sampler_params={}
    )

    log_tensor("Texture SLat", tex_slat)
    print(f"Texture SLat time: {time.time() - start:.2f}s")
    log_memory()

    # ========================================================================
    # MESH DECODING (O-Voxel)
    # ========================================================================
    print("\n" + "="*70)
    print("STAGE 4: MESH DECODING (O-Voxel)")
    print("="*70)
    start = time.time()

    torch.cuda.empty_cache()
    out_mesh = pipeline.decode_latent(shape_slat, tex_slat, res)

    print(f"Output type: {type(out_mesh)}")
    if isinstance(out_mesh, list):
        out_mesh = out_mesh[0]

    # Log mesh properties before postprocess
    print("\n--- Raw mesh from decoder ---")
    if hasattr(out_mesh, 'vertices'):
        log_tensor("vertices", out_mesh.vertices)
    if hasattr(out_mesh, 'faces'):
        log_tensor("faces", out_mesh.faces)
    if hasattr(out_mesh, 'feats'):
        log_tensor("feats", out_mesh.feats)
    if hasattr(out_mesh, 'coords'):
        log_tensor("coords", out_mesh.coords)

    print(f"Mesh decoding time: {time.time() - start:.2f}s")
    log_memory()

    # ========================================================================
    # POSTPROCESS TO GLB
    # ========================================================================
    print("\n" + "="*70)
    print("STAGE 5: POSTPROCESS (cumesh + texture baking)")
    print("="*70)
    start = time.time()

    # The decode_latent already returns GLB if mesh type
    # Let's check what we got
    if hasattr(out_mesh, 'export'):
        # It's a trimesh object, already postprocessed
        glb = out_mesh
        print("Mesh is already postprocessed (trimesh object)")
    else:
        # Need to run postprocess manually
        from o_voxel.o_voxel import postprocess

        glb = postprocess.to_glb(
            vertices=out_mesh.vertices,
            faces=out_mesh.faces,
            attr_volume=out_mesh.feats,
            coords=out_mesh.coords,
            attr_layout=pipeline.pbr_attr_layout,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            grid_size=res,
            decimation_target=1000000,
            texture_size=2048,
            verbose=True,
            use_tqdm=True,
        )

    print(f"\n--- Final GLB ---")
    print(f"Vertices: {len(glb.vertices)}")
    print(f"Faces: {len(glb.faces)}")
    print(f"Postprocess time: {time.time() - start:.2f}s")
    log_memory()

    # ========================================================================
    # SAVE OUTPUT
    # ========================================================================
    if output_dir is None:
        output_dir = Path(r"B:\M\ArtificialArchitecture\spatial\trellis-forge\outputs\benchmarks\trace_test")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"traced_seed{seed}.glb"
    glb.export(str(output_path))

    print(f"\n{'='*70}")
    print(f"OUTPUT SAVED: {output_path}")
    print(f"{'='*70}")

    return glb, output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=r"B:\Downloads\spiral_input.png")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_traced_generation(args.image, args.seed)
