"""
Full TRELLIS.2 Pipeline Trace v2

Traces every component with detailed mesh quality metrics at each stage.
"""
from pathlib import Path as _Path
ROOT = _Path(__file__).resolve().parents[2]  # repo root

import sys
import os
import time
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from datetime import datetime

# Add paths
sys.path.insert(0, str(ROOT))

# Disable gradients for inference
torch.set_grad_enabled(False)


def count_degenerate_faces(vertices, faces, threshold=1e-10):
    """Count faces with area below threshold."""
    v = vertices[faces.cpu().numpy() if torch.is_tensor(faces) else faces]
    if torch.is_tensor(v):
        v = v.cpu().numpy()
    cross = np.cross(v[:,1]-v[:,0], v[:,2]-v[:,0])
    areas = np.linalg.norm(cross, axis=1) / 2
    return int((areas < threshold).sum()), areas.min(), areas.max()


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
    """Run full pipeline with detailed mesh quality tracing."""
    print("="*70)
    print("TRELLIS.2 FULL PIPELINE TRACE v2")
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
        rf"{ROOT}\models\hub\TRELLIS.2-4B"
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
        print(f"Vertices: {out_mesh.vertices.shape}")
        print(f"Faces: {out_mesh.faces.shape}")
        degen, amin, amax = count_degenerate_faces(out_mesh.vertices, out_mesh.faces)
        print(f"Degenerate faces: {degen}")
        print(f"Face area range: [{amin:.2e}, {amax:.2e}]")
    if hasattr(out_mesh, 'coords'):
        print(f"Voxel coords: {out_mesh.coords.shape}")
    if hasattr(out_mesh, 'attrs'):
        print(f"Voxel attrs: {out_mesh.attrs.shape}")

    print(f"Mesh decoding time: {time.time() - start:.2f}s")
    log_memory()

    # ========================================================================
    # EXPORT TO GLB (with detailed tracing)
    # ========================================================================
    print("\n" + "="*70)
    print("STAGE 5: EXPORT TO GLB (cumesh + texture baking)")
    print("="*70)
    start = time.time()

    if output_dir is None:
        output_dir = Path(rf"{ROOT}\outputs\benchmarks\trace_test")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"traced_v2_seed{seed}.glb"

    # Export with official parameters:
    # - texture_size=2048 (official default)
    # - decimation_target=1000000 (official default)
    out_mesh.export(
        str(output_path),
        texture_size=2048,
        decimation_target=1000000,
        remesh=True,
        remesh_band=1.0,
        verbose=True
    )

    print(f"\nExport time: {time.time() - start:.2f}s")
    log_memory()

    # ========================================================================
    # VERIFY OUTPUT
    # ========================================================================
    print("\n" + "="*70)
    print("STAGE 6: VERIFY OUTPUT")
    print("="*70)

    import trimesh
    final_mesh = trimesh.load(str(output_path))
    if hasattr(final_mesh, 'geometry'):
        geom = list(final_mesh.geometry.values())[0]
    else:
        geom = final_mesh

    print(f"Final vertices: {len(geom.vertices)}")
    print(f"Final faces: {len(geom.faces)}")

    degen, amin, amax = count_degenerate_faces(geom.vertices, geom.faces)
    print(f"Degenerate faces: {degen}")
    print(f"Face area range: [{amin:.2e}, {amax:.2e}]")

    if hasattr(geom, 'visual') and hasattr(geom.visual, 'material'):
        mat = geom.visual.material
        if hasattr(mat, 'baseColorTexture') and mat.baseColorTexture is not None:
            print(f"Texture size: {mat.baseColorTexture.size}")

    print(f"\n{'='*70}")
    print(f"OUTPUT SAVED: {output_path}")
    print(f"{'='*70}")

    return final_mesh, output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=rf"{ROOT}\reference\spiral_input.png")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_traced_generation(args.image, args.seed)
