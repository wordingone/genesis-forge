"""Diagnose texture sampling to understand why textures are washed out."""
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
import numpy as np
from PIL import Image
from flex_gemm.ops.grid_sample import grid_sample_3d
import nvdiffrast.torch as dr
import cumesh

# Ensure venv is used
assert "venv311" in sys.executable, f"Wrong Python: {sys.executable}"

def main():
    print("=" * 60)
    print("TEXTURE SAMPLING DIAGNOSTICS")
    print("=" * 60)

    # Load the pipeline
    print("\n1. Loading pipeline...")
    from trellis2.pipelines.trellis2_image_to_3d import Trellis2ImageTo3DPipeline

    MODEL_PATH = MODELS_DIR / "hub" / "TRELLIS.2-4B"
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(str(MODEL_PATH))

    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    pipeline.configure_vram_mode(vram_gb)
    pipeline.to("cuda")
    print(f"   Pipeline loaded")

    # Load test image
    print("\n2. Loading test image...")
    image_path = TRELLIS_ROOT / "reference" / "spiral_input.png"
    image = Image.open(str(image_path)).convert("RGBA")
    print(f"   Image: {image.size}")

    # Run pipeline
    print("\n3. Running generation (seed 42)...")
    result = pipeline.run(
        image,
        seed=42,
        return_latent=True,
    )

    out_mesh_list, (shape_slat, tex_slat, res) = result
    out_mesh = out_mesh_list[0]

    print(f"\n4. Mesh data extracted:")
    print(f"   Vertices: {out_mesh.vertices.shape}")
    print(f"   Faces: {out_mesh.faces.shape}")

    vertices = out_mesh.vertices.cuda()
    faces = out_mesh.faces.cuda()
    attrs = out_mesh.attrs  # Already on GPU
    coords = out_mesh.coords  # Already on GPU
    layout = out_mesh.layout

    print(f"\n5. Voxel data:")
    print(f"   Attrs shape: {attrs.shape}")
    print(f"   Coords shape: {coords.shape}")
    print(f"   Coords range X: [{coords[:, 0].min()}, {coords[:, 0].max()}]")
    print(f"   Coords range Y: [{coords[:, 1].min()}, {coords[:, 1].max()}]")
    print(f"   Coords range Z: [{coords[:, 2].min()}, {coords[:, 2].max()}]")

    # Setup same parameters as postprocess.py
    aabb = torch.tensor([[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], device='cuda', dtype=torch.float32)
    grid_size = torch.tensor([1024, 1024, 1024], device='cuda', dtype=torch.int32)
    voxel_size = (aabb[1] - aabb[0]) / grid_size.float()

    print(f"\n6. Grid parameters:")
    print(f"   AABB: {aabb}")
    print(f"   Grid size: {grid_size}")
    print(f"   Voxel size: {voxel_size}")

    # Simplify mesh for testing (just take first 10000 vertices)
    print("\n7. Building BVH on original mesh...")
    bvh = cumesh.cuBVH(vertices, faces)
    print("   BVH built")

    # Sample a subset of texture positions to analyze
    # Take random samples from the vertex positions
    n_samples = min(10000, vertices.shape[0])
    sample_indices = torch.randperm(vertices.shape[0])[:n_samples]
    sample_pos = vertices[sample_indices]

    print(f"\n8. Sampling {n_samples} positions...")
    print(f"   Sample pos range: [{sample_pos.min(0).values.tolist()}, {sample_pos.max(0).values.tolist()}]")

    # BVH projection (as done in postprocess.py)
    _, face_id, uvw = bvh.unsigned_distance(sample_pos, return_uvw=True)
    orig_tri_verts = vertices[faces[face_id.long()]]
    projected_pos = (orig_tri_verts * uvw.unsqueeze(-1)).sum(dim=1)

    print(f"   Projected pos range: [{projected_pos.min(0).values.tolist()}, {projected_pos.max(0).values.tolist()}]")

    # Convert to grid coordinates
    grid = ((projected_pos - aabb[0]) / voxel_size).reshape(1, -1, 3)
    print(f"\n9. Grid coordinates:")
    print(f"   Grid range: [{grid.min(1).values.tolist()}, {grid.max(1).values.tolist()}]")
    print(f"   Grid mean: {grid.mean(dim=1).tolist()}")

    # Check how many are out of bounds
    oob_low = (grid < 0).sum().item()
    oob_high = (grid >= 1024).sum().item()
    print(f"   Out of bounds (< 0): {oob_low}")
    print(f"   Out of bounds (>= 1024): {oob_high}")

    # Sample attributes
    print(f"\n10. Sampling attributes with grid_sample_3d...")
    sampled_attrs = grid_sample_3d(
        attrs,
        torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=-1),
        shape=torch.Size([1, attrs.shape[1], *grid_size.tolist()]),
        grid=grid,
        mode='trilinear',
    )

    print(f"   Sampled attrs shape: {sampled_attrs.shape}")
    print(f"   Sampled attrs range: [{sampled_attrs.min():.4f}, {sampled_attrs.max():.4f}]")
    print(f"   Sampled attrs mean: {sampled_attrs.mean():.4f}")

    # Check individual channels
    base_color_slice = slice(0, 3)
    sampled_rgb = sampled_attrs[:, base_color_slice]
    print(f"\n11. Sampled RGB analysis:")
    print(f"   R: min={sampled_rgb[:, 0].min():.4f}, max={sampled_rgb[:, 0].max():.4f}, mean={sampled_rgb[:, 0].mean():.4f}")
    print(f"   G: min={sampled_rgb[:, 1].min():.4f}, max={sampled_rgb[:, 1].max():.4f}, mean={sampled_rgb[:, 1].mean():.4f}")
    print(f"   B: min={sampled_rgb[:, 2].min():.4f}, max={sampled_rgb[:, 2].max():.4f}, mean={sampled_rgb[:, 2].mean():.4f}")

    # Check for zeros (failed samples)
    zero_samples = (sampled_attrs.abs().sum(dim=1) < 0.01).sum().item()
    print(f"\n12. Zero/near-zero samples: {zero_samples} / {sampled_attrs.shape[0]} ({100*zero_samples/sampled_attrs.shape[0]:.1f}%)")

    # Compare original voxel RGB with sampled RGB
    orig_rgb = attrs[:, base_color_slice]
    print(f"\n13. Comparison: Original voxels vs Sampled:")
    print(f"   Original RGB mean: R={orig_rgb[:, 0].mean():.4f}, G={orig_rgb[:, 1].mean():.4f}, B={orig_rgb[:, 2].mean():.4f}")
    print(f"   Sampled RGB mean:  R={sampled_rgb[:, 0].mean():.4f}, G={sampled_rgb[:, 1].mean():.4f}, B={sampled_rgb[:, 2].mean():.4f}")
    print(f"   Brightness change: {sampled_rgb.mean() / orig_rgb.mean():.3f}x")

    # Distribution analysis
    print(f"\n14. Distribution analysis:")
    for name, (orig, samp) in [('R', (orig_rgb[:, 0], sampled_rgb[:, 0])),
                               ('G', (orig_rgb[:, 1], sampled_rgb[:, 1])),
                               ('B', (orig_rgb[:, 2], sampled_rgb[:, 2]))]:
        orig_hist, _ = np.histogram(orig.cpu().numpy(), bins=10, range=(0, 1))
        samp_hist, _ = np.histogram(samp.cpu().numpy(), bins=10, range=(0, 1))
        print(f"   {name} original: {list(orig_hist)}")
        print(f"   {name} sampled:  {list(samp_hist)}")

    print("\n" + "=" * 60)
    print("DIAGNOSTICS COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
