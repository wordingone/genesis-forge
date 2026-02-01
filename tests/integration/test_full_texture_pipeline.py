"""Test the full texture baking pipeline to find where colors go wrong."""
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
    print("FULL TEXTURE BAKING PIPELINE TEST")
    print("=" * 60)

    # Load the pipeline and run to get real data
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

    orig_vertices = out_mesh.vertices.cuda()
    orig_faces = out_mesh.faces.cuda()
    attrs = out_mesh.attrs.cuda()
    coords = out_mesh.coords.cuda()
    layout = out_mesh.layout

    print(f"\n4. Original mesh:")
    print(f"   Vertices: {orig_vertices.shape}")
    print(f"   Faces: {orig_faces.shape}")

    print(f"\n5. Voxel data:")
    print(f"   Attrs: {attrs.shape}")
    print(f"   Coords: {coords.shape}")

    # Step 1: Fill holes (like postprocess.py)
    print("\n" + "=" * 60)
    print("STEP 1: Fill holes and build BVH on original mesh")
    print("=" * 60)

    mesh = cumesh.CuMesh()
    mesh.init(orig_vertices, orig_faces)
    mesh.fill_holes(max_hole_perimeter=3e-2)
    orig_vertices_clean, orig_faces_clean = mesh.read()
    print(f"   After fill_holes: {orig_vertices_clean.shape[0]} verts, {orig_faces_clean.shape[0]} faces")

    # Build BVH on the CLEANED original mesh
    bvh = cumesh.cuBVH(orig_vertices_clean, orig_faces_clean)
    print(f"   BVH built on cleaned original mesh")

    # Step 2: Simplify (like postprocess.py)
    print("\n" + "=" * 60)
    print("STEP 2: Simplify mesh (3x target, then target)")
    print("=" * 60)

    target = 1000000
    mesh.simplify(target * 3, verbose=False)
    mesh.remove_duplicate_faces()
    mesh.repair_non_manifold_edges()
    mesh.remove_small_connected_components(1e-5)
    mesh.fill_holes(max_hole_perimeter=3e-2)
    print(f"   After 3x simplify: {mesh.num_vertices} verts, {mesh.num_faces} faces")

    mesh.simplify(target, verbose=False)
    mesh.remove_duplicate_faces()
    mesh.repair_non_manifold_edges()
    mesh.remove_small_connected_components(1e-5)
    mesh.fill_holes(max_hole_perimeter=3e-2)
    mesh.unify_face_orientations()
    simp_vertices, simp_faces = mesh.read()
    print(f"   After target simplify: {simp_vertices.shape[0]} verts, {simp_faces.shape[0]} faces")

    # Step 3: UV unwrap (like postprocess.py)
    print("\n" + "=" * 60)
    print("STEP 3: UV unwrap")
    print("=" * 60)

    out_vertices, out_faces, out_uvs, out_vmaps = mesh.uv_unwrap(
        compute_charts_kwargs={
            "threshold_cone_half_angle_rad": np.radians(90.0),
            "refine_iterations": 0,
            "global_iterations": 1,
            "smooth_strength": 1,
        },
        return_vmaps=True,
        verbose=False,
    )
    out_vertices = out_vertices.cuda()
    out_faces = out_faces.cuda()
    out_uvs = out_uvs.cuda()
    print(f"   After UV unwrap: {out_vertices.shape[0]} verts, {out_faces.shape[0]} faces, {out_uvs.shape[0]} uvs")
    print(f"   UV range: [{out_uvs.min():.4f}, {out_uvs.max():.4f}]")
    print(f"   UV U range: [{out_uvs[:, 0].min():.4f}, {out_uvs[:, 0].max():.4f}]")
    print(f"   UV V range: [{out_uvs[:, 1].min():.4f}, {out_uvs[:, 1].max():.4f}]")

    # Step 4: Rasterize in UV space
    print("\n" + "=" * 60)
    print("STEP 4: Rasterize in UV space")
    print("=" * 60)

    texture_size = 512  # Smaller for faster test
    ctx = dr.RasterizeCudaContext()
    uvs_rast = torch.cat([out_uvs * 2 - 1, torch.zeros_like(out_uvs[:, :1]), torch.ones_like(out_uvs[:, :1])], dim=-1).unsqueeze(0)
    rast, _ = dr.rasterize(ctx, uvs_rast, out_faces, resolution=[texture_size, texture_size])
    mask = rast[0, ..., 3] > 0
    print(f"   Raster mask coverage: {mask.sum().item()} / {mask.numel()} ({100*mask.sum()/mask.numel():.1f}%)")

    # Step 5: Interpolate 3D positions
    print("\n" + "=" * 60)
    print("STEP 5: Interpolate 3D positions")
    print("=" * 60)

    pos = dr.interpolate(out_vertices.unsqueeze(0), rast, out_faces)[0][0]
    valid_pos = pos[mask]
    print(f"   Valid positions: {valid_pos.shape}")
    print(f"   Position range: [{valid_pos.min(0).values.tolist()}, {valid_pos.max(0).values.tolist()}]")

    # Step 6: BVH projection (THE CRITICAL STEP)
    print("\n" + "=" * 60)
    print("STEP 6: BVH projection (project simplified mesh to original)")
    print("=" * 60)

    _, face_id, uvw = bvh.unsigned_distance(valid_pos, return_uvw=True)
    orig_tri_verts = orig_vertices_clean[orig_faces_clean[face_id.long()]]
    projected_pos = (orig_tri_verts * uvw.unsqueeze(-1)).sum(dim=1)

    print(f"   Face IDs: min={face_id.min()}, max={face_id.max()}, unique={face_id.unique().shape[0]}")
    print(f"   Projected positions: {projected_pos.shape}")
    print(f"   Projected range: [{projected_pos.min(0).values.tolist()}, {projected_pos.max(0).values.tolist()}]")

    # Compare projected vs original positions
    pos_diff = (projected_pos - valid_pos).abs()
    print(f"   Position difference: max={pos_diff.max():.6f}, mean={pos_diff.mean():.6f}")

    # Step 7: Grid sample (THE SECOND CRITICAL STEP)
    print("\n" + "=" * 60)
    print("STEP 7: Grid sampling")
    print("=" * 60)

    aabb = torch.tensor([[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], device='cuda', dtype=torch.float32)
    grid_size = torch.tensor([1024, 1024, 1024], device='cuda', dtype=torch.int32)
    voxel_size = (aabb[1] - aabb[0]) / grid_size.float()

    # Method A: Sample at projected positions (like postprocess.py)
    grid_projected = ((projected_pos - aabb[0]) / voxel_size).reshape(1, -1, 3)
    print(f"   Grid (projected) range: [{grid_projected.min():.1f}, {grid_projected.max():.1f}]")

    # Method B: Sample at original interpolated positions (without BVH)
    grid_direct = ((valid_pos - aabb[0]) / voxel_size).reshape(1, -1, 3)
    print(f"   Grid (direct) range: [{grid_direct.min():.1f}, {grid_direct.max():.1f}]")

    # Sample using both methods
    shape = torch.Size([1, attrs.shape[1], *grid_size.tolist()])
    coords_4d = torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=-1)

    sampled_projected = grid_sample_3d(attrs, coords_4d, shape, grid_projected, mode='trilinear')[0]
    sampled_direct = grid_sample_3d(attrs, coords_4d, shape, grid_direct, mode='trilinear')[0]

    base_color = layout['base_color']
    rgb_projected = sampled_projected[:, base_color]
    rgb_direct = sampled_direct[:, base_color]

    print(f"\n   RGB stats (PROJECTED - via BVH):")
    print(f"   R: mean={rgb_projected[:, 0].mean():.4f}")
    print(f"   G: mean={rgb_projected[:, 1].mean():.4f}")
    print(f"   B: mean={rgb_projected[:, 2].mean():.4f}")

    print(f"\n   RGB stats (DIRECT - no BVH):")
    print(f"   R: mean={rgb_direct[:, 0].mean():.4f}")
    print(f"   G: mean={rgb_direct[:, 1].mean():.4f}")
    print(f"   B: mean={rgb_direct[:, 2].mean():.4f}")

    # Original voxel stats for comparison
    voxel_rgb = attrs[:, base_color]
    print(f"\n   RGB stats (ORIGINAL VOXELS):")
    print(f"   R: mean={voxel_rgb[:, 0].mean():.4f}")
    print(f"   G: mean={voxel_rgb[:, 1].mean():.4f}")
    print(f"   B: mean={voxel_rgb[:, 2].mean():.4f}")

    # Check for zero samples
    zero_proj = (sampled_projected.abs().sum(dim=1) < 0.01).sum().item()
    zero_direct = (sampled_direct.abs().sum(dim=1) < 0.01).sum().item()
    print(f"\n   Zero samples (projected): {zero_proj} / {sampled_projected.shape[0]}")
    print(f"   Zero samples (direct): {zero_direct} / {sampled_direct.shape[0]}")

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
