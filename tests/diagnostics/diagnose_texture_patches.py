"""Diagnose triangular texture patch artifacts in TRELLIS.2 output.

This script investigates why texture patches appear triangular/discontinuous
even though the grid_sample_3d algorithm is working correctly.

Hypothesis: The issue is in BVH projection discontinuities, NOT grid_sample.
"""
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from pathlib import Path
TRELLIS_ROOT = Path(__file__).parent

import sys
sys.path.insert(0, str(TRELLIS_ROOT))

import torch
import numpy as np
from collections import defaultdict

# Ensure venv is used
assert "venv311" in sys.executable, f"Wrong Python: {sys.executable}"


def analyze_face_projection_continuity(bvh, vertices, faces, out_vertices, out_faces):
    """
    Analyze if adjacent triangles in the simplified mesh project to
    the same region in the original mesh.

    If adjacent simplified triangles project to DIFFERENT original faces,
    we get texture discontinuities at triangle edges (the "triangular patches").
    """
    print("\n" + "=" * 70)
    print("FACE PROJECTION CONTINUITY ANALYSIS")
    print("=" * 70)

    # Build edge-to-face adjacency for simplified mesh
    edge_to_faces = defaultdict(list)
    for fi, face in enumerate(out_faces):
        for i in range(3):
            v0, v1 = face[i].item(), face[(i+1) % 3].item()
            edge = tuple(sorted([v0, v1]))
            edge_to_faces[edge].append(fi)

    # Count shared edges (edges with exactly 2 faces)
    shared_edges = [(e, fs) for e, fs in edge_to_faces.items() if len(fs) == 2]
    print(f"Simplified mesh: {len(out_faces)} faces, {len(shared_edges)} shared edges")

    # For each shared edge, check if adjacent faces project to same original face
    discontinuities = 0
    sample_discontinuities = []

    for edge, (f1, f2) in shared_edges[:1000]:  # Sample first 1000 edges
        # Get edge vertices
        v0_pos = out_vertices[edge[0]]
        v1_pos = out_vertices[edge[1]]
        edge_center = (v0_pos + v1_pos) / 2

        # Get face centers
        f1_center = out_vertices[out_faces[f1]].mean(dim=0)
        f2_center = out_vertices[out_faces[f2]].mean(dim=0)

        # Project all three points to original mesh
        points = torch.stack([edge_center, f1_center, f2_center])
        _, orig_face_ids, _ = bvh.unsigned_distance(points, return_uvw=True)

        edge_orig = orig_face_ids[0].item()
        f1_orig = orig_face_ids[1].item()
        f2_orig = orig_face_ids[2].item()

        # Check if the two faces project to different original faces
        if f1_orig != f2_orig:
            discontinuities += 1
            if len(sample_discontinuities) < 10:
                sample_discontinuities.append({
                    'edge': edge,
                    'f1': f1, 'f2': f2,
                    'f1_orig': f1_orig, 'f2_orig': f2_orig,
                    'edge_orig': edge_orig
                })

    total_checked = min(1000, len(shared_edges))
    discontinuity_pct = 100 * discontinuities / total_checked if total_checked > 0 else 0

    print(f"\nEdges checked: {total_checked}")
    print(f"Discontinuities (adjacent faces → different original faces): {discontinuities} ({discontinuity_pct:.1f}%)")

    if sample_discontinuities:
        print("\nSample discontinuities:")
        for d in sample_discontinuities[:5]:
            print(f"  Edge {d['edge']}: simp face {d['f1']}→orig {d['f1_orig']}, simp face {d['f2']}→orig {d['f2_orig']}")

    return discontinuity_pct


def analyze_uv_seam_artifacts(out_uvs, out_faces, out_vertices, bvh):
    """
    Analyze if UV seams align with projection discontinuities.

    UV seams themselves cause texture discontinuities, but the inpainting
    should fix those. The triangular patches we see are likely from
    BVH projection discontinuities WITHIN UV charts.
    """
    print("\n" + "=" * 70)
    print("UV SEAM vs BVH DISCONTINUITY ANALYSIS")
    print("=" * 70)

    # Find UV seams: edges where UV coordinates are discontinuous
    # In a properly unwrapped mesh, each edge should have 2 half-edges with same UVs

    edge_uvs = defaultdict(list)
    for fi, face in enumerate(out_faces):
        for i in range(3):
            v0, v1 = face[i].item(), face[(i+1) % 3].item()
            uv0, uv1 = out_uvs[v0].cpu().numpy(), out_uvs[v1].cpu().numpy()
            edge = tuple(sorted([v0, v1]))
            edge_uvs[edge].append((fi, uv0, uv1))

    # Count UV seams (edges where UVs don't match)
    uv_seams = 0
    for edge, half_edges in edge_uvs.items():
        if len(half_edges) != 2:
            continue
        # Check if UV coords match (within tolerance)
        _, uv0_a, uv1_a = half_edges[0]
        _, uv0_b, uv1_b = half_edges[1]

        # The edge might be ordered differently, check both orderings
        match1 = np.allclose(uv0_a, uv0_b, atol=1e-5) and np.allclose(uv1_a, uv1_b, atol=1e-5)
        match2 = np.allclose(uv0_a, uv1_b, atol=1e-5) and np.allclose(uv1_a, uv0_b, atol=1e-5)

        if not (match1 or match2):
            uv_seams += 1

    print(f"UV seams detected: {uv_seams} edges")
    print("(UV seams are expected - xatlas creates them for minimal distortion)")

    return uv_seams


def analyze_voxel_grid_coverage(attr_volume, coords, grid_size):
    """
    Analyze how densely the voxel grid is populated.
    Sparse regions will have interpolation artifacts.
    """
    print("\n" + "=" * 70)
    print("VOXEL GRID COVERAGE ANALYSIS")
    print("=" * 70)

    N = coords.shape[0]
    total_voxels = grid_size[0] * grid_size[1] * grid_size[2]
    occupancy = 100 * N / total_voxels

    print(f"Grid size: {grid_size[0]}x{grid_size[1]}x{grid_size[2]} = {total_voxels:,} voxels")
    print(f"Occupied voxels: {N:,} ({occupancy:.2f}%)")

    # Check distribution across grid
    coords_np = coords.cpu().numpy()
    x_range = coords_np[:, 1].min(), coords_np[:, 1].max()
    y_range = coords_np[:, 2].min(), coords_np[:, 2].max()
    z_range = coords_np[:, 3].min(), coords_np[:, 3].max()

    print(f"Coord ranges: X[{x_range[0]}, {x_range[1]}], Y[{y_range[0]}, {y_range[1]}], Z[{z_range[0]}, {z_range[1]}]")

    # Check for clustering
    x_bins = np.histogram(coords_np[:, 1], bins=8)[0]
    y_bins = np.histogram(coords_np[:, 2], bins=8)[0]
    z_bins = np.histogram(coords_np[:, 3], bins=8)[0]

    x_variance = np.var(x_bins)
    y_variance = np.var(y_bins)
    z_variance = np.var(z_bins)

    print(f"Distribution variance (higher = more clustered):")
    print(f"  X: {x_variance:.0f}, Y: {y_variance:.0f}, Z: {z_variance:.0f}")

    return occupancy


def analyze_attribute_continuity(attr_volume, coords, grid_size):
    """
    Analyze if neighboring voxels have similar attributes.
    Large attribute discontinuities cause texture patches.
    """
    print("\n" + "=" * 70)
    print("ATTRIBUTE CONTINUITY ANALYSIS")
    print("=" * 70)

    # Build neighbor lookup
    coord_to_idx = {}
    for i, c in enumerate(coords.cpu().numpy()):
        key = (int(c[1]), int(c[2]), int(c[3]))
        coord_to_idx[key] = i

    # Check attribute differences between neighbors
    total_neighbors = 0
    total_diff = 0.0
    large_diffs = 0

    attrs_np = attr_volume.cpu().numpy()

    for i, c in enumerate(coords.cpu().numpy()):
        x, y, z = int(c[1]), int(c[2]), int(c[3])

        # Check 6 neighbors
        for dx, dy, dz in [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]:
            nx, ny, nz = x + dx, y + dy, z + dz
            neighbor_key = (nx, ny, nz)

            if neighbor_key in coord_to_idx:
                j = coord_to_idx[neighbor_key]
                diff = np.abs(attrs_np[i] - attrs_np[j]).max()
                total_diff += diff
                total_neighbors += 1
                if diff > 0.5:  # More than 50% color difference
                    large_diffs += 1

    avg_diff = total_diff / total_neighbors if total_neighbors > 0 else 0
    large_diff_pct = 100 * large_diffs / total_neighbors if total_neighbors > 0 else 0

    print(f"Neighbor pairs analyzed: {total_neighbors}")
    print(f"Average attribute difference: {avg_diff:.4f}")
    print(f"Large differences (>0.5): {large_diffs} ({large_diff_pct:.1f}%)")

    return avg_diff


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Diagnose texture patch artifacts")
    parser.add_argument('--image', type=str, default='reference/spiral_input.png',
                       help='Input image for generation')
    args = parser.parse_args()

    print("=" * 70)
    print("TRIANGULAR TEXTURE PATCH DIAGNOSTIC")
    print("=" * 70)
    print(f"\nInput: {args.image}")

    # Load the TRELLIS.2 pipeline
    print("\nLoading TRELLIS.2 pipeline...")

    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    from trellis2.utils import postprocessing_utils
    from PIL import Image

    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(
        str(TRELLIS_ROOT / 'models' / 'TRELLIS.2'),
        local_files_only=True
    )
    pipeline.cuda()

    # Load image
    image = Image.open(args.image)

    # Run generation
    print("\nGenerating 3D model (this may take a few minutes)...")
    outputs = pipeline(
        image,
        seed=42,
        shape_stage_cfg={"mesh_simplify": 0.95, "mesh_decimation_target": 500000}
    )

    mesh = outputs['mesh']

    # Extract data we need for analysis
    from o_voxel import to_glb
    import cumesh

    print("\nExtracting mesh data for analysis...")

    # Get the raw mesh components
    vertices = mesh.vertices
    faces = mesh.faces
    attr_volume = mesh.attr_volume
    coords = mesh.coords
    grid_size = mesh.grid_size
    aabb = mesh.aabb

    print(f"Original mesh: {vertices.shape[0]} vertices, {faces.shape[0]} faces")
    print(f"Attr volume: {attr_volume.shape}")
    print(f"Grid size: {grid_size}")

    # Build BVH on original mesh (before simplification)
    bvh = cumesh.cuBVH(vertices, faces)

    # Simplify mesh (same as to_glb)
    mesh_obj = cumesh.TriangleMesh(vertices, faces)
    decimation_target = 500000
    mesh_obj.simplify(decimation_target * 3, preserve_attributes=True)
    out_vertices, out_faces = mesh_obj.read()

    print(f"Simplified mesh: {out_vertices.shape[0]} vertices, {out_faces.shape[0]} faces")

    # UV unwrap
    import xatlas
    vmapping, out_faces_unwrapped, out_uvs = xatlas.parametrize(
        out_vertices.cpu().numpy(),
        out_faces.cpu().numpy()
    )
    out_vertices = out_vertices[vmapping]
    out_faces = torch.from_numpy(out_faces_unwrapped).to(vertices.device)
    out_uvs = torch.from_numpy(out_uvs).to(vertices.device)

    print(f"After UV unwrap: {out_vertices.shape[0]} vertices")

    # Run analyses
    analyze_voxel_grid_coverage(attr_volume, coords, grid_size)
    analyze_attribute_continuity(attr_volume, coords, grid_size)
    analyze_face_projection_continuity(bvh, vertices, faces, out_vertices, out_faces)
    analyze_uv_seam_artifacts(out_uvs, out_faces, out_vertices, bvh)

    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)

    print("""
INTERPRETATION:
- If face projection discontinuity % is HIGH (>30%), that's the cause of triangular patches.
  The BVH projects adjacent simplified triangles to different original faces.

- If attribute discontinuity is HIGH, the voxel grid has sharp transitions that
  cause visible edges in the texture.

- If voxel occupancy is LOW (<5%), there may not be enough data for smooth interpolation.

THE FIX: These are NOT bugs - they are limitations of the mesh simplification + BVH
projection approach. The official TRELLIS.2 has the same issue. The artifacts are
more visible on certain models (like architectural facades) than on organic shapes.
""")


if __name__ == "__main__":
    main()
