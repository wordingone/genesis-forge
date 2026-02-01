"""Test BVH projection accuracy - does it introduce texture artifacts?

This test checks if the BVH projection step (mapping simplified mesh positions
back to original mesh positions) causes color discontinuities at triangle boundaries.

The hypothesis is that adjacent texels within the same simplified triangle
might get projected to different original triangles, causing color discontinuities.
"""
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from pathlib import Path
TRELLIS_ROOT = Path(__file__).parent

import sys
sys.path.insert(0, str(TRELLIS_ROOT))

import torch
import numpy as np
import cumesh

# Ensure venv is used
assert "venv311" in sys.executable, f"Wrong Python: {sys.executable}"

def create_test_meshes():
    """Create a simple test case: original high-res mesh and simplified version."""
    device = 'cuda'

    # Create a simple cube as "original" mesh
    # 8 vertices, 12 triangles (2 per face)
    vertices_orig = torch.tensor([
        [-0.5, -0.5, -0.5],  # 0
        [ 0.5, -0.5, -0.5],  # 1
        [ 0.5,  0.5, -0.5],  # 2
        [-0.5,  0.5, -0.5],  # 3
        [-0.5, -0.5,  0.5],  # 4
        [ 0.5, -0.5,  0.5],  # 5
        [ 0.5,  0.5,  0.5],  # 6
        [-0.5,  0.5,  0.5],  # 7
    ], dtype=torch.float32, device=device)

    faces_orig = torch.tensor([
        # Front face
        [0, 1, 2], [0, 2, 3],
        # Back face
        [5, 4, 7], [5, 7, 6],
        # Top face
        [3, 2, 6], [3, 6, 7],
        # Bottom face
        [4, 5, 1], [4, 1, 0],
        # Right face
        [1, 5, 6], [1, 6, 2],
        # Left face
        [4, 0, 3], [4, 3, 7],
    ], dtype=torch.int32, device=device)

    # Create a "simplified" version by slightly perturbing vertices
    # This simulates what happens when mesh is simplified
    noise = torch.randn_like(vertices_orig) * 0.01
    vertices_simp = vertices_orig + noise
    faces_simp = faces_orig.clone()

    return vertices_orig, faces_orig, vertices_simp, faces_simp


def test_bvh_projection_continuity():
    """Test if BVH projection preserves continuity across triangle boundaries."""
    print("=" * 60)
    print("BVH PROJECTION CONTINUITY TEST")
    print("=" * 60)

    device = 'cuda'

    # Create test meshes
    vertices_orig, faces_orig, vertices_simp, faces_simp = create_test_meshes()

    print(f"\nOriginal mesh: {vertices_orig.shape[0]} vertices, {faces_orig.shape[0]} faces")
    print(f"Simplified mesh: {vertices_simp.shape[0]} vertices, {faces_simp.shape[0]} faces")

    # Build BVH on original mesh
    bvh = cumesh.cuBVH(vertices_orig, faces_orig)
    print(f"BVH built on original mesh")

    # For each edge in simplified mesh, sample points along it
    # and check if BVH projection gives consistent face_ids

    print("\n" + "=" * 60)
    print("EDGE CONTINUITY TEST")
    print("=" * 60)

    # Get all unique edges from simplified mesh
    edges = set()
    for face in faces_simp.cpu().numpy():
        for i in range(3):
            v0, v1 = face[i], face[(i+1) % 3]
            if v0 > v1:
                v0, v1 = v1, v0
            edges.add((v0, v1))

    print(f"Testing {len(edges)} unique edges...")

    discontinuities = 0
    total_tests = 0

    for v0_idx, v1_idx in list(edges)[:20]:  # Test first 20 edges
        v0 = vertices_simp[v0_idx]
        v1 = vertices_simp[v1_idx]

        # Sample 10 points along the edge
        t_values = torch.linspace(0, 1, 10, device=device)
        edge_points = torch.stack([v0 + t * (v1 - v0) for t in t_values])

        # Project each point using BVH
        _, face_ids, _ = bvh.unsigned_distance(edge_points, return_uvw=True)

        # Check if face_ids change along the edge
        unique_faces = face_ids.unique()
        if len(unique_faces) > 2:  # More than 2 faces along single edge is suspicious
            discontinuities += 1
            print(f"  Edge ({v0_idx}, {v1_idx}): projects to {len(unique_faces)} different faces!")
        total_tests += 1

    print(f"\nResults: {discontinuities}/{total_tests} edges had discontinuous projections")

    # Test: sample grid of points and check projection consistency
    print("\n" + "=" * 60)
    print("GRID PROJECTION TEST")
    print("=" * 60)

    # Create a grid of points on one face of the cube
    face_idx = 0
    v0 = vertices_simp[faces_simp[face_idx, 0]]
    v1 = vertices_simp[faces_simp[face_idx, 1]]
    v2 = vertices_simp[faces_simp[face_idx, 2]]

    # Sample points inside the triangle
    n_samples = 100
    grid_points = []
    for _ in range(n_samples):
        # Random barycentric coords
        r1, r2 = torch.rand(2, device=device)
        if r1 + r2 > 1:
            r1, r2 = 1 - r1, 1 - r2
        r3 = 1 - r1 - r2
        point = r1 * v0 + r2 * v1 + r3 * v2
        grid_points.append(point)

    grid_points = torch.stack(grid_points)

    # Project all points
    _, face_ids, uvw = bvh.unsigned_distance(grid_points, return_uvw=True)
    unique_faces = face_ids.unique()

    print(f"Sampled {n_samples} points inside single simplified triangle")
    print(f"These points project to {len(unique_faces)} different original faces")
    print(f"Original face IDs: {unique_faces.tolist()}")

    # This is expected if simplified triangle overlaps multiple original triangles
    # But if the distribution is uneven, it could cause texture artifacts

    face_counts = {}
    for fid in face_ids.cpu().numpy():
        face_counts[fid] = face_counts.get(fid, 0) + 1

    print(f"\nProjection distribution:")
    for fid, count in sorted(face_counts.items()):
        print(f"  Original face {fid}: {count} samples ({100*count/n_samples:.1f}%)")

    print("\n" + "=" * 60)
    print("DISTANCE ANALYSIS")
    print("=" * 60)

    # Check unsigned distance values - large distances indicate projection issues
    dist, _, _ = bvh.unsigned_distance(grid_points, return_uvw=True)

    print(f"Distance to original mesh:")
    print(f"  min: {dist.min():.6f}")
    print(f"  max: {dist.max():.6f}")
    print(f"  mean: {dist.mean():.6f}")

    # For a correctly simplified mesh, distances should be small and uniform

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


def test_with_loaded_mesh():
    """Test with actual mesh data if a GLB file is available."""
    print("\n" + "=" * 60)
    print("TESTING WITH GENERATED MESH")
    print("=" * 60)

    import trimesh

    # Load our generated mesh
    glb_path = TRELLIS_ROOT / "outputs" / "hybrid_precision_test" / "generated_spiral_input_seed42.glb"
    if not glb_path.exists():
        print(f"No generated mesh found at {glb_path}")
        return

    scene = trimesh.load(str(glb_path))
    if isinstance(scene, trimesh.Scene):
        mesh = scene.dump(concatenate=True)
    else:
        mesh = scene

    print(f"Loaded mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

    # Check triangle quality
    triangles = mesh.vertices[mesh.faces]  # (N, 3, 3)

    # Calculate edge lengths
    edge1 = np.linalg.norm(triangles[:, 1] - triangles[:, 0], axis=1)
    edge2 = np.linalg.norm(triangles[:, 2] - triangles[:, 1], axis=1)
    edge3 = np.linalg.norm(triangles[:, 0] - triangles[:, 2], axis=1)

    # Aspect ratio: longest edge / shortest edge
    edges = np.stack([edge1, edge2, edge3], axis=1)
    longest = edges.max(axis=1)
    shortest = edges.min(axis=1)

    # Avoid division by zero
    shortest = np.maximum(shortest, 1e-10)
    aspect_ratios = longest / shortest

    print(f"\nTriangle aspect ratios:")
    print(f"  min: {aspect_ratios.min():.2f}")
    print(f"  max: {aspect_ratios.max():.2f}")
    print(f"  mean: {aspect_ratios.mean():.2f}")
    print(f"  median: {np.median(aspect_ratios):.2f}")

    # Count problematic triangles
    sliver_10 = (aspect_ratios > 10).sum()
    sliver_100 = (aspect_ratios > 100).sum()
    sliver_1000 = (aspect_ratios > 1000).sum()

    print(f"\nProblematic triangles:")
    print(f"  Aspect ratio > 10: {sliver_10} ({100*sliver_10/len(aspect_ratios):.1f}%)")
    print(f"  Aspect ratio > 100: {sliver_100} ({100*sliver_100/len(aspect_ratios):.1f}%)")
    print(f"  Aspect ratio > 1000: {sliver_1000} ({100*sliver_1000/len(aspect_ratios):.1f}%)")

    # Check for degenerate triangles
    areas = mesh.area_faces
    degenerate = (areas < 1e-10).sum()
    print(f"  Degenerate (area < 1e-10): {degenerate}")


if __name__ == "__main__":
    test_bvh_projection_continuity()
    test_with_loaded_mesh()
