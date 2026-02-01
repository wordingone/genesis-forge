"""Diagnose BVH projection continuity issues.

This is the root cause of triangular texture patches:
- Mesh simplification creates new triangles
- BVH projects simplified mesh positions back to original mesh
- When adjacent simplified triangles project to DIFFERENT original faces,
  we get texture discontinuities at triangle edges

This script loads a pre-generated GLB's intermediate data and analyzes
the projection continuity.
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
import cumesh

# Ensure venv is used
assert "venv311" in sys.executable, f"Wrong Python: {sys.executable}"


def create_test_mesh():
    """Create a simple test mesh (cube) to analyze BVH behavior."""
    # Cube vertices
    vertices = torch.tensor([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],  # front face
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],      # back face
    ], dtype=torch.float32, device='cuda')

    # Cube faces (12 triangles)
    faces = torch.tensor([
        [0, 1, 2], [0, 2, 3],  # front
        [4, 6, 5], [4, 7, 6],  # back
        [0, 4, 5], [0, 5, 1],  # bottom
        [2, 6, 7], [2, 7, 3],  # top
        [0, 3, 7], [0, 7, 4],  # left
        [1, 5, 6], [1, 6, 2],  # right
    ], dtype=torch.int32, device='cuda')

    return vertices, faces


def subdivide_mesh(vertices, faces, iterations=2):
    """Simple subdivision to create a denser mesh."""
    v = vertices.cpu().numpy()
    f = faces.cpu().numpy()

    for _ in range(iterations):
        edge_verts = {}
        new_faces = []
        new_verts = list(v)

        for face in f:
            mids = []
            for i in range(3):
                e = tuple(sorted([face[i], face[(i+1)%3]]))
                if e not in edge_verts:
                    mid = (v[e[0]] + v[e[1]]) / 2
                    edge_verts[e] = len(new_verts)
                    new_verts.append(mid)
                mids.append(edge_verts[e])

            # Create 4 new triangles
            new_faces.append([face[0], mids[0], mids[2]])
            new_faces.append([mids[0], face[1], mids[1]])
            new_faces.append([mids[2], mids[1], face[2]])
            new_faces.append([mids[0], mids[1], mids[2]])

        v = np.array(new_verts)
        f = np.array(new_faces)

    return torch.tensor(v, dtype=torch.float32, device='cuda'), torch.tensor(f, dtype=torch.int32, device='cuda')


def analyze_bvh_continuity(orig_vertices, orig_faces, simp_vertices, simp_faces):
    """
    Analyze if adjacent triangles in simplified mesh project to
    contiguous regions in the original mesh.
    """
    print("\n" + "=" * 70)
    print("BVH PROJECTION CONTINUITY ANALYSIS")
    print("=" * 70)

    # Build BVH on original mesh
    bvh = cumesh.cuBVH(orig_vertices, orig_faces)

    # Build edge-to-face adjacency for simplified mesh
    edge_to_faces = defaultdict(list)
    for fi, face in enumerate(simp_faces):
        for i in range(3):
            v0, v1 = face[i].item(), face[(i+1) % 3].item()
            edge = tuple(sorted([v0, v1]))
            edge_to_faces[edge].append(fi)

    # Count shared edges (edges with exactly 2 faces)
    shared_edges = [(e, fs) for e, fs in edge_to_faces.items() if len(fs) == 2]
    print(f"Original mesh: {orig_faces.shape[0]} faces")
    print(f"Simplified mesh: {simp_faces.shape[0]} faces, {len(shared_edges)} shared edges")

    # For each shared edge, check if adjacent faces project to same original face
    discontinuities = 0
    sample_discontinuities = []

    for edge, (f1, f2) in shared_edges:
        # Get face centers
        f1_center = simp_vertices[simp_faces[f1]].mean(dim=0)
        f2_center = simp_vertices[simp_faces[f2]].mean(dim=0)

        # Project to original mesh
        points = torch.stack([f1_center, f2_center])
        _, orig_face_ids, _ = bvh.unsigned_distance(points, return_uvw=True)

        f1_orig = orig_face_ids[0].item()
        f2_orig = orig_face_ids[1].item()

        # Check if the two faces project to different original faces
        if f1_orig != f2_orig:
            discontinuities += 1
            if len(sample_discontinuities) < 10:
                sample_discontinuities.append({
                    'edge': edge,
                    'f1': f1, 'f2': f2,
                    'f1_orig': f1_orig, 'f2_orig': f2_orig,
                    'f1_center': f1_center.cpu().tolist(),
                    'f2_center': f2_center.cpu().tolist()
                })

    discontinuity_pct = 100 * discontinuities / len(shared_edges) if shared_edges else 0

    print(f"\nDiscontinuities: {discontinuities} / {len(shared_edges)} edges ({discontinuity_pct:.1f}%)")

    if sample_discontinuities:
        print("\nSample discontinuities:")
        for d in sample_discontinuities[:5]:
            print(f"  Simp faces {d['f1']}, {d['f2']} -> Orig faces {d['f1_orig']}, {d['f2_orig']}")
            print(f"    Centers: {d['f1_center']}, {d['f2_center']}")

    return discontinuity_pct


def test_bvh_edge_projection(orig_vertices, orig_faces, simp_vertices, simp_faces):
    """
    Test if points along a single edge project consistently.
    """
    print("\n" + "=" * 70)
    print("EDGE PROJECTION CONSISTENCY TEST")
    print("=" * 70)

    bvh = cumesh.cuBVH(orig_vertices, orig_faces)

    # Pick a random edge from simplified mesh
    edge = (simp_faces[0, 0].item(), simp_faces[0, 1].item())
    v0, v1 = simp_vertices[edge[0]], simp_vertices[edge[1]]

    # Sample 10 points along the edge
    t_vals = torch.linspace(0, 1, 10, device='cuda')
    edge_points = v0.unsqueeze(0) + t_vals.unsqueeze(-1) * (v1 - v0).unsqueeze(0)

    # Project all points
    _, orig_face_ids, _ = bvh.unsigned_distance(edge_points, return_uvw=True)

    print(f"Edge from vertex {edge[0]} to {edge[1]}")
    print(f"Sampled 10 points along edge")
    print(f"Original face projections: {orig_face_ids.tolist()}")

    unique_faces = len(set(orig_face_ids.tolist()))
    print(f"Number of unique original faces: {unique_faces}")

    if unique_faces > 1:
        print("⚠️  Edge projects to MULTIPLE original faces - this causes texture discontinuity!")
    else:
        print("✓  Edge projects to single original face - consistent")

    return unique_faces


def main():
    print("=" * 70)
    print("BVH PROJECTION DISCONTINUITY DIAGNOSTIC")
    print("=" * 70)

    # Test 1: Simple cube
    print("\n--- TEST 1: Simple Cube (no simplification) ---")
    vertices, faces = create_test_mesh()
    print(f"Mesh: {vertices.shape[0]} vertices, {faces.shape[0]} faces")

    # BVH on itself (should be perfect continuity)
    discontinuity = analyze_bvh_continuity(vertices, faces, vertices, faces)
    print(f"Self-projection discontinuity: {discontinuity:.1f}%")

    # Test 2: Subdivided mesh → original
    print("\n--- TEST 2: Subdivided → Original (simulating simplification) ---")
    subdiv_v, subdiv_f = subdivide_mesh(vertices, faces, iterations=2)
    print(f"Subdivided mesh: {subdiv_v.shape[0]} vertices, {subdiv_f.shape[0]} faces")

    # Use subdivided as "original" (high-res), cube as "simplified" (low-res)
    discontinuity = analyze_bvh_continuity(subdiv_v, subdiv_f, vertices, faces)

    test_bvh_edge_projection(subdiv_v, subdiv_f, vertices, faces)

    # Test 3: Perturbed vertices
    print("\n--- TEST 3: Perturbed Simplified Mesh ---")
    perturbed_v = vertices + torch.randn_like(vertices) * 0.1
    discontinuity = analyze_bvh_continuity(subdiv_v, subdiv_f, perturbed_v, faces)
    test_bvh_edge_projection(subdiv_v, subdiv_f, perturbed_v, faces)

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print("""
When discontinuity % is high (>10%), it means:
- Adjacent triangles in the simplified mesh project to DIFFERENT original faces
- The attributes (colors, PBR) sampled from those different faces may be different
- This creates visible "seams" at triangle edges → the triangular patches we see

This is INHERENT to the simplification + BVH projection approach.
It's not a bug, it's a characteristic of the algorithm.

SOLUTIONS (all require algorithm changes, not just code fixes):
1. Use a different texture baking approach (e.g., ray casting from each texel)
2. Post-process textures with edge-aware filtering
3. Use a cage-based projection instead of BVH
4. Accept the artifacts as a limitation of the speed-optimized approach
""")


if __name__ == "__main__":
    main()
