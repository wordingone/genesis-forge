"""
Diagnostic: Test mesh watertightness at each stage of postprocess pipeline.

This identifies WHERE watertight property is lost in our Windows-compiled cumesh.
"""

import sys
import torch
import numpy as np
import trimesh

# Add paths
sys.path.insert(0, r"B:\M\ArtificialArchitecture\spatial\trellis-forge")

def check_watertight(vertices, faces, stage_name):
    """Check if mesh is watertight and report issues."""
    if isinstance(vertices, torch.Tensor):
        vertices = vertices.cpu().numpy()
    if isinstance(faces, torch.Tensor):
        faces = faces.cpu().numpy()

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    # Watertight checks
    is_watertight = mesh.is_watertight
    euler = mesh.euler_number
    num_holes = len(mesh.outline().entities) if hasattr(mesh.outline(), 'entities') else 0

    # Degenerate face check
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    edge1 = np.linalg.norm(v1 - v0, axis=1)
    edge2 = np.linalg.norm(v2 - v1, axis=1)
    edge3 = np.linalg.norm(v0 - v2, axis=1)
    degenerate_count = np.sum((edge1 < 1e-7) | (edge2 < 1e-7) | (edge3 < 1e-7))

    # Non-manifold edges
    edges = mesh.edges_unique
    edges_face_count = mesh.edges_unique_inverse
    # Count how many faces share each edge
    edge_counts = np.bincount(edges_face_count)
    non_manifold_edges = np.sum(edge_counts > 2)
    boundary_edges = np.sum(edge_counts == 1)

    print(f"\n{'='*60}")
    print(f"Stage: {stage_name}")
    print(f"{'='*60}")
    print(f"  Vertices: {len(vertices):,}")
    print(f"  Faces: {len(faces):,}")
    print(f"  Watertight: {is_watertight}")
    print(f"  Euler number: {euler} (should be 2 for closed mesh)")
    print(f"  Boundary edges: {boundary_edges} (should be 0 for watertight)")
    print(f"  Non-manifold edges: {non_manifold_edges} (should be 0)")
    print(f"  Degenerate faces: {degenerate_count}")

    if not is_watertight:
        print(f"  WARNING: Mesh is NOT watertight!")

    return {
        'stage': stage_name,
        'vertices': len(vertices),
        'faces': len(faces),
        'watertight': is_watertight,
        'euler': euler,
        'boundary_edges': boundary_edges,
        'non_manifold_edges': non_manifold_edges,
        'degenerate_faces': degenerate_count,
    }


def run_diagnostic():
    """Run mesh through cumesh pipeline stages with watertight checks."""
    import cumesh

    print("="*60)
    print("CUMESH WATERTIGHT DIAGNOSTIC")
    print("="*60)

    # Load a test mesh - use the reference spiral if available
    test_glb = r"B:\M\ArtificialArchitecture\spatial\trellis-forge\outputs\benchmarks\spiral_official.glb"

    try:
        loaded = trimesh.load(test_glb)
        if isinstance(loaded, trimesh.Scene):
            # Get the first mesh from scene
            for name, geom in loaded.geometry.items():
                if isinstance(geom, trimesh.Trimesh):
                    test_mesh = geom
                    break
        else:
            test_mesh = loaded

        vertices = torch.tensor(test_mesh.vertices, dtype=torch.float32).cuda()
        faces = torch.tensor(test_mesh.faces, dtype=torch.int32).cuda()
        print(f"\nLoaded test mesh from: {test_glb}")
    except Exception as e:
        print(f"Could not load {test_glb}: {e}")
        print("Creating synthetic test mesh (sphere)...")
        sphere = trimesh.creation.icosphere(subdivisions=4, radius=1.0)
        vertices = torch.tensor(sphere.vertices, dtype=torch.float32).cuda()
        faces = torch.tensor(sphere.faces, dtype=torch.int32).cuda()

    results = []

    # Stage 0: Initial mesh
    results.append(check_watertight(vertices, faces, "0. Initial mesh"))

    # Initialize cumesh
    mesh = cumesh.CuMesh()
    mesh.init(vertices, faces)

    # Stage 1: After fill_holes
    mesh.fill_holes(max_hole_perimeter=3e-2)
    v, f = mesh.read()
    results.append(check_watertight(v, f, "1. After fill_holes"))

    # Stage 2: After first simplify (3x target)
    target = 100000  # Use reasonable target
    mesh.simplify(target * 3, verbose=False)
    v, f = mesh.read()
    results.append(check_watertight(v, f, "2. After simplify (3x target)"))

    # Stage 3: After remove_duplicate_faces
    mesh.remove_duplicate_faces()
    v, f = mesh.read()
    results.append(check_watertight(v, f, "3. After remove_duplicate_faces"))

    # Stage 4: After repair_non_manifold_edges
    mesh.repair_non_manifold_edges()
    v, f = mesh.read()
    results.append(check_watertight(v, f, "4. After repair_non_manifold_edges"))

    # Stage 5: After remove_small_connected_components
    mesh.remove_small_connected_components(1e-5)
    v, f = mesh.read()
    results.append(check_watertight(v, f, "5. After remove_small_components"))

    # Stage 6: After second fill_holes
    mesh.fill_holes(max_hole_perimeter=3e-2)
    v, f = mesh.read()
    results.append(check_watertight(v, f, "6. After fill_holes (2nd)"))

    # Stage 7: After final simplify
    mesh.simplify(target, verbose=False)
    v, f = mesh.read()
    results.append(check_watertight(v, f, "7. After simplify (final)"))

    # Stage 8: After final cleanup
    mesh.remove_duplicate_faces()
    mesh.repair_non_manifold_edges()
    mesh.remove_small_connected_components(1e-5)
    mesh.fill_holes(max_hole_perimeter=3e-2)
    v, f = mesh.read()
    results.append(check_watertight(v, f, "8. After final cleanup"))

    # Stage 9: After unify_face_orientations
    mesh.unify_face_orientations()
    v, f = mesh.read()
    results.append(check_watertight(v, f, "9. After unify_face_orientations"))

    # Summary
    print("\n" + "="*60)
    print("SUMMARY: Where watertight is lost")
    print("="*60)

    was_watertight = results[0]['watertight']
    for i, r in enumerate(results):
        status = "OK" if r['watertight'] else "BROKEN"
        if i > 0 and was_watertight and not r['watertight']:
            status = "<<< BREAKS HERE"
        print(f"  {r['stage']}: {status} (boundary={r['boundary_edges']}, euler={r['euler']})")
        was_watertight = r['watertight']

    return results


if __name__ == "__main__":
    run_diagnostic()
