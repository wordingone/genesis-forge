"""
Diagnostic script to analyze mesh quality and identify artifact sources.
Loads the generated GLB and checks for degenerate triangles, normal issues, etc.
"""
import numpy as np
import trimesh

def analyze_mesh(glb_path):
    """Analyze mesh for potential artifact sources."""
    print(f"\n{'='*60}")
    print(f"Analyzing: {glb_path}")
    print(f"{'='*60}\n")

    # Load GLB
    scene = trimesh.load(glb_path)
    if isinstance(scene, trimesh.Scene):
        mesh = scene.dump(concatenate=True)
    else:
        mesh = scene

    vertices = mesh.vertices
    faces = mesh.faces

    print(f"Mesh Statistics:")
    print(f"  Vertices: {len(vertices):,}")
    print(f"  Faces: {len(faces):,}")
    print(f"  Is watertight: {mesh.is_watertight}")
    print(f"  Is volume: {mesh.is_volume}")

    # Calculate face areas
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    # Cross product for area
    edge1 = v1 - v0
    edge2 = v2 - v0
    cross = np.cross(edge1, edge2)
    areas = 0.5 * np.linalg.norm(cross, axis=1)

    print(f"\nFace Area Analysis:")
    print(f"  Min area: {areas.min():.2e}")
    print(f"  Max area: {areas.max():.2e}")
    print(f"  Mean area: {areas.mean():.2e}")
    print(f"  Median area: {np.median(areas):.2e}")

    # Count degenerate faces (near-zero area)
    degenerate_threshold = 1e-10
    degenerate_count = np.sum(areas < degenerate_threshold)
    print(f"  Degenerate faces (area < {degenerate_threshold}): {degenerate_count}")

    # Count very small faces (potential artifacts)
    small_threshold = areas.mean() * 0.01  # 1% of mean
    small_count = np.sum(areas < small_threshold)
    print(f"  Very small faces (area < 1% of mean): {small_count}")

    # Calculate edge lengths
    edge_lengths = []
    for i, j in [(0,1), (1,2), (2,0)]:
        edge = vertices[faces[:, j]] - vertices[faces[:, i]]
        edge_lengths.append(np.linalg.norm(edge, axis=1))
    edge_lengths = np.array(edge_lengths)

    print(f"\nEdge Length Analysis:")
    print(f"  Min edge: {edge_lengths.min():.2e}")
    print(f"  Max edge: {edge_lengths.max():.2e}")
    print(f"  Mean edge: {edge_lengths.mean():.2e}")

    # Calculate aspect ratio (longest edge / shortest edge per face)
    min_edges = edge_lengths.min(axis=0)
    max_edges = edge_lengths.max(axis=0)
    aspect_ratios = max_edges / (min_edges + 1e-10)  # Avoid div by zero

    print(f"\nTriangle Aspect Ratio (sliver detection):")
    print(f"  Min ratio: {aspect_ratios.min():.2f}")
    print(f"  Max ratio: {aspect_ratios.max():.2f}")
    print(f"  Mean ratio: {aspect_ratios.mean():.2f}")

    # Count sliver triangles (aspect ratio > 10 is very elongated)
    sliver_threshold = 10
    sliver_count = np.sum(aspect_ratios > sliver_threshold)
    print(f"  Sliver triangles (ratio > {sliver_threshold}): {sliver_count}")

    extreme_sliver_count = np.sum(aspect_ratios > 100)
    print(f"  Extreme slivers (ratio > 100): {extreme_sliver_count}")

    # Analyze normals
    if mesh.vertex_normals is not None:
        normals = mesh.vertex_normals
        normal_lengths = np.linalg.norm(normals, axis=1)

        print(f"\nVertex Normal Analysis:")
        print(f"  Min length: {normal_lengths.min():.4f}")
        print(f"  Max length: {normal_lengths.max():.4f}")
        print(f"  Mean length: {normal_lengths.mean():.4f}")

        # Check for zero/nan normals
        zero_normals = np.sum(normal_lengths < 0.01)
        print(f"  Near-zero normals: {zero_normals}")

        nan_normals = np.sum(np.isnan(normal_lengths))
        print(f"  NaN normals: {nan_normals}")

    # Check face normals consistency
    face_normals = mesh.face_normals
    fn_lengths = np.linalg.norm(face_normals, axis=1)

    print(f"\nFace Normal Analysis:")
    zero_fn = np.sum(fn_lengths < 0.01)
    print(f"  Near-zero face normals: {zero_fn}")

    # Check for inverted normals (inconsistent winding)
    # Look for adjacent faces with opposing normals
    face_adjacency = mesh.face_adjacency
    if len(face_adjacency) > 0:
        adj_normals_1 = face_normals[face_adjacency[:, 0]]
        adj_normals_2 = face_normals[face_adjacency[:, 1]]
        dot_products = np.sum(adj_normals_1 * adj_normals_2, axis=1)

        # Opposing normals have dot product < 0
        opposing = np.sum(dot_products < -0.5)
        print(f"  Adjacent faces with opposing normals: {opposing}")

    # Connected components analysis
    components = mesh.split(only_watertight=False)
    print(f"\nConnected Components:")
    print(f"  Total components: {len(components)}")

    if len(components) > 1:
        component_sizes = [len(c.faces) for c in components]
        print(f"  Largest component: {max(component_sizes):,} faces")
        print(f"  Smallest component: {min(component_sizes):,} faces")

        # Count tiny components (potential floating artifacts)
        tiny_threshold = 100  # faces
        tiny_components = sum(1 for s in component_sizes if s < tiny_threshold)
        print(f"  Tiny components (<{tiny_threshold} faces): {tiny_components}")

    print(f"\n{'='*60}")
    print("DIAGNOSIS SUMMARY")
    print(f"{'='*60}")

    issues = []
    if degenerate_count > 0:
        issues.append(f"- {degenerate_count} degenerate faces (zero area)")
    if sliver_count > len(faces) * 0.01:  # More than 1%
        issues.append(f"- {sliver_count} sliver triangles ({100*sliver_count/len(faces):.1f}%)")
    if extreme_sliver_count > 0:
        issues.append(f"- {extreme_sliver_count} extreme sliver triangles")
    if zero_normals > 0:
        issues.append(f"- {zero_normals} vertices with near-zero normals")
    if zero_fn > 0:
        issues.append(f"- {zero_fn} faces with near-zero normals")
    if len(components) > 1:
        issues.append(f"- {len(components)} disconnected components (potential floating fragments)")

    if issues:
        print("POTENTIAL ARTIFACT SOURCES:")
        for issue in issues:
            print(issue)
    else:
        print("No obvious mesh quality issues detected.")

    return {
        'vertices': len(vertices),
        'faces': len(faces),
        'degenerate_faces': degenerate_count,
        'sliver_triangles': sliver_count,
        'extreme_slivers': extreme_sliver_count,
        'zero_normals': zero_normals if mesh.vertex_normals is not None else 0,
        'components': len(components),
    }


if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Default paths
    ours = Path("outputs/hybrid_precision_test/generated_spiral_input_seed42.glb")
    official = Path("outputs/hybrid_precision_test/spiral_official.glb")

    if len(sys.argv) > 1:
        ours = Path(sys.argv[1])

    if ours.exists():
        our_stats = analyze_mesh(str(ours))
    else:
        print(f"File not found: {ours}")

    print("\n" + "="*60)
    print("COMPARING WITH OFFICIAL")
    print("="*60)

    if official.exists():
        official_stats = analyze_mesh(str(official))

        print("\n" + "="*60)
        print("COMPARISON")
        print("="*60)
        print(f"{'Metric':<30} {'Ours':>15} {'Official':>15}")
        print("-"*60)
        for key in our_stats:
            print(f"{key:<30} {our_stats[key]:>15,} {official_stats[key]:>15,}")
    else:
        print(f"Official file not found: {official}")
