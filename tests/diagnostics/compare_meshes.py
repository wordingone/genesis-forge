"""
Diagnostic: Compare our generated mesh vs official benchmark.

Identifies structural differences that could cause texture patches.
"""
from pathlib import Path as _Path
ROOT = _Path(__file__).resolve().parents[2]  # repo root

import sys
import numpy as np
import trimesh
from pathlib import Path

def load_glb(path):
    """Load GLB and extract first mesh."""
    loaded = trimesh.load(path)
    if isinstance(loaded, trimesh.Scene):
        for name, geom in loaded.geometry.items():
            if isinstance(geom, trimesh.Trimesh):
                return geom, name
    return loaded, "mesh"

def analyze_mesh(mesh, name):
    """Deep analysis of mesh properties."""
    print(f"\n{'='*60}")
    print(f"MESH: {name}")
    print(f"{'='*60}")

    # Basic stats
    print(f"\n--- Geometry ---")
    print(f"  Vertices: {len(mesh.vertices):,}")
    print(f"  Faces: {len(mesh.faces):,}")
    print(f"  Edges: {len(mesh.edges):,}")

    # Bounds
    bounds = mesh.bounds
    size = bounds[1] - bounds[0]
    print(f"\n--- Bounds ---")
    print(f"  Min: [{bounds[0][0]:.4f}, {bounds[0][1]:.4f}, {bounds[0][2]:.4f}]")
    print(f"  Max: [{bounds[1][0]:.4f}, {bounds[1][1]:.4f}, {bounds[1][2]:.4f}]")
    print(f"  Size: [{size[0]:.4f}, {size[1]:.4f}, {size[2]:.4f}]")

    # Topology
    print(f"\n--- Topology ---")
    print(f"  Watertight: {mesh.is_watertight}")
    print(f"  Euler number: {mesh.euler_number}")
    # Skip connected components - can cause memory issues on large meshes

    # Edge analysis
    edge_counts = np.bincount(mesh.edges_unique_inverse)
    boundary_edges = np.sum(edge_counts == 1)
    manifold_edges = np.sum(edge_counts == 2)
    non_manifold_edges = np.sum(edge_counts > 2)
    print(f"\n--- Edges ---")
    print(f"  Boundary (1 face): {boundary_edges:,}")
    print(f"  Manifold (2 faces): {manifold_edges:,}")
    print(f"  Non-manifold (>2): {non_manifold_edges:,}")

    # Face analysis
    v0 = mesh.vertices[mesh.faces[:, 0]]
    v1 = mesh.vertices[mesh.faces[:, 1]]
    v2 = mesh.vertices[mesh.faces[:, 2]]

    # Face areas
    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    print(f"\n--- Face Areas ---")
    print(f"  Min: {areas.min():.2e}")
    print(f"  Max: {areas.max():.4f}")
    print(f"  Mean: {areas.mean():.6f}")
    print(f"  Std: {areas.std():.6f}")
    print(f"  Zero-area (<1e-10): {np.sum(areas < 1e-10)}")
    print(f"  Near-zero (<1e-7): {np.sum(areas < 1e-7)}")
    print(f"  Small (<1e-5): {np.sum(areas < 1e-5)}")

    # Degenerate faces (edge lengths)
    edge1 = np.linalg.norm(v1 - v0, axis=1)
    edge2 = np.linalg.norm(v2 - v1, axis=1)
    edge3 = np.linalg.norm(v0 - v2, axis=1)
    degenerate = (edge1 < 1e-7) | (edge2 < 1e-7) | (edge3 < 1e-7)
    print(f"\n--- Degenerate Faces ---")
    print(f"  Count (edge < 1e-7): {np.sum(degenerate)}")

    # UV analysis
    if hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None:
        uv = mesh.visual.uv
        print(f"\n--- UVs ---")
        print(f"  UV count: {len(uv):,}")
        print(f"  UV range U: [{uv[:, 0].min():.4f}, {uv[:, 0].max():.4f}]")
        print(f"  UV range V: [{uv[:, 1].min():.4f}, {uv[:, 1].max():.4f}]")

        # Check for UV overlaps or bad values
        uv_out_of_range = np.sum((uv < 0) | (uv > 1))
        print(f"  Out of [0,1] range: {uv_out_of_range}")

    # Texture analysis
    if hasattr(mesh.visual, 'material') and mesh.visual.material is not None:
        mat = mesh.visual.material
        print(f"\n--- Material ---")
        print(f"  Type: {type(mat).__name__}")
        if hasattr(mat, 'baseColorTexture') and mat.baseColorTexture is not None:
            tex = mat.baseColorTexture
            print(f"  Base color texture: {tex.size if hasattr(tex, 'size') else 'N/A'}")
        if hasattr(mat, 'metallicRoughnessTexture') and mat.metallicRoughnessTexture is not None:
            tex = mat.metallicRoughnessTexture
            print(f"  Metallic/Roughness texture: {tex.size if hasattr(tex, 'size') else 'N/A'}")

    return {
        'vertices': len(mesh.vertices),
        'faces': len(mesh.faces),
        'boundary_edges': boundary_edges,
        'non_manifold_edges': non_manifold_edges,
        'degenerate_faces': np.sum(degenerate),
        'mean_area': areas.mean(),
        'min_area': areas.min(),
    }


def compare_textures(mesh1, mesh2, name1, name2):
    """Compare texture maps between two meshes."""
    print(f"\n{'='*60}")
    print(f"TEXTURE COMPARISON: {name1} vs {name2}")
    print(f"{'='*60}")

    def get_texture(mesh, tex_name):
        if not hasattr(mesh.visual, 'material'):
            return None
        mat = mesh.visual.material
        if hasattr(mat, tex_name):
            tex = getattr(mat, tex_name)
            if tex is not None:
                return np.array(tex)
        return None

    for tex_name in ['baseColorTexture', 'metallicRoughnessTexture']:
        tex1 = get_texture(mesh1, tex_name)
        tex2 = get_texture(mesh2, tex_name)

        print(f"\n--- {tex_name} ---")
        if tex1 is None:
            print(f"  {name1}: None")
        else:
            print(f"  {name1}: shape={tex1.shape}, dtype={tex1.dtype}")
            print(f"    Range: [{tex1.min()}, {tex1.max()}]")
            print(f"    Mean: {tex1.mean():.2f}")

        if tex2 is None:
            print(f"  {name2}: None")
        else:
            print(f"  {name2}: shape={tex2.shape}, dtype={tex2.dtype}")
            print(f"    Range: [{tex2.min()}, {tex2.max()}]")
            print(f"    Mean: {tex2.mean():.2f}")

        # Compare if both exist and same shape
        if tex1 is not None and tex2 is not None and tex1.shape == tex2.shape:
            diff = np.abs(tex1.astype(float) - tex2.astype(float))
            print(f"  Difference:")
            print(f"    Max diff: {diff.max():.2f}")
            print(f"    Mean diff: {diff.mean():.2f}")
            print(f"    Pixels > 10 diff: {np.sum(diff > 10):,}")
            print(f"    Pixels > 50 diff: {np.sum(diff > 50):,}")


def main():
    # Paths - compare specific files
    official_path = Path(rf"{ROOT}\outputs\benchmarks\spiral_official.glb")
    our_path = Path(rf"{ROOT}\outputs\benchmarks\hybrid_precision_test\generated_spiral_input_seed42.glb")

    our_paths = [our_path] if our_path.exists() else []

    # Also check for other spiral outputs
    gen_dir = Path(rf"{ROOT}\outputs\benchmarks\hybrid_precision_test")
    if gen_dir.exists():
        for p in gen_dir.glob("generated_spiral*.glb"):
            if p not in our_paths:
                our_paths.append(p)

    print("="*60)
    print("MESH COMPARISON DIAGNOSTIC")
    print("="*60)

    # Load official
    if not official_path.exists():
        print(f"ERROR: Official benchmark not found at {official_path}")
        return

    official_mesh, official_name = load_glb(official_path)
    official_stats = analyze_mesh(official_mesh, f"OFFICIAL: {official_path.name}")

    # Load and compare each of our outputs
    if not our_paths:
        print(f"\nNo generated spiral GLBs found in:")
        print(f"  - {gen_dir}")
        print(f"  - {gen_dir2}")
        print("\nRun a generation first, then re-run this diagnostic.")
        return

    for our_path in sorted(our_paths, key=lambda p: p.stat().st_mtime, reverse=True)[:3]:
        our_mesh, our_name = load_glb(our_path)
        our_stats = analyze_mesh(our_mesh, f"OURS: {our_path.name}")

        # Compare
        print(f"\n{'='*60}")
        print(f"COMPARISON SUMMARY")
        print(f"{'='*60}")
        print(f"  Vertex diff: {our_stats['vertices'] - official_stats['vertices']:+,}")
        print(f"  Face diff: {our_stats['faces'] - official_stats['faces']:+,}")
        print(f"  Boundary edge diff: {our_stats['boundary_edges'] - official_stats['boundary_edges']:+,}")
        print(f"  Degenerate face diff: {our_stats['degenerate_faces'] - official_stats['degenerate_faces']:+,}")

        # Texture comparison
        compare_textures(official_mesh, our_mesh, "Official", "Ours")

        print("\n" + "-"*60)


if __name__ == "__main__":
    main()
