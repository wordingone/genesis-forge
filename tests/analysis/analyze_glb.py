"""
GLB Structural Analysis Script
Compares structural metrics between generated output and benchmark GLB using trimesh.
"""
import json
import sys
import numpy as np
from pathlib import Path

TRELLIS_ROOT = Path(__file__).parent
OUTPUT_DIR = TRELLIS_ROOT / "outputs" / "generation_test"

GENERATED_GLB = OUTPUT_DIR / "generated_output.glb"
BENCHMARK_GLB = TRELLIS_ROOT / "reference" / "sample_2026-01-24T055452.643.glb"


def analyze_mesh(glb_path: Path) -> dict:
    """Analyze a single GLB file and return structural metrics."""
    import trimesh

    scene = trimesh.load(str(glb_path))

    # Handle both single mesh and scene
    if isinstance(scene, trimesh.Scene):
        meshes = [g for g in scene.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            return {"error": "No meshes found in scene"}
        # Combine all meshes for aggregate stats
        combined = trimesh.util.concatenate(meshes)
    else:
        combined = scene
        meshes = [scene]

    # Basic counts
    result = {
        "num_meshes": len(meshes),
        "total_vertices": int(combined.vertices.shape[0]),
        "total_faces": int(combined.faces.shape[0]),
    }

    # Bounding box
    result["bounds_min"] = combined.bounds[0].tolist()
    result["bounds_max"] = combined.bounds[1].tolist()
    result["extents"] = combined.extents.tolist()
    result["bounding_box_volume"] = float(np.prod(combined.extents))

    # Topology
    result["is_watertight"] = bool(combined.is_watertight)
    try:
        import networkx as nx
        edges = combined.face_adjacency
        g = nx.Graph()
        g.add_edges_from(edges)
        result["connected_components"] = nx.number_connected_components(g)
    except Exception:
        result["connected_components"] = "N/A (too large for split)"

    # Surface metrics
    result["surface_area"] = float(combined.area)

    # Face quality
    face_areas = combined.area_faces
    result["degenerate_faces"] = int(np.sum(face_areas < 1e-10))
    result["min_face_area"] = float(np.min(face_areas))
    result["max_face_area"] = float(np.max(face_areas))
    result["mean_face_area"] = float(np.mean(face_areas))

    # Edge statistics
    edges = combined.edges_unique_length
    result["min_edge_length"] = float(np.min(edges))
    result["max_edge_length"] = float(np.max(edges))
    result["mean_edge_length"] = float(np.mean(edges))
    result["std_edge_length"] = float(np.std(edges))

    # Material info
    has_material = False
    has_base_color_tex = False
    has_metallic_roughness_tex = False
    has_normal_tex = False

    for m in meshes:
        if hasattr(m, 'visual') and hasattr(m.visual, 'material'):
            mat = m.visual.material
            has_material = True
            if hasattr(mat, 'baseColorTexture') and mat.baseColorTexture is not None:
                has_base_color_tex = True
            if hasattr(mat, 'metallicRoughnessTexture') and mat.metallicRoughnessTexture is not None:
                has_metallic_roughness_tex = True
            if hasattr(mat, 'normalTexture') and mat.normalTexture is not None:
                has_normal_tex = True

    result["has_material"] = has_material
    result["has_base_color_texture"] = has_base_color_tex
    result["has_metallic_roughness_texture"] = has_metallic_roughness_tex
    result["has_normal_texture"] = has_normal_tex

    return result


def main():
    if not GENERATED_GLB.exists():
        print(f"ERROR: Generated GLB not found: {GENERATED_GLB}")
        sys.exit(1)
    if not BENCHMARK_GLB.exists():
        print(f"ERROR: Benchmark GLB not found: {BENCHMARK_GLB}")
        sys.exit(1)

    print("Analyzing generated GLB...")
    gen_metrics = analyze_mesh(GENERATED_GLB)

    print("Analyzing benchmark GLB...")
    bench_metrics = analyze_mesh(BENCHMARK_GLB)

    # Comparison
    comparison = {}
    for key in gen_metrics:
        if key in bench_metrics:
            g = gen_metrics[key]
            b = bench_metrics[key]
            if isinstance(g, (int, float)) and isinstance(b, (int, float)) and b != 0:
                comparison[key] = {
                    "generated": g,
                    "benchmark": b,
                    "ratio": round(g / b, 3) if b != 0 else None,
                }
            else:
                comparison[key] = {
                    "generated": g,
                    "benchmark": b,
                }

    report = {
        "generated": gen_metrics,
        "benchmark": bench_metrics,
        "comparison": comparison,
    }

    output_path = OUTPUT_DIR / "structural_analysis.json"
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nStructural analysis saved: {output_path}")

    # Print summary table
    print(f"\n{'='*70}")
    print(f"  Structural Comparison: Generated vs Benchmark")
    print(f"{'='*70}")
    print(f"  {'Metric':<35} {'Generated':>15} {'Benchmark':>15}")
    print(f"  {'-'*35} {'-'*15} {'-'*15}")
    for key in ["total_vertices", "total_faces", "connected_components",
                 "is_watertight", "surface_area", "degenerate_faces",
                 "has_base_color_texture", "has_metallic_roughness_texture"]:
        g = gen_metrics.get(key, "N/A")
        b = bench_metrics.get(key, "N/A")
        if isinstance(g, float):
            g = f"{g:.4f}"
        if isinstance(b, float):
            b = f"{b:.4f}"
        print(f"  {key:<35} {str(g):>15} {str(b):>15}")

    print(f"\n  Extents (Generated): {gen_metrics.get('extents', 'N/A')}")
    print(f"  Extents (Benchmark): {bench_metrics.get('extents', 'N/A')}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
