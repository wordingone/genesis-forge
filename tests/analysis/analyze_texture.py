"""
Texture Analysis Script
Extracts and analyzes textures from both generated and benchmark GLBs.
Computes color stats, black pixel ratio, repetitive pattern detection,
UV coverage, histogram distance, and PBR metallic/roughness stats.
"""
import json
import sys
import numpy as np
from pathlib import Path
from PIL import Image

TRELLIS_ROOT = Path(__file__).parent
OUTPUT_DIR = TRELLIS_ROOT / "outputs" / "generation_test"

GENERATED_GLB = OUTPUT_DIR / "generated_output.glb"
BENCHMARK_GLB = TRELLIS_ROOT / "reference" / "sample_2026-01-24T055452.643.glb"


def extract_textures(glb_path: Path, output_dir: Path) -> dict:
    """Extract textures from a GLB file and save as PNGs. Return texture info."""
    import trimesh

    output_dir.mkdir(parents=True, exist_ok=True)
    scene = trimesh.load(str(glb_path))

    if isinstance(scene, trimesh.Scene):
        meshes = list(scene.geometry.values())
    else:
        meshes = [scene]

    textures = {}
    for i, mesh in enumerate(meshes):
        if not isinstance(mesh, trimesh.Trimesh):
            continue
        if not hasattr(mesh, 'visual') or not hasattr(mesh.visual, 'material'):
            continue

        mat = mesh.visual.material

        # Base color texture
        if hasattr(mat, 'baseColorTexture') and mat.baseColorTexture is not None:
            tex = mat.baseColorTexture
            if isinstance(tex, Image.Image):
                img = tex
            elif hasattr(tex, 'image'):
                img = tex.image
            else:
                img = Image.fromarray(np.array(tex))
            path = output_dir / "base_color.png"
            img.save(str(path))
            textures["base_color"] = {"path": str(path), "size": img.size}
        elif hasattr(mat, 'image') and mat.image is not None:
            img = mat.image
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            path = output_dir / "base_color.png"
            img.save(str(path))
            textures["base_color"] = {"path": str(path), "size": img.size}

        # Metallic-roughness texture
        if hasattr(mat, 'metallicRoughnessTexture') and mat.metallicRoughnessTexture is not None:
            tex = mat.metallicRoughnessTexture
            if isinstance(tex, Image.Image):
                img = tex
            elif hasattr(tex, 'image'):
                img = tex.image
            else:
                img = Image.fromarray(np.array(tex))
            path = output_dir / "metallic_roughness.png"
            img.save(str(path))
            textures["metallic_roughness"] = {"path": str(path), "size": img.size}

        # Normal texture
        if hasattr(mat, 'normalTexture') and mat.normalTexture is not None:
            tex = mat.normalTexture
            if isinstance(tex, Image.Image):
                img = tex
            elif hasattr(tex, 'image'):
                img = tex.image
            else:
                img = Image.fromarray(np.array(tex))
            path = output_dir / "normal.png"
            img.save(str(path))
            textures["normal"] = {"path": str(path), "size": img.size}

        break  # Analyze first mesh with material

    return textures


def analyze_texture_image(img_path: str) -> dict:
    """Analyze a texture image for quality metrics."""
    img = Image.open(img_path).convert("RGB")
    arr = np.array(img, dtype=np.float32)

    h, w = arr.shape[:2]

    # Per-channel statistics
    channel_stats = {}
    for c, name in enumerate(["R", "G", "B"]):
        ch = arr[:, :, c]
        channel_stats[name] = {
            "mean": float(np.mean(ch)),
            "std": float(np.std(ch)),
            "min": float(np.min(ch)),
            "max": float(np.max(ch)),
        }

    # Black pixel ratio (unmapped UV regions)
    brightness = np.mean(arr, axis=2)
    black_pixels = np.sum(brightness < 5)
    black_ratio = float(black_pixels / (h * w))

    # Repetitive pattern detection via local patch variance
    patch_size = 64
    variances = []
    for y in range(0, h - patch_size, patch_size):
        for x in range(0, w - patch_size, patch_size):
            patch = arr[y:y+patch_size, x:x+patch_size]
            variances.append(float(np.var(patch)))

    mean_patch_variance = float(np.mean(variances)) if variances else 0
    std_patch_variance = float(np.std(variances)) if variances else 0

    # Color histogram (per channel, 32 bins)
    histograms = {}
    for c, name in enumerate(["R", "G", "B"]):
        hist, _ = np.histogram(arr[:, :, c], bins=32, range=(0, 256))
        histograms[name] = hist.tolist()

    return {
        "size": [w, h],
        "channel_stats": channel_stats,
        "black_pixel_ratio": black_ratio,
        "mean_patch_variance": mean_patch_variance,
        "std_patch_variance": std_patch_variance,
        "histograms": histograms,
    }


def bhattacharyya_distance(hist1: list, hist2: list) -> float:
    """Compute Bhattacharyya distance between two histograms."""
    h1 = np.array(hist1, dtype=np.float64)
    h2 = np.array(hist2, dtype=np.float64)
    # Normalize
    h1 = h1 / (np.sum(h1) + 1e-10)
    h2 = h2 / (np.sum(h2) + 1e-10)
    # Bhattacharyya coefficient
    bc = np.sum(np.sqrt(h1 * h2))
    # Distance
    dist = -np.log(bc + 1e-10)
    return float(dist)


def analyze_metallic_roughness(img_path: str) -> dict:
    """Analyze metallic-roughness texture (glTF convention: B=metallic, G=roughness)."""
    img = Image.open(img_path).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0

    metallic = arr[:, :, 2]  # Blue channel
    roughness = arr[:, :, 1]  # Green channel

    return {
        "metallic_mean": float(np.mean(metallic)),
        "metallic_std": float(np.std(metallic)),
        "roughness_mean": float(np.mean(roughness)),
        "roughness_std": float(np.std(roughness)),
    }


def main():
    if not GENERATED_GLB.exists():
        print(f"ERROR: Generated GLB not found: {GENERATED_GLB}")
        sys.exit(1)
    if not BENCHMARK_GLB.exists():
        print(f"ERROR: Benchmark GLB not found: {BENCHMARK_GLB}")
        sys.exit(1)

    ours_dir = OUTPUT_DIR / "textures_ours"
    bench_dir = OUTPUT_DIR / "textures_bench"

    print("Extracting textures from generated GLB...")
    ours_textures = extract_textures(GENERATED_GLB, ours_dir)
    print(f"  Found textures: {list(ours_textures.keys())}")

    print("Extracting textures from benchmark GLB...")
    bench_textures = extract_textures(BENCHMARK_GLB, bench_dir)
    print(f"  Found textures: {list(bench_textures.keys())}")

    report = {
        "generated_textures": {},
        "benchmark_textures": {},
        "comparison": {},
    }

    # Analyze base color textures
    for label, textures, report_key in [
        ("Generated", ours_textures, "generated_textures"),
        ("Benchmark", bench_textures, "benchmark_textures"),
    ]:
        if "base_color" in textures:
            print(f"\nAnalyzing {label} base color texture...")
            analysis = analyze_texture_image(textures["base_color"]["path"])
            analysis["texture_size"] = textures["base_color"]["size"]
            report[report_key]["base_color"] = analysis

        if "metallic_roughness" in textures:
            print(f"Analyzing {label} metallic-roughness texture...")
            mr = analyze_metallic_roughness(textures["metallic_roughness"]["path"])
            report[report_key]["metallic_roughness"] = mr

    # Histogram distance comparison
    if "base_color" in report["generated_textures"] and "base_color" in report["benchmark_textures"]:
        gen_hists = report["generated_textures"]["base_color"]["histograms"]
        bench_hists = report["benchmark_textures"]["base_color"]["histograms"]

        distances = {}
        for channel in ["R", "G", "B"]:
            d = bhattacharyya_distance(gen_hists[channel], bench_hists[channel])
            distances[channel] = round(d, 4)

        distances["mean"] = round(np.mean(list(distances.values())), 4)
        report["comparison"]["histogram_distance"] = distances

        print(f"\n  Histogram Distance (Bhattacharyya):")
        for ch, d in distances.items():
            status = "PASS" if d < 0.1 else "WARN" if d < 0.2 else "FAIL"
            print(f"    {ch}: {d:.4f} [{status}]")

    # Black pixel comparison
    if "base_color" in report["generated_textures"] and "base_color" in report["benchmark_textures"]:
        gen_black = report["generated_textures"]["base_color"]["black_pixel_ratio"]
        bench_black = report["benchmark_textures"]["base_color"]["black_pixel_ratio"]
        report["comparison"]["black_pixel_ratio"] = {
            "generated": gen_black,
            "benchmark": bench_black,
        }
        print(f"\n  Black pixel ratio:")
        print(f"    Generated: {gen_black:.4f}")
        print(f"    Benchmark: {bench_black:.4f}")

    output_path = OUTPUT_DIR / "texture_analysis.json"
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nTexture analysis saved: {output_path}")


if __name__ == "__main__":
    main()
