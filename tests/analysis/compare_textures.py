"""Compare texture colors between our GLB and official."""
from pathlib import Path
import numpy as np
import trimesh

TRELLIS_ROOT = Path(__file__).parent

def analyze_glb(path, name):
    """Extract and analyze texture from GLB."""
    if not path.exists():
        print(f"{name}: File not found at {path}")
        return None

    scene = trimesh.load(str(path))

    if isinstance(scene, trimesh.Scene):
        for gname, geom in scene.geometry.items():
            if hasattr(geom, 'visual') and hasattr(geom.visual, 'material'):
                mat = geom.visual.material
                if hasattr(mat, 'baseColorTexture') and mat.baseColorTexture is not None:
                    tex = np.array(mat.baseColorTexture)

                    if tex.shape[-1] >= 3:
                        rgb = tex[..., :3]
                        # Mask for non-black pixels
                        valid = rgb.max(axis=-1) > 10
                        valid_rgb = rgb[valid]

                        r_mean = valid_rgb[:, 0].mean()
                        g_mean = valid_rgb[:, 1].mean()
                        b_mean = valid_rgb[:, 2].mean()
                        brightness = (r_mean + g_mean + b_mean) / 3

                        return {
                            'name': name,
                            'r': r_mean,
                            'g': g_mean,
                            'b': b_mean,
                            'brightness': brightness,
                            'valid_pixels': valid_rgb.shape[0]
                        }
    return None


def main():
    print("="*70)
    print("TEXTURE COLOR COMPARISON")
    print("="*70)

    files = [
        (TRELLIS_ROOT / "reference" / "spiral_official.glb", "Official"),
        (TRELLIS_ROOT / "outputs" / "generation_test" / "seed789_spiral.glb", "Seed 789"),
        (TRELLIS_ROOT / "outputs" / "generation_test" / "no_autocast_test.glb", "Seed 42 (no autocast)"),
    ]

    results = []
    for path, name in files:
        result = analyze_glb(path, name)
        if result:
            results.append(result)
            print(f"\n{name}:")
            print(f"  R={result['r']:.1f}, G={result['g']:.1f}, B={result['b']:.1f}")
            print(f"  Brightness: {result['brightness']:.1f}")

    if len(results) >= 2:
        official = results[0]
        print(f"\n{'='*70}")
        print("COMPARISON TO OFFICIAL")
        print(f"{'='*70}")
        print(f"{'Name':<30} {'Brightness':>12} {'Diff':>12}")
        print("-" * 54)
        for r in results:
            diff = r['brightness'] - official['brightness']
            sign = '+' if diff > 0 else ''
            print(f"{r['name']:<30} {r['brightness']:>12.1f} {sign}{diff:>11.1f}")


if __name__ == "__main__":
    main()
