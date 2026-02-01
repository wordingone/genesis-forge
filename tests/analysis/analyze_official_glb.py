"""Analyze official GLB to understand its texture quality."""
from pathlib import Path
import trimesh
import numpy as np
from PIL import Image

TRELLIS_ROOT = Path(__file__).parent

def main():
    official_path = TRELLIS_ROOT / "reference" / "spiral_official.glb"
    our_path = TRELLIS_ROOT / "outputs" / "generation_test" / "no_autocast_test.glb"

    print("=" * 70)
    print("ANALYZING OFFICIAL vs OUR GLB")
    print("=" * 70)

    for name, path in [("Official", official_path), ("Ours", our_path)]:
        print(f"\n{'='*70}")
        print(f"  {name}: {path}")
        print("=" * 70)

        if not path.exists():
            print(f"  File not found!")
            continue

        scene = trimesh.load(str(path))

        if isinstance(scene, trimesh.Scene):
            print(f"\n  Geometries: {len(scene.geometry)}")
            for gname, geom in scene.geometry.items():
                print(f"\n  Geometry: {gname}")
                print(f"    Vertices: {len(geom.vertices)}")
                print(f"    Faces: {len(geom.faces)}")

                if hasattr(geom, 'visual') and hasattr(geom.visual, 'material'):
                    mat = geom.visual.material
                    print(f"    Material type: {type(mat).__name__}")

                    # Check base color texture
                    if hasattr(mat, 'baseColorTexture') and mat.baseColorTexture is not None:
                        tex = mat.baseColorTexture
                        if hasattr(tex, 'size'):
                            print(f"    Base color texture: {tex.size}")
                        elif hasattr(tex, 'shape'):
                            print(f"    Base color texture: {tex.shape}")

                        # Convert to numpy for analysis
                        tex_arr = np.array(tex)
                        print(f"    Texture array shape: {tex_arr.shape}")
                        print(f"    Texture dtype: {tex_arr.dtype}")

                        # Analyze color distribution
                        if len(tex_arr.shape) == 3:
                            # RGB channels
                            for c, cname in enumerate(['R', 'G', 'B']):
                                if c < tex_arr.shape[-1]:
                                    ch = tex_arr[..., c]
                                    print(f"      {cname}: min={ch.min()}, max={ch.max()}, mean={ch.mean():.1f}, std={ch.std():.1f}")

                            # Check for black pixels
                            if tex_arr.shape[-1] >= 3:
                                black_mask = (tex_arr[..., :3].sum(axis=-1) == 0)
                                black_pct = 100 * black_mask.sum() / black_mask.size
                                print(f"      Black pixels (RGB=0): {black_mask.sum()} ({black_pct:.2f}%)")

                            # Check for fully saturated pixels
                            white_mask = (tex_arr[..., :3] == 255).all(axis=-1)
                            white_pct = 100 * white_mask.sum() / white_mask.size
                            print(f"      White pixels (RGB=255): {white_mask.sum()} ({white_pct:.2f}%)")

                        # Save texture for visual comparison
                        output_dir = TRELLIS_ROOT / "outputs" / "glb_analysis"
                        output_dir.mkdir(parents=True, exist_ok=True)
                        tex_path = output_dir / f"{name.lower()}_{gname}_baseColor.png"
                        Image.fromarray(tex_arr).save(tex_path)
                        print(f"    Saved: {tex_path}")

                    # Check metallic roughness texture
                    if hasattr(mat, 'metallicRoughnessTexture') and mat.metallicRoughnessTexture is not None:
                        mr_tex = mat.metallicRoughnessTexture
                        mr_arr = np.array(mr_tex)
                        print(f"    Metallic/Roughness texture: {mr_arr.shape}")

        else:
            print(f"  Single mesh: {type(scene)}")
            print(f"    Vertices: {len(scene.vertices)}")
            print(f"    Faces: {len(scene.faces)}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
