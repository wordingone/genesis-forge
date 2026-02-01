"""Extract textures from GLB files for comparison."""
import sys
from pathlib import Path

import trimesh
import numpy as np
from PIL import Image

def extract_textures(glb_path, output_dir):
    """Extract all textures from a GLB file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scene = trimesh.load(glb_path)

    if isinstance(scene, trimesh.Scene):
        for name, geom in scene.geometry.items():
            print(f"Geometry: {name}")
            if hasattr(geom, 'visual') and hasattr(geom.visual, 'material'):
                mat = geom.visual.material
                print(f"  Material type: {type(mat)}")

                if hasattr(mat, 'baseColorTexture') and mat.baseColorTexture is not None:
                    img = mat.baseColorTexture
                    if hasattr(img, 'save'):
                        path = output_dir / f"{name}_baseColor.png"
                        img.save(path)
                        print(f"  Saved: {path}")
                    else:
                        # It might be a numpy array
                        img_array = np.array(img)
                        path = output_dir / f"{name}_baseColor.png"
                        Image.fromarray(img_array).save(path)
                        print(f"  Saved: {path}")

                if hasattr(mat, 'metallicRoughnessTexture') and mat.metallicRoughnessTexture is not None:
                    img = mat.metallicRoughnessTexture
                    if hasattr(img, 'save'):
                        path = output_dir / f"{name}_metallicRoughness.png"
                        img.save(path)
                        print(f"  Saved: {path}")
    else:
        print(f"Single mesh: {type(scene)}")
        if hasattr(scene, 'visual') and hasattr(scene.visual, 'material'):
            mat = scene.visual.material
            if hasattr(mat, 'baseColorTexture') and mat.baseColorTexture is not None:
                path = output_dir / "baseColor.png"
                mat.baseColorTexture.save(path)
                print(f"Saved: {path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python extract_glb_texture.py <glb_path> <output_dir>")
        sys.exit(1)

    glb_path = sys.argv[1]
    output_dir = sys.argv[2]
    extract_textures(glb_path, output_dir)
