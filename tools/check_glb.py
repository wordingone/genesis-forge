"""Check GLB file for PBR materials."""
import trimesh
import sys

glb_path = sys.argv[1] if len(sys.argv) > 1 else "outputs/0bfa118d-2e1d-475b-826b-43b7ca54dfb2_model.glb"

print(f"Checking: {glb_path}")
scene = trimesh.load(glb_path)

if isinstance(scene, trimesh.Scene):
    for name, geom in scene.geometry.items():
        print(f"\nGeometry: {name}")
        print(f"  Vertices: {len(geom.vertices)}")
        print(f"  Faces: {len(geom.faces)}")
        if hasattr(geom, 'visual'):
            visual = geom.visual
            print(f"  Visual type: {type(visual).__name__}")
            if hasattr(visual, 'material'):
                mat = visual.material
                print(f"  Material type: {type(mat).__name__}")
                if hasattr(mat, 'baseColorTexture') and mat.baseColorTexture is not None:
                    print(f"    baseColorTexture: {mat.baseColorTexture.size if hasattr(mat.baseColorTexture, 'size') else 'present'}")
                if hasattr(mat, 'metallicRoughnessTexture') and mat.metallicRoughnessTexture is not None:
                    print(f"    metallicRoughnessTexture: {mat.metallicRoughnessTexture.size if hasattr(mat.metallicRoughnessTexture, 'size') else 'present'}")
                if hasattr(mat, 'normalTexture') and mat.normalTexture is not None:
                    print(f"    normalTexture: {mat.normalTexture.size if hasattr(mat.normalTexture, 'size') else 'present'}")
else:
    print(f"Mesh type: {type(scene).__name__}")
    print(f"Vertices: {len(scene.vertices)}")
    print(f"Faces: {len(scene.faces)}")
    if hasattr(scene, 'visual'):
        visual = scene.visual
        print(f"Visual type: {type(visual).__name__}")
        if hasattr(visual, 'material'):
            mat = visual.material
            print(f"Material type: {type(mat).__name__}")
            for attr in ['baseColorTexture', 'metallicRoughnessTexture', 'normalTexture', 'occlusionTexture', 'emissiveTexture']:
                if hasattr(mat, attr) and getattr(mat, attr) is not None:
                    tex = getattr(mat, attr)
                    print(f"  {attr}: {tex.size if hasattr(tex, 'size') else 'present'}")
