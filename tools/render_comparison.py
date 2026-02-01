"""Render GLB models for visual comparison."""
import trimesh
import numpy as np
from PIL import Image
import sys

def render_glb(glb_path, output_path, resolution=(800, 800)):
    """Render a GLB file to an image."""
    print(f"Loading: {glb_path}")
    scene = trimesh.load(glb_path)

    if isinstance(scene, trimesh.Scene):
        # Get the scene as a single mesh for rendering
        print(f"  Scene with {len(scene.geometry)} geometries")
        for name, geom in scene.geometry.items():
            print(f"    {name}: {len(geom.vertices)} vertices, {len(geom.faces)} faces")

    # Use trimesh's built-in scene rendering
    try:
        # Create a scene if we have a single mesh
        if isinstance(scene, trimesh.Trimesh):
            scene = trimesh.Scene(scene)

        # Try to render with pyrender if available
        try:
            import pyrender

            # Convert trimesh scene to pyrender scene
            pr_scene = pyrender.Scene.from_trimesh_scene(scene)

            # Add camera
            camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)

            # Position camera to see the model
            bounds = scene.bounds
            center = (bounds[0] + bounds[1]) / 2
            size = np.linalg.norm(bounds[1] - bounds[0])

            camera_pose = np.eye(4)
            camera_pose[:3, 3] = center + np.array([0, 0, size * 1.5])
            camera_pose[:3, :3] = trimesh.transformations.rotation_matrix(
                np.pi, [1, 0, 0])[:3, :3]

            pr_scene.add(camera, pose=camera_pose)

            # Add light
            light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
            pr_scene.add(light, pose=camera_pose)

            # Render
            renderer = pyrender.OffscreenRenderer(*resolution)
            color, _ = renderer.render(pr_scene)
            renderer.delete()

            Image.fromarray(color).save(output_path)
            print(f"  Saved: {output_path}")
            return True

        except ImportError:
            print("  pyrender not available, using trimesh scene export")

            # Export scene as PNG using trimesh's save_image
            png_data = scene.save_image(resolution=resolution)
            with open(output_path, 'wb') as f:
                f.write(png_data)
            print(f"  Saved: {output_path}")
            return True

    except Exception as e:
        print(f"  Render failed: {e}")

        # Fallback: just show mesh statistics
        print("  Creating info image instead...")
        img = Image.new('RGB', resolution, (40, 40, 40))
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)

        if isinstance(scene, trimesh.Scene):
            total_verts = sum(g.vertices.shape[0] for g in scene.geometry.values())
            total_faces = sum(g.faces.shape[0] for g in scene.geometry.values())
        else:
            total_verts = scene.vertices.shape[0]
            total_faces = scene.faces.shape[0]

        text = f"GLB Model\nVertices: {total_verts:,}\nFaces: {total_faces:,}"
        draw.text((20, 20), text, fill=(255, 255, 255))
        img.save(output_path)
        print(f"  Saved info image: {output_path}")
        return False

if __name__ == "__main__":
    our_model = "outputs/6e1837d4-a484-4e01-bcfc-ae7aac1104b4_model.glb"
    benchmark = "reference/official_output.glb"

    render_glb(our_model, "outputs/our_model_render.png")
    render_glb(benchmark, "outputs/benchmark_render.png")

    print("\nDone! Check outputs/our_model_render.png and outputs/benchmark_render.png")
