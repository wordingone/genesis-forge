import open3d as o3d
import numpy as np
import time

print("Testing Open3D decimation...")

# Create a test mesh similar to TRELLIS output size
N_verts = 1000000  # 1M vertices
N_faces = 2000000  # 2M faces

vertices = np.random.randn(N_verts, 3).astype(np.float64)
faces = np.random.randint(0, N_verts, (N_faces, 3)).astype(np.int32)

mesh = o3d.geometry.TriangleMesh()
mesh.vertices = o3d.utility.Vector3dVector(vertices)
mesh.triangles = o3d.utility.Vector3iVector(faces)
print(f"Original: {len(mesh.vertices)} vertices, {len(mesh.triangles)} faces")

target_faces = int(N_faces * 0.05)  # 5% = 95% reduction
print(f"Target faces: {target_faces}")

start = time.time()
mesh = mesh.simplify_quadric_decimation(target_faces)
elapsed = time.time() - start
print(f"After decimate: {len(mesh.vertices)} vertices, {len(mesh.triangles)} faces")
print(f"Time: {elapsed:.2f}s")
print("SUCCESS")
