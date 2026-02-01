"""Quick test of CuMesh read() after simplify."""
import os
import sys
sys.path.insert(0, '.')
os.environ['HF_HOME'] = os.path.abspath('./models')
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import torch
import cumesh

print("Creating test mesh...")
# Simple cube mesh
vertices = torch.tensor([
    [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
    [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]
], dtype=torch.float32).cuda()

faces = torch.tensor([
    [0, 1, 2], [0, 2, 3],  # bottom
    [4, 6, 5], [4, 7, 6],  # top
    [0, 4, 1], [1, 4, 5],  # front
    [2, 6, 3], [3, 6, 7],  # back
    [0, 3, 4], [4, 3, 7],  # left
    [1, 5, 2], [2, 5, 6],  # right
], dtype=torch.int32).cuda()

print(f"Original: {vertices.shape[0]} vertices, {faces.shape[0]} faces")

print("Creating CuMesh...")
mesh = cumesh.CuMesh()
mesh.init(vertices, faces)

print("Simplifying to 6 faces...")
mesh.simplify(6, verbose=True)

print("Reading simplified mesh...")
sys.stdout.flush()
simp_v, simp_f = mesh.read()
print(f"Simplified: {simp_v.shape[0]} vertices, {simp_f.shape[0]} faces")

print("SUCCESS: CuMesh read() works!")
