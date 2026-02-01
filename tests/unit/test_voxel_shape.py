import os
import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'o_voxel')

# Set environment before any other imports
os.environ['HF_HOME'] = os.path.abspath('./models')
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import torch
from trellis2.pipelines.trellis2_image_to_3d import Trellis2ImageTo3DPipeline
from PIL import Image

print("Loading pipeline...")
pipe = Trellis2ImageTo3DPipeline.from_pretrained('./models/hub/TRELLIS.2-4B')
pipe.cuda()

print("Loading image...")
img = Image.open("reference/test_input.jpg.JPG").resize((256, 256))

print("Running quick generation...")
output = pipe.run(img, seed=42)

mesh = output[0]
print(f"Voxel shape: {mesh.voxel_shape}")
print(f"Coords shape: {mesh.coords.shape}")
print(f"Attrs shape: {mesh.attrs.shape}")
print(f"Max coord: {mesh.coords.max().item()}")

# Test to_dense_voxel_grid memory requirements
if len(mesh.voxel_shape) >= 4:
    X_size, Y_size, Z_size = mesh.voxel_shape[-3], mesh.voxel_shape[-2], mesh.voxel_shape[-1]
else:
    X_size = Y_size = Z_size = int(mesh.coords.max().item()) + 1

C = mesh.attrs.shape[1]
mem_required = 1 * C * X_size * Y_size * Z_size * 4 / 1e9  # 4 bytes per float32
print(f"Dense grid dimensions: [1, {C}, {X_size}, {Y_size}, {Z_size}]")
print(f"Dense grid memory required: {mem_required:.2f} GB")
