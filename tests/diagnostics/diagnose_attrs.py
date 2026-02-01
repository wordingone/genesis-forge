"""Diagnose O-Voxel attribute values to understand why textures are washed out."""
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from pathlib import Path
TRELLIS_ROOT = Path(__file__).parent
MODELS_DIR = TRELLIS_ROOT / "models"
os.environ['HF_HOME'] = str(MODELS_DIR)
os.environ['HF_HUB_CACHE'] = str(MODELS_DIR / "hub")
os.environ['TORCH_HOME'] = str(MODELS_DIR / "torch_hub")
os.environ['HF_HUB_OFFLINE'] = '1'

import sys
sys.path.insert(0, str(TRELLIS_ROOT))

import torch
import numpy as np
from PIL import Image

# Ensure venv is used
assert "venv311" in sys.executable, f"Wrong Python: {sys.executable}"

def main():
    print("=" * 60)
    print("O-VOXEL ATTRIBUTE DIAGNOSTICS")
    print("=" * 60)

    # Load the pipeline
    print("\n1. Loading pipeline...")
    from trellis2.pipelines.trellis2_image_to_3d import Trellis2ImageTo3DPipeline

    MODEL_PATH = MODELS_DIR / "hub" / "TRELLIS.2-4B"
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(str(MODEL_PATH))

    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    pipeline.configure_vram_mode(vram_gb)
    pipeline.to("cuda")
    print(f"   Pipeline loaded")

    # Load test image
    print("\n2. Loading test image...")
    image_path = TRELLIS_ROOT / "reference" / "spiral_input.png"
    image = Image.open(str(image_path)).convert("RGBA")
    print(f"   Image: {image.size}")

    # Run pipeline with return_latent to get intermediate data
    print("\n3. Running generation (seed 42)...")
    result = pipeline.run(
        image,
        seed=42,
        return_latent=True,
    )

    out_mesh_list, (shape_slat, tex_slat, res) = result
    out_mesh = out_mesh_list[0]

    print(f"\n4. Output mesh analysis:")
    print(f"   Vertices: {out_mesh.vertices.shape}")
    print(f"   Faces: {out_mesh.faces.shape}")
    print(f"   Voxel origin: {out_mesh.origin}")
    print(f"   Voxel size: {out_mesh.voxel_size}")
    print(f"   Resolution: {res}")

    print(f"\n5. Voxel attributes analysis:")
    attrs = out_mesh.attrs
    coords = out_mesh.coords
    print(f"   Attrs shape: {attrs.shape}")
    print(f"   Coords shape: {coords.shape}")
    print(f"   Attrs dtype: {attrs.dtype}")

    print(f"\n6. Attribute range by channel:")
    layout = out_mesh.layout
    print(f"   Layout: {layout}")

    for name, sl in layout.items():
        ch_data = attrs[:, sl]
        print(f"   {name} [{sl}]: min={ch_data.min():.4f}, max={ch_data.max():.4f}, mean={ch_data.mean():.4f}")

    # Detailed base_color analysis
    base_color = attrs[:, layout['base_color']]
    print(f"\n7. Base color (RGB) detailed analysis:")
    print(f"   R: min={base_color[:, 0].min():.4f}, max={base_color[:, 0].max():.4f}, mean={base_color[:, 0].mean():.4f}")
    print(f"   G: min={base_color[:, 1].min():.4f}, max={base_color[:, 1].max():.4f}, mean={base_color[:, 1].mean():.4f}")
    print(f"   B: min={base_color[:, 2].min():.4f}, max={base_color[:, 2].max():.4f}, mean={base_color[:, 2].mean():.4f}")

    # Check color distribution
    r, g, b = base_color[:, 0], base_color[:, 1], base_color[:, 2]
    gray_threshold = 0.1  # Color variation threshold
    is_grayish = ((r - g).abs() < gray_threshold) & ((g - b).abs() < gray_threshold) & ((r - b).abs() < gray_threshold)
    is_bright = (r + g + b) / 3 > 0.5
    is_bright_gray = is_grayish & is_bright

    print(f"\n8. Color distribution:")
    print(f"   Total voxels: {base_color.shape[0]}")
    print(f"   Gray-ish voxels (low color variance): {is_grayish.sum().item()} ({100*is_grayish.sum().item()/base_color.shape[0]:.1f}%)")
    print(f"   Bright voxels (mean > 0.5): {is_bright.sum().item()} ({100*is_bright.sum().item()/base_color.shape[0]:.1f}%)")
    print(f"   Bright gray voxels: {is_bright_gray.sum().item()} ({100*is_bright_gray.sum().item()/base_color.shape[0]:.1f}%)")

    # Check for blue voxels (glass)
    is_blue = (b > r) & (b > g) & (b > 0.4)
    print(f"   Blue-ish voxels (B > R, B > G, B > 0.4): {is_blue.sum().item()} ({100*is_blue.sum().item()/base_color.shape[0]:.1f}%)")

    # Histogram of R, G, B
    print(f"\n9. Color histograms:")
    for i, (ch, name) in enumerate([(r, 'R'), (g, 'G'), (b, 'B')]):
        hist, bins = np.histogram(ch.cpu().numpy(), bins=10, range=(0, 1))
        print(f"   {name}: {list(hist)}")

    # Check shape_slat and tex_slat
    print(f"\n10. Shape SLat analysis:")
    print(f"    Coords: {shape_slat.coords.shape}")
    print(f"    Feats range: [{shape_slat.feats.min():.4f}, {shape_slat.feats.max():.4f}]")
    print(f"    Feats mean: {shape_slat.feats.mean():.4f}")

    print(f"\n11. Texture SLat analysis:")
    print(f"    Coords: {tex_slat.coords.shape}")
    print(f"    Feats range: [{tex_slat.feats.min():.4f}, {tex_slat.feats.max():.4f}]")
    print(f"    Feats mean: {tex_slat.feats.mean():.4f}")

    print("\n" + "=" * 60)
    print("DIAGNOSTICS COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
