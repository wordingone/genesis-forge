"""
Diagnose color difference between our output and official TRELLIS.2.

Tests hypothesis: Is the brightness difference in:
1. Model inference (attrs coming out brighter)?
2. Grid sampling (trilinear interpolation brightening)?
3. Texture baking (post-processing)?

This script dumps intermediate values at each stage.
"""
import os
import sys
from pathlib import Path

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

TRELLIS_ROOT = Path(__file__).parent
MODELS_DIR = TRELLIS_ROOT / "models"
os.environ['HF_HOME'] = str(MODELS_DIR)
os.environ['HF_HUB_CACHE'] = str(MODELS_DIR / "hub")
os.environ['HUGGINGFACE_HUB_CACHE'] = str(MODELS_DIR / "hub")
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

sys.path.insert(0, str(TRELLIS_ROOT))
o_voxel_parent = TRELLIS_ROOT / "o_voxel"
if o_voxel_parent.exists():
    sys.path.insert(0, str(o_voxel_parent))

import torch
import numpy as np
from PIL import Image
import trimesh

# Ensure correct venv
assert "venv311" in sys.executable, f"Wrong Python: {sys.executable}"


def analyze_official_glb():
    """Extract and analyze texture from official GLB."""
    official_path = TRELLIS_ROOT / "reference" / "spiral_official.glb"
    if not official_path.exists():
        print(f"Official GLB not found: {official_path}")
        return None

    print(f"\n{'='*70}")
    print("OFFICIAL GLB TEXTURE ANALYSIS")
    print(f"{'='*70}")

    scene = trimesh.load(str(official_path))

    if isinstance(scene, trimesh.Scene):
        for gname, geom in scene.geometry.items():
            if hasattr(geom, 'visual') and hasattr(geom.visual, 'material'):
                mat = geom.visual.material
                if hasattr(mat, 'baseColorTexture') and mat.baseColorTexture is not None:
                    tex = np.array(mat.baseColorTexture)
                    print(f"\nTexture shape: {tex.shape}")

                    # Sample non-black pixels
                    if tex.shape[-1] >= 3:
                        rgb = tex[..., :3]
                        # Mask for non-black pixels (any channel > 10)
                        valid = rgb.max(axis=-1) > 10
                        valid_rgb = rgb[valid]

                        print(f"Valid pixels: {valid_rgb.shape[0]}")
                        print(f"R: min={valid_rgb[:, 0].min()}, max={valid_rgb[:, 0].max()}, mean={valid_rgb[:, 0].mean():.1f}")
                        print(f"G: min={valid_rgb[:, 1].min()}, max={valid_rgb[:, 1].max()}, mean={valid_rgb[:, 1].mean():.1f}")
                        print(f"B: min={valid_rgb[:, 2].min()}, max={valid_rgb[:, 2].max()}, mean={valid_rgb[:, 2].mean():.1f}")

                        # Histogram
                        print(f"\nBrightness distribution:")
                        brightness = valid_rgb.mean(axis=-1)
                        for thresh in [0, 50, 100, 150, 200, 255]:
                            count = (brightness <= thresh).sum()
                            pct = 100 * count / len(brightness)
                            print(f"  <= {thresh}: {pct:.1f}%")

                        return {'r_mean': valid_rgb[:, 0].mean(),
                                'g_mean': valid_rgb[:, 1].mean(),
                                'b_mean': valid_rgb[:, 2].mean()}
    return None


def analyze_raw_attrs(mesh):
    """Analyze raw attribute values from mesh before any processing."""
    print(f"\n{'='*70}")
    print("RAW ATTRIBUTE ANALYSIS (before grid_sample)")
    print(f"{'='*70}")

    attrs = mesh.attrs.cpu().numpy()
    print(f"Attribute shape: {attrs.shape}")  # (N_voxels, 6) - RGB, Metallic, Roughness, Alpha

    # Extract color channels (first 3)
    colors = attrs[:, :3]  # These are in [0, 1] range (decoded via *0.5+0.5)

    print(f"\nRaw color values (should be 0-1 range):")
    print(f"  R: min={colors[:, 0].min():.4f}, max={colors[:, 0].max():.4f}, mean={colors[:, 0].mean():.4f}")
    print(f"  G: min={colors[:, 1].min():.4f}, max={colors[:, 1].max():.4f}, mean={colors[:, 1].mean():.4f}")
    print(f"  B: min={colors[:, 2].min():.4f}, max={colors[:, 2].max():.4f}, mean={colors[:, 2].mean():.4f}")

    # Convert to 0-255 for comparison
    colors_255 = colors * 255
    print(f"\nAs 0-255:")
    print(f"  R: mean={colors_255[:, 0].mean():.1f}")
    print(f"  G: mean={colors_255[:, 1].mean():.1f}")
    print(f"  B: mean={colors_255[:, 2].mean():.1f}")

    # Histogram of brightness
    brightness = colors.mean(axis=-1)
    print(f"\nBrightness (0-1) distribution:")
    for thresh in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        count = (brightness <= thresh).sum()
        pct = 100 * count / len(brightness)
        print(f"  <= {thresh}: {pct:.1f}%")

    return colors.mean(axis=0) * 255


def main():
    print("="*70)
    print("COLOR DIFFERENCE DIAGNOSIS")
    print("="*70)

    # First, analyze official GLB
    official_stats = analyze_official_glb()

    # Now generate our mesh
    print(f"\n{'='*70}")
    print("GENERATING OUR MESH")
    print(f"{'='*70}")

    from trellis2.pipelines.trellis2_image_to_3d import Trellis2ImageTo3DPipeline
    MODEL_PATH = MODELS_DIR / "hub" / "TRELLIS.2-4B"

    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(str(MODEL_PATH))
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    pipeline.configure_vram_mode(vram_gb)
    pipeline.to("cuda")

    # Use spiral input for comparison with official
    input_path = TRELLIS_ROOT / "reference" / "spiral_input.png"
    if not input_path.exists():
        input_path = TRELLIS_ROOT / "reference" / "test_input.jpg.JPG"

    print(f"Input: {input_path}")

    raw_image = Image.open(str(input_path)).convert("RGBA")
    preprocessed = pipeline.preprocess_image(raw_image)

    SEED = 42
    torch.manual_seed(SEED)

    SPARSE_PARAMS = {"steps": 12, "guidance_strength": 7.5, "guidance_rescale": 0.7, "rescale_t": 5.0}
    SHAPE_PARAMS = {"steps": 12, "guidance_strength": 7.5, "guidance_rescale": 0.5, "rescale_t": 3.0}
    TEX_PARAMS = {"steps": 12, "guidance_strength": 1.0, "guidance_rescale": 0.0, "rescale_t": 3.0}

    print(f"\nAutocast enabled: {torch.is_autocast_enabled()} (should be False)")

    cond_512 = pipeline.get_cond([preprocessed], 512)
    cond_1024 = pipeline.get_cond([preprocessed], 1024)

    coords = pipeline.sample_sparse_structure(cond_512, 32, num_samples=1, sampler_params=SPARSE_PARAMS)
    shape_slat, hr_resolution = pipeline.sample_shape_slat_cascade(
        cond_512, cond_1024,
        pipeline.models['shape_slat_flow_model_512'],
        pipeline.models['shape_slat_flow_model_1024'],
        512, 1024, coords, sampler_params=SHAPE_PARAMS,
    )
    tex_slat = pipeline.sample_tex_slat(
        cond_1024, pipeline.models['tex_slat_flow_model_1024'],
        shape_slat, sampler_params=TEX_PARAMS,
    )

    torch.cuda.empty_cache()

    out_meshes = pipeline.decode_latent(shape_slat, tex_slat, hr_resolution)
    mesh = out_meshes[0]

    # Analyze raw attributes
    our_colors = analyze_raw_attrs(mesh)

    # Compare
    if official_stats:
        print(f"\n{'='*70}")
        print("COMPARISON")
        print(f"{'='*70}")
        print(f"           {'Ours':>12} {'Official':>12} {'Diff':>12}")
        print(f"  R mean:  {our_colors[0]:>12.1f} {official_stats['r_mean']:>12.1f} {our_colors[0] - official_stats['r_mean']:>+12.1f}")
        print(f"  G mean:  {our_colors[1]:>12.1f} {official_stats['g_mean']:>12.1f} {our_colors[1] - official_stats['g_mean']:>+12.1f}")
        print(f"  B mean:  {our_colors[2]:>12.1f} {official_stats['b_mean']:>12.1f} {our_colors[2] - official_stats['b_mean']:>+12.1f}")

        # The key insight: if our RAW attrs are already brighter, the problem is in model inference
        # If our raw attrs match but output is brighter, the problem is in grid_sample/texture baking

        raw_avg = our_colors.mean()
        official_avg = (official_stats['r_mean'] + official_stats['g_mean'] + official_stats['b_mean']) / 3

        if raw_avg > official_avg + 20:
            print(f"\n  DIAGNOSIS: RAW attributes are already {raw_avg - official_avg:.1f} points brighter.")
            print(f"  The problem is likely in MODEL INFERENCE, not in texture baking.")
            print(f"  Check: different random seed? different model weights? platform-specific behavior?")
        else:
            print(f"\n  DIAGNOSIS: RAW attributes are similar (diff={raw_avg - official_avg:.1f}).")
            print(f"  If final texture is still brighter, problem is in grid_sample/texture baking.")


if __name__ == "__main__":
    main()
