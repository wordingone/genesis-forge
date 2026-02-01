"""
Test if different seeds produce different brightness levels.
If brightness varies significantly with seed, the difference from official
is just seed variance, not a bug.
"""
import os
import sys
from pathlib import Path

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

TRELLIS_ROOT = Path(__file__).parent
MODELS_DIR = TRELLIS_ROOT / "models"
os.environ['HF_HOME'] = str(MODELS_DIR)
os.environ['HF_HUB_CACHE'] = str(MODELS_DIR / "hub")
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

sys.path.insert(0, str(TRELLIS_ROOT))
o_voxel_parent = TRELLIS_ROOT / "o_voxel"
if o_voxel_parent.exists():
    sys.path.insert(0, str(o_voxel_parent))

import torch
import numpy as np
from PIL import Image

assert "venv311" in sys.executable, f"Wrong Python: {sys.executable}"


def test_seed(pipeline, preprocessed, seed):
    """Run inference with a specific seed and return color stats."""
    SPARSE_PARAMS = {"steps": 12, "guidance_strength": 7.5, "guidance_rescale": 0.7, "rescale_t": 5.0}
    SHAPE_PARAMS = {"steps": 12, "guidance_strength": 7.5, "guidance_rescale": 0.5, "rescale_t": 3.0}
    TEX_PARAMS = {"steps": 12, "guidance_strength": 1.0, "guidance_rescale": 0.0, "rescale_t": 3.0}

    torch.manual_seed(seed)

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
    attrs = mesh.attrs.cpu().numpy()
    colors = attrs[:, :3] * 255

    return {
        'seed': seed,
        'r_mean': colors[:, 0].mean(),
        'g_mean': colors[:, 1].mean(),
        'b_mean': colors[:, 2].mean(),
        'brightness': colors.mean(),
    }


def main():
    print("="*70)
    print("SEED VARIATION TEST")
    print("="*70)

    from trellis2.pipelines.trellis2_image_to_3d import Trellis2ImageTo3DPipeline
    MODEL_PATH = MODELS_DIR / "hub" / "TRELLIS.2-4B"

    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(str(MODEL_PATH))
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    pipeline.configure_vram_mode(vram_gb)
    pipeline.to("cuda")

    input_path = TRELLIS_ROOT / "reference" / "spiral_input.png"
    if not input_path.exists():
        input_path = TRELLIS_ROOT / "reference" / "test_input.jpg.JPG"

    raw_image = Image.open(str(input_path)).convert("RGBA")
    preprocessed = pipeline.preprocess_image(raw_image)

    # Test multiple seeds
    seeds = [0, 42, 123, 456, 789]
    results = []

    for seed in seeds:
        print(f"\nTesting seed {seed}...")
        result = test_seed(pipeline, preprocessed, seed)
        results.append(result)
        print(f"  Brightness: {result['brightness']:.1f} (R={result['r_mean']:.1f}, G={result['g_mean']:.1f}, B={result['b_mean']:.1f})")

    print(f"\n{'='*70}")
    print("SEED VARIANCE ANALYSIS")
    print(f"{'='*70}")

    brightnesses = [r['brightness'] for r in results]
    print(f"Brightness range: {min(brightnesses):.1f} - {max(brightnesses):.1f}")
    print(f"Brightness std: {np.std(brightnesses):.1f}")
    print(f"Brightness mean: {np.mean(brightnesses):.1f}")

    print(f"\nOfficial GLB brightness: ~124 (R=109, G=122, B=142)")

    # Check if any seed gets close to official
    for r in results:
        diff = abs(r['brightness'] - 124)
        print(f"  Seed {r['seed']}: brightness={r['brightness']:.1f}, diff from official={diff:.1f}")

    # Conclusion
    if np.std(brightnesses) > 15:
        print(f"\n  HIGH VARIANCE: Brightness varies significantly with seed ({np.std(brightnesses):.1f} std)")
        print(f"  The difference from official is likely just seed variance.")
    else:
        print(f"\n  LOW VARIANCE: Brightness is consistent across seeds ({np.std(brightnesses):.1f} std)")
        print(f"  If all seeds are brighter than official, there may be a systematic issue.")


if __name__ == "__main__":
    main()
