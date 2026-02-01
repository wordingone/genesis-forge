"""
Quick test for TRELLIS.2 pipeline - validates all components work.
"""
import os
import sys
import time

# Set paths before importing anything
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

# Add trellis-forge to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

# CRITICAL: Add o_voxel/o_voxel to path BEFORE trellis-forge so it shadows the namespace package
# This is needed because o_voxel/ folder creates a namespace package that shadows o_voxel/o_voxel/
o_voxel_pkg = os.path.join(script_dir, 'o_voxel', 'o_voxel')
if os.path.isdir(o_voxel_pkg):
    # Remove any existing o_voxel entries
    sys.path = [p for p in sys.path if 'o_voxel' not in p or p == o_voxel_pkg]
    # Insert the actual package path at the front
    sys.path.insert(0, os.path.join(script_dir, 'o_voxel'))

# Debug: print where o_voxel will be imported from
print(f"sys.path[0:3]: {sys.path[0:3]}")

import torch
from PIL import Image

def test_imports():
    """Test all critical imports."""
    print("Testing imports...")

    # Test cumesh
    import cumesh
    assert hasattr(cumesh, 'CuMesh'), "cumesh.CuMesh missing"
    assert hasattr(cumesh, 'cuBVH'), "cumesh.cuBVH missing"
    print("  cumesh: OK")

    # Test nvdiffrast
    import nvdiffrast.torch as dr
    assert hasattr(dr, 'rasterize'), "nvdiffrast.torch.rasterize missing"
    print("  nvdiffrast: OK")

    # Test o_voxel
    import o_voxel
    assert hasattr(o_voxel, 'convert'), "o_voxel.convert missing"
    print("  o_voxel: OK")

    # Test our sparse conv
    from trellis2.modules.sparse.conv import SparseConv3d
    print("  SparseConv3d: OK")

    # Test grid_sample_3d replacement
    from trellis2.utils.grid_sample_3d_torch import grid_sample_3d
    print("  grid_sample_3d: OK")

    print("All imports OK!\n")

def test_pipeline_load():
    """Test pipeline loading."""
    print("Testing pipeline load...")

    from trellis2.pipelines import Trellis2ImageTo3DPipeline

    model_path = os.path.join(os.path.dirname(__file__), 'models', 'trellis2')

    print(f"  Loading from: {model_path}")
    start = time.time()
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(model_path)
    pipeline.to('cuda')
    elapsed = time.time() - start
    print(f"  Pipeline loaded in {elapsed:.1f}s")

    # Check models
    print(f"  Models loaded: {list(pipeline.models.keys())}")

    return pipeline

def test_inference(pipeline, image_path):
    """Test actual inference."""
    print(f"\nTesting inference with: {image_path}")

    # Load image
    image = Image.open(image_path)
    print(f"  Image size: {image.size}")

    # Run pipeline
    print("  Running pipeline (this will take a few minutes)...")
    torch.cuda.empty_cache()

    start = time.time()

    # Monitor GPU memory
    initial_mem = torch.cuda.memory_allocated() / 1024**3
    print(f"  Initial GPU memory: {initial_mem:.2f} GB")

    # Use 512 resolution for faster testing
    results = pipeline.run(
        image,
        seed=42,
        pipeline_type='512',  # Faster for testing
    )

    elapsed = time.time() - start
    peak_mem = torch.cuda.max_memory_allocated() / 1024**3
    print(f"  Pipeline completed in {elapsed:.1f}s")
    print(f"  Peak GPU memory: {peak_mem:.2f} GB")

    # Check results
    assert len(results) > 0, "No results returned"
    mesh = results[0]

    print(f"  Vertices: {mesh.vertices.shape[0]}")
    print(f"  Faces: {mesh.faces.shape[0]}")
    print(f"  Voxel coords: {mesh.coords.shape}")
    print(f"  Attrs: {mesh.attrs.shape}")

    return results

def test_export(results, output_path):
    """Test GLB export."""
    print(f"\nTesting export to: {output_path}")

    mesh = results[0]

    start = time.time()
    mesh.export(
        output_path,
        simplify=0.95,
        texture_size=1024,
        verbose=True
    )
    elapsed = time.time() - start

    print(f"  Export completed in {elapsed:.1f}s")

    # Check file
    size = os.path.getsize(output_path)
    print(f"  File size: {size / 1024:.1f} KB")

    return output_path

if __name__ == '__main__':
    print("=" * 60)
    print("TRELLIS.2 Pipeline Test")
    print("=" * 60)

    # Test imports
    test_imports()

    # Test pipeline load
    pipeline = test_pipeline_load()

    # Test inference
    image_path = os.path.join(os.path.dirname(__file__), 'reference', 'test_input.jpg')
    if os.path.exists(image_path):
        results = test_inference(pipeline, image_path)

        # Test export
        output_path = os.path.join(os.path.dirname(__file__), 'outputs', 'test_output.glb')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        test_export(results, output_path)

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED")
        print("=" * 60)
    else:
        print(f"\nWARNING: Test image not found: {image_path}")
        print("Skipping inference test.")
