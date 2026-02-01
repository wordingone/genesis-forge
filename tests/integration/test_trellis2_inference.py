"""
TRELLIS.2 End-to-End Inference Test

Tests the complete TRELLIS.2 Image-to-3D pipeline with aligned official implementation.
Uses official GPU memory optimization: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"""
import os
import sys
import time

# === GPU Memory Optimization (MUST be before torch import) ===
# Official TRELLIS.2 pattern: expandable_segments saves GPU memory
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Set up environment
os.environ['HF_HOME'] = 'B:/M/ArtificialArchitecture/spatial/trellis-forge/models/hub'
os.environ['TORCH_HOME'] = 'B:/M/ArtificialArchitecture/spatial/trellis-forge/models/torch_hub'
os.environ['SPARSE_CONV_BACKEND'] = 'spconv'  # Use spconv with flex_gemm-compatible weight storage
os.environ['SPARSE_ATTN_BACKEND'] = 'sdpa'    # Windows compatibility (no flash_attn)

# Add trellis-forge to path
sys.path.insert(0, 'B:/M/ArtificialArchitecture/spatial/trellis-forge')

import torch
from PIL import Image
# Import from trellis2 (completely separate from TRELLIS 1)
from trellis2.pipelines import Trellis2ImageTo3DPipeline


def print_gpu_stats(label: str):
    """Print GPU memory statistics."""
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"[GPU {label}] Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")


def main():
    print("=" * 70)
    print("TRELLIS.2 End-to-End Inference Test")
    print("=" * 70)

    # Check CUDA
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return

    device = torch.device('cuda:0')
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"VRAM Total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"CUDA Alloc Config: {os.environ.get('PYTORCH_CUDA_ALLOC_CONF', 'default')}")
    print()

    # Load test image
    image_path = 'B:/M/ArtificialArchitecture/spatial/trellis-forge/outputs/pbr_test/input_thumbnail.jpg'
    print(f"Loading image: {image_path}")
    image = Image.open(image_path)
    print(f"Image size: {image.size}")
    print()

    # Load pipeline
    model_path = 'B:/M/ArtificialArchitecture/spatial/trellis-forge/models/hub/TRELLIS.2-4B'
    print(f"Loading pipeline from: {model_path}")
    t0 = time.time()

    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(model_path)
    pipeline.to(device)

    t1 = time.time()
    print(f"Pipeline loaded in {t1-t0:.1f}s")
    print(f"Pipeline type: {pipeline.default_pipeline_type}")
    print(f"Low VRAM mode: {pipeline.low_vram}")
    print_gpu_stats("after load")
    print()

    # Run inference
    print("Starting inference...")
    print("-" * 70)

    t2 = time.time()

    try:
        # Use 512 resolution first for testing, then try cascade
        # max_num_tokens=32768 is more conservative than default 49152
        results = pipeline.run(
            image=image,
            num_samples=1,
            seed=42,
            preprocess_image=True,
            pipeline_type='512',  # Start with 512 for memory testing
            max_num_tokens=32768,  # Conservative token limit
        )

        t3 = time.time()
        print("-" * 70)
        print(f"Inference completed in {t3-t2:.1f}s")
        print_gpu_stats("after inference")
        print()

        # Analyze results
        print("Results:")
        if isinstance(results, list):
            print(f"  Output count: {len(results)}")
            for i, mesh in enumerate(results):
                print(f"  Mesh {i}:")
                if hasattr(mesh, 'vertices'):
                    print(f"    Vertices: {mesh.vertices.shape}")
                if hasattr(mesh, 'faces'):
                    print(f"    Faces: {mesh.faces.shape}")
                if hasattr(mesh, 'coords'):
                    print(f"    Voxel coords: {mesh.coords.shape}")
                if hasattr(mesh, 'attrs'):
                    print(f"    Voxel attrs: {mesh.attrs.shape}")
                if hasattr(mesh, 'layout'):
                    print(f"    PBR layout: {mesh.layout}")

                # Simplify mesh if needed (official pattern)
                if hasattr(mesh, 'simplify'):
                    print("    Simplifying mesh...")
                    mesh.simplify(16777216)  # nvdiffrast limit
        else:
            print(f"  Result type: {type(results)}")
            print(f"  Result: {results}")

        # Export GLB if possible
        output_dir = 'B:/M/ArtificialArchitecture/spatial/trellis-forge/outputs/trellis2_test'
        os.makedirs(output_dir, exist_ok=True)

        if isinstance(results, list) and len(results) > 0:
            mesh = results[0]
            if hasattr(mesh, 'to_glb'):
                glb_path = os.path.join(output_dir, 'output.glb')
                mesh.to_glb(glb_path)
                print(f"  Exported to: {glb_path}")

        print()
        print("=" * 70)
        print("TEST PASSED")
        print("=" * 70)

    except Exception as e:
        t3 = time.time()
        print("-" * 70)
        print(f"ERROR after {t3-t2:.1f}s: {e}")
        print_gpu_stats("at error")
        import traceback
        traceback.print_exc()
        print()
        print("=" * 70)
        print("TEST FAILED")
        print("=" * 70)

    # Memory cleanup
    print()
    print("Cleaning up...")
    del pipeline
    torch.cuda.empty_cache()
    print_gpu_stats("after cleanup")


if __name__ == '__main__':
    main()
