#!/usr/bin/env python3
"""
TRELLIS Model Downloader - Pre-cache all models for offline use

This script downloads all required models to local cache so TRELLIS can run
completely offline after initial setup.

Models downloaded:
- microsoft/TRELLIS-text-xlarge (~3.9GB) - Text-to-3D
- microsoft/TRELLIS-image-large (~2.5GB) - Image-to-3D
- facebookresearch/dinov2 (~1GB) - Image encoder for Image-to-3D
- rembg u2net model (~176MB) - Background removal

Usage:
    python scripts/download_models.py [--text-only] [--image-only] [--verify]
"""

import sys
import os
import argparse
from pathlib import Path

# Add trellis to path
script_dir = Path(__file__).parent.absolute()
trellis_root = script_dir.parent
sys.path.insert(0, str(trellis_root))

def download_text_model():
    """Download TRELLIS text-to-3D model"""
    print("\n" + "="*60)
    print("Downloading TRELLIS-text-xlarge (Text-to-3D)")
    print("Size: ~3.9GB")
    print("="*60)

    from trellis.pipelines import TrellisTextTo3DPipeline
    pipeline = TrellisTextTo3DPipeline.from_pretrained("microsoft/TRELLIS-text-xlarge")
    print("[OK] TRELLIS-text-xlarge downloaded successfully!")
    return True

def download_image_model():
    """Download TRELLIS image-to-3D model and DINOv2"""
    print("\n" + "="*60)
    print("Downloading TRELLIS-image-large (Image-to-3D)")
    print("Size: ~2.5GB + DINOv2 (~1GB)")
    print("="*60)

    from trellis.pipelines import TrellisImageTo3DPipeline
    pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
    print("[OK] TRELLIS-image-large downloaded successfully!")
    print("[OK] DINOv2 (image encoder) downloaded via torch.hub!")
    return True

def download_rembg():
    """Download rembg background removal model"""
    print("\n" + "="*60)
    print("Downloading rembg u2net model (Background Removal)")
    print("Size: ~176MB")
    print("="*60)

    import rembg
    # Trigger model download by creating a session
    session = rembg.new_session('u2net')
    print("[OK] rembg u2net model downloaded successfully!")
    return True

def verify_models():
    """Verify all models are cached and loadable offline"""
    print("\n" + "="*60)
    print("Verifying models (offline mode)")
    print("="*60)

    # Set HF to offline mode
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'

    results = {}

    # Check text model
    try:
        from huggingface_hub import HfFileSystem
        fs = HfFileSystem()
        # Check if pipeline.json exists in cache
        from huggingface_hub import hf_hub_download
        hf_hub_download("microsoft/TRELLIS-text-xlarge", "pipeline.json", local_files_only=True)
        results['text'] = True
        print("[OK] TRELLIS-text-xlarge: Cached")
    except Exception as e:
        results['text'] = False
        print(f"[MISSING] TRELLIS-text-xlarge: NOT cached - {e}")

    # Check image model
    try:
        from huggingface_hub import hf_hub_download
        hf_hub_download("microsoft/TRELLIS-image-large", "pipeline.json", local_files_only=True)
        results['image'] = True
        print("[OK] TRELLIS-image-large: Cached")
    except Exception as e:
        results['image'] = False
        print(f"[MISSING] TRELLIS-image-large: NOT cached - {e}")

    # Check DINOv2 (torch.hub)
    try:
        import torch
        torch.hub.set_dir(os.path.expanduser("~/.cache/torch/hub"))
        # Check if model exists in cache
        dinov2_path = Path(os.path.expanduser("~/.cache/torch/hub/facebookresearch_dinov2_main"))
        if dinov2_path.exists():
            results['dinov2'] = True
            print("[OK] DINOv2 (torch.hub): Cached")
        else:
            results['dinov2'] = False
            print("[MISSING] DINOv2 (torch.hub): NOT cached")
    except Exception as e:
        results['dinov2'] = False
        print(f"[MISSING] DINOv2 (torch.hub): Error - {e}")

    # Check rembg
    try:
        rembg_path = Path(os.path.expanduser("~/.u2net/u2net.onnx"))
        if rembg_path.exists():
            results['rembg'] = True
            print("[OK] rembg u2net: Cached")
        else:
            results['rembg'] = False
            print("[MISSING] rembg u2net: NOT cached")
    except Exception as e:
        results['rembg'] = False
        print(f"[MISSING] rembg u2net: Error - {e}")

    return results

def main():
    parser = argparse.ArgumentParser(description="Download TRELLIS models for offline use")
    parser.add_argument('--text-only', action='store_true', help='Only download text-to-3D model')
    parser.add_argument('--image-only', action='store_true', help='Only download image-to-3D model')
    parser.add_argument('--verify', action='store_true', help='Verify models are cached (no download)')
    args = parser.parse_args()

    print("="*60)
    print("TRELLIS Model Downloader")
    print("="*60)
    print(f"Cache locations:")
    print(f"  HuggingFace: ~/.cache/huggingface/hub/")
    print(f"  Torch Hub:   ~/.cache/torch/hub/")
    print(f"  rembg:       ~/.u2net/")

    if args.verify:
        results = verify_models()
        all_cached = all(results.values())
        print("\n" + "="*60)
        if all_cached:
            print("[OK] All models cached - TRELLIS can run offline!")
        else:
            print("[MISSING] Some models missing - run without --verify to download")
        return 0 if all_cached else 1

    success = True

    if args.text_only:
        success = download_text_model() and success
    elif args.image_only:
        success = download_image_model() and success
        success = download_rembg() and success
    else:
        # Download everything
        success = download_text_model() and success
        success = download_image_model() and success
        success = download_rembg() and success

    print("\n" + "="*60)
    if success:
        print("[OK] All models downloaded successfully!")
        print("TRELLIS can now run fully offline.")
    else:
        print("[MISSING] Some downloads failed. Check errors above.")
    print("="*60)

    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
