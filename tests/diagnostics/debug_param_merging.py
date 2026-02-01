"""
Debug script to compare parameter merging between test script path and GUI path.
Verifies that guidance_interval is correctly preserved in both paths.
"""
import sys
import os
sys.path.insert(0, r"B:\M\ArtificialArchitecture\spatial\trellis-forge")

import torch
from PIL import Image
from trellis2.pipelines import Trellis2ImageTo3DPipeline

def main():
    print("=" * 80)
    print("PARAMETER MERGING DEBUG")
    print("=" * 80)

    # Load pipeline
    model_path = r"B:\M\ArtificialArchitecture\spatial\trellis-forge\models\hub\TRELLIS.2-4B"
    print(f"\nLoading pipeline from: {model_path}")
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(model_path)

    # Print internal defaults from pipeline.json
    print("\n" + "=" * 80)
    print("INTERNAL DEFAULTS (from pipeline.json)")
    print("=" * 80)

    print("\n1. Sparse Structure Sampler Params:")
    for k, v in pipeline.sparse_structure_sampler_params.items():
        print(f"   {k}: {v}")

    print("\n2. Shape SLat Sampler Params:")
    for k, v in pipeline.shape_slat_sampler_params.items():
        print(f"   {k}: {v}")

    print("\n3. Tex SLat Sampler Params:")
    for k, v in pipeline.tex_slat_sampler_params.items():
        print(f"   {k}: {v}")

    # Simulate test script path (no explicit params)
    print("\n" + "=" * 80)
    print("TEST SCRIPT PATH (no explicit params)")
    print("=" * 80)

    test_script_sparse_params = {}
    test_script_shape_params = {}
    test_script_tex_params = {}

    # Simulate merge (same as in sample_sparse_structure, etc.)
    merged_sparse_test = {**pipeline.sparse_structure_sampler_params, **test_script_sparse_params}
    merged_shape_test = {**pipeline.shape_slat_sampler_params, **test_script_shape_params}
    merged_tex_test = {**pipeline.tex_slat_sampler_params, **test_script_tex_params}

    print("\n1. Merged Sparse Structure Params:")
    for k, v in merged_sparse_test.items():
        print(f"   {k}: {v}")

    print("\n2. Merged Shape Params:")
    for k, v in merged_shape_test.items():
        print(f"   {k}: {v}")

    print("\n3. Merged Tex Params:")
    for k, v in merged_tex_test.items():
        print(f"   {k}: {v}")

    # Simulate GUI path (explicit params, matching pipeline.json values)
    print("\n" + "=" * 80)
    print("GUI PATH (explicit params)")
    print("=" * 80)

    gui_sparse_params = {
        "steps": 12,
        "guidance_strength": 7.5,
        "guidance_rescale": 0.7,
        "rescale_t": 5.0,
        # NOTE: guidance_interval NOT passed
    }
    gui_shape_params = {
        "steps": 12,
        "guidance_strength": 7.5,
        "guidance_rescale": 0.5,
        "rescale_t": 3.0,
        # NOTE: guidance_interval NOT passed
    }
    gui_tex_params = {
        "steps": 12,
        "guidance_strength": 1.0,
        "guidance_rescale": 0.0,
        "rescale_t": 3.0,
        # NOTE: guidance_interval NOT passed
    }

    # Simulate merge
    merged_sparse_gui = {**pipeline.sparse_structure_sampler_params, **gui_sparse_params}
    merged_shape_gui = {**pipeline.shape_slat_sampler_params, **gui_shape_params}
    merged_tex_gui = {**pipeline.tex_slat_sampler_params, **gui_tex_params}

    print("\n1. Merged Sparse Structure Params:")
    for k, v in merged_sparse_gui.items():
        print(f"   {k}: {v}")

    print("\n2. Merged Shape Params:")
    for k, v in merged_shape_gui.items():
        print(f"   {k}: {v}")

    print("\n3. Merged Tex Params:")
    for k, v in merged_tex_gui.items():
        print(f"   {k}: {v}")

    # Compare
    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)

    print("\nSparse Structure Params IDENTICAL:", merged_sparse_test == merged_sparse_gui)
    print("Shape Params IDENTICAL:", merged_shape_test == merged_shape_gui)
    print("Tex Params IDENTICAL:", merged_tex_test == merged_tex_gui)

    if merged_sparse_test != merged_sparse_gui:
        print("\nDIFFERENCES in Sparse Structure:")
        for k in set(merged_sparse_test.keys()) | set(merged_sparse_gui.keys()):
            v1 = merged_sparse_test.get(k, "MISSING")
            v2 = merged_sparse_gui.get(k, "MISSING")
            if v1 != v2:
                print(f"   {k}: test={v1} vs gui={v2}")

    if merged_shape_test != merged_shape_gui:
        print("\nDIFFERENCES in Shape:")
        for k in set(merged_shape_test.keys()) | set(merged_shape_gui.keys()):
            v1 = merged_shape_test.get(k, "MISSING")
            v2 = merged_shape_gui.get(k, "MISSING")
            if v1 != v2:
                print(f"   {k}: test={v1} vs gui={v2}")

    if merged_tex_test != merged_tex_gui:
        print("\nDIFFERENCES in Tex:")
        for k in set(merged_tex_test.keys()) | set(merged_tex_gui.keys()):
            v1 = merged_tex_test.get(k, "MISSING")
            v2 = merged_tex_gui.get(k, "MISSING")
            if v1 != v2:
                print(f"   {k}: test={v1} vs gui={v2}")

    # Check guidance_interval specifically
    print("\n" + "=" * 80)
    print("GUIDANCE_INTERVAL CHECK")
    print("=" * 80)

    print(f"\nSparse guidance_interval in test: {merged_sparse_test.get('guidance_interval', 'MISSING')}")
    print(f"Sparse guidance_interval in GUI:  {merged_sparse_gui.get('guidance_interval', 'MISSING')}")
    print(f"Sparse guidance_interval MATCH:   {merged_sparse_test.get('guidance_interval') == merged_sparse_gui.get('guidance_interval')}")

    print(f"\nShape guidance_interval in test: {merged_shape_test.get('guidance_interval', 'MISSING')}")
    print(f"Shape guidance_interval in GUI:  {merged_shape_gui.get('guidance_interval', 'MISSING')}")
    print(f"Shape guidance_interval MATCH:   {merged_shape_test.get('guidance_interval') == merged_shape_gui.get('guidance_interval')}")

    print(f"\nTex guidance_interval in test: {merged_tex_test.get('guidance_interval', 'MISSING')}")
    print(f"Tex guidance_interval in GUI:  {merged_tex_gui.get('guidance_interval', 'MISSING')}")
    print(f"Tex guidance_interval MATCH:   {merged_tex_test.get('guidance_interval') == merged_tex_gui.get('guidance_interval')}")

    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)

    all_match = (
        merged_sparse_test == merged_sparse_gui and
        merged_shape_test == merged_shape_gui and
        merged_tex_test == merged_tex_gui
    )

    if all_match:
        print("\nAll merged params are IDENTICAL between test script and GUI paths.")
        print("The difference must be elsewhere (random seed, CUDA state, export params).")
    else:
        print("\nParams DIFFER between test script and GUI paths!")
        print("This is likely the cause of the quality difference.")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
