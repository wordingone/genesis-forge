"""Test TRELLIS.2 pipeline with texture fix validation and GPU monitoring.

IMPORTANT: Run with venv python:
  cd B:\\M\\ArtificialArchitecture\\spatial\\trellis-forge
  .\\venv311\\Scripts\\python.exe test_remesh.py

Resolution options (same as official TRELLIS.2):
  - '512': Fast, lower quality (~3M voxels)
  - '1024': Medium, high quality (~17M voxels) - RECOMMENDED
  - '1024_cascade': Same as '1024' with cascade
  - '1536_cascade': Highest quality (~27M voxels)
"""
import os
import sys
import time
import argparse
import threading
import csv
import subprocess

# Add local source paths
sys.path.insert(0, 'o_voxel')
sys.path.insert(0, 'cumesh_source')

os.environ['HF_HOME'] = os.path.abspath('./models')
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
# Reduce PyTorch allocator fragmentation for large remeshing operations.
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256'

import torch
import numpy as np
from PIL import Image


class GpuMonitor:
    """Background GPU monitoring via nvidia-smi, logs to CSV."""

    def __init__(self, csv_path: str, interval: float = 1.0):
        self.csv_path = csv_path
        self.interval = interval
        self._stop = threading.Event()
        self._thread = None
        self._start_time = None
        self._stage = "init"
        self._rows = []

    def set_stage(self, stage: str):
        self._stage = stage

    def start(self):
        self._start_time = time.time()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        self._write_csv()

    def _poll(self):
        while not self._stop.is_set():
            try:
                result = subprocess.run(
                    ['nvidia-smi',
                     '--query-gpu=utilization.gpu,memory.used,memory.total',
                     '--format=csv,noheader,nounits'],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    parts = result.stdout.strip().split(',')
                    if len(parts) >= 3:
                        elapsed = time.time() - self._start_time
                        gpu_util = float(parts[0].strip())
                        mem_used = float(parts[1].strip())  # MiB
                        mem_total = float(parts[2].strip())
                        self._rows.append({
                            'elapsed_s': round(elapsed, 1),
                            'stage': self._stage,
                            'gpu_util_pct': gpu_util,
                            'mem_used_mb': mem_used,
                            'mem_total_mb': mem_total,
                            'mem_used_gb': round(mem_used / 1024, 2),
                        })
            except Exception:
                pass
            self._stop.wait(self.interval)

    def _write_csv(self):
        if not self._rows:
            return
        os.makedirs(os.path.dirname(self.csv_path) or '.', exist_ok=True)
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self._rows[0].keys())
            writer.writeheader()
            writer.writerows(self._rows)
        print(f"[MONITOR] GPU log saved to {self.csv_path} ({len(self._rows)} samples)")

    def print_summary(self):
        if not self._rows:
            print("[MONITOR] No GPU data collected")
            return
        utils = [r['gpu_util_pct'] for r in self._rows]
        mems = [r['mem_used_gb'] for r in self._rows]
        print(f"\n{'='*60}")
        print(f"GPU MONITORING SUMMARY")
        print(f"{'='*60}")
        print(f"  Samples collected: {len(self._rows)}")
        print(f"  Duration: {self._rows[-1]['elapsed_s']:.1f}s")
        print(f"  GPU Utilization: avg={np.mean(utils):.1f}%, max={max(utils):.0f}%, min={min(utils):.0f}%")
        print(f"  GPU Memory: avg={np.mean(mems):.2f}GB, peak={max(mems):.2f}GB, min={min(mems):.2f}GB")

        # Per-stage breakdown
        stages = {}
        for r in self._rows:
            s = r['stage']
            if s not in stages:
                stages[s] = {'utils': [], 'mems': [], 'start': r['elapsed_s'], 'end': r['elapsed_s']}
            stages[s]['utils'].append(r['gpu_util_pct'])
            stages[s]['mems'].append(r['mem_used_gb'])
            stages[s]['end'] = r['elapsed_s']

        print(f"\n  Per-stage breakdown:")
        print(f"  {'Stage':<25} {'Duration':>8} {'GPU%':>6} {'MemGB':>8} {'PeakGB':>8}")
        print(f"  {'-'*25} {'-'*8} {'-'*6} {'-'*8} {'-'*8}")
        for stage, data in stages.items():
            dur = data['end'] - data['start']
            print(f"  {stage:<25} {dur:>7.1f}s {np.mean(data['utils']):>5.1f}% {np.mean(data['mems']):>7.2f} {max(data['mems']):>7.2f}")
        print(f"{'='*60}")


def get_gpu_memory():
    """Get current GPU memory usage in GB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3
    return 0


def main():
    parser = argparse.ArgumentParser(description='Test TRELLIS.2 pipeline')
    parser.add_argument('--resolution', type=str, default='1024',
                        choices=['512', '1024', '1024_cascade', '1536_cascade'],
                        help='Pipeline resolution (default: 1024)')
    parser.add_argument('--input', type=str, default='reference/test_input.jpg.JPG',
                        help='Input image path')
    parser.add_argument('--output', type=str, default='outputs/test_output.glb',
                        help='Output GLB path')
    parser.add_argument('--texture-size', type=int, default=4096,
                        help='Texture resolution (default: 4096, matches official TRELLIS.2)')
    parser.add_argument('--decimation', type=int, default=1000000,
                        help='Target face count (default: 1000000)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--gpu-log', type=str, default='outputs/gpu_monitor.csv',
                        help='GPU monitoring CSV output path')
    args = parser.parse_args()

    # Start GPU monitor
    monitor = GpuMonitor(args.gpu_log, interval=1.0)
    monitor.start()

    print("=" * 80)
    print(f"TEST: TRELLIS.2 Pipeline (resolution={args.resolution})")
    print("=" * 80)

    # Map simple resolution to pipeline_type
    pipeline_type = args.resolution
    if pipeline_type == '1024':
        pipeline_type = '1024_cascade'  # Match official default

    print(f"\n[CONFIG]")
    print(f"  Resolution: {args.resolution} -> pipeline_type={pipeline_type}")
    print(f"  Input: {args.input}")
    print(f"  Output: {args.output}")
    print(f"  Texture size: {args.texture_size}")
    print(f"  Decimation target: {args.decimation}")
    print(f"  Seed: {args.seed}")

    # Load pipeline
    monitor.set_stage("load_pipeline")
    from trellis2.pipelines.trellis2_image_to_3d import Trellis2ImageTo3DPipeline
    print("\n[LOADING] Pipeline...")
    t0 = time.time()
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained('./models/hub/TRELLIS.2-4B')
    pipeline.cuda()
    load_time = time.time() - t0
    print(f"[LOADED] Pipeline ready ({load_time:.1f}s)")
    print(f"[MEMORY] GPU: {get_gpu_memory():.2f} GB")

    # Load test image
    if not os.path.exists(args.input):
        print(f"[ERROR] Input image not found: {args.input}")
        monitor.stop()
        return
    image = Image.open(args.input)
    print(f"[INPUT] Image size: {image.size}")

    # Run generation
    monitor.set_stage("generation")
    print(f"\n[RUNNING] Generating mesh (pipeline_type={pipeline_type})...")
    t0 = time.time()
    outputs = pipeline.run(
        image,
        seed=args.seed,
        sparse_structure_sampler_params={
            "steps": 12,
            "guidance_strength": 7.5,
        },
        shape_slat_sampler_params={
            "steps": 12,
            "guidance_strength": 3.0,
        },
        tex_slat_sampler_params={
            "steps": 12,
        },
        pipeline_type=pipeline_type,
        max_num_tokens=50000,
    )
    mesh = outputs[0]
    gen_time = time.time() - t0

    print(f"\n[MESH INFO]")
    print(f"  Generation time: {gen_time:.1f}s")
    print(f"  Vertices: {mesh.vertices.shape}")
    print(f"  Faces: {mesh.faces.shape}")
    print(f"  Coords: {mesh.coords.shape}")
    print(f"  Attrs: {mesh.attrs.shape}")
    print(f"  Voxel shape: {mesh.voxel_shape}")
    print(f"[MEMORY] GPU: {get_gpu_memory():.2f} GB (peak: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB)")

    # Export
    monitor.set_stage("export")
    os.makedirs(os.path.dirname(args.output) or 'outputs', exist_ok=True)
    print(f"\n[EXPORTING] Writing {args.output}...")
    t0 = time.time()
    mesh.export(
        args.output,
        decimation_target=args.decimation,
        texture_size=args.texture_size,
        remesh=True,
        remesh_band=1.0,
        remesh_project=0.0,
        verbose=True
    )
    export_time = time.time() - t0

    # Stop monitor
    monitor.stop()

    # Check output
    if os.path.exists(args.output):
        size_mb = os.path.getsize(args.output) / 1024**2
        print(f"\n[SUCCESS] Output: {args.output} ({size_mb:.2f} MB)")
    else:
        print(f"\n[ERROR] Output file not created!")
        monitor.print_summary()
        return

    print(f"\n[TIMING SUMMARY]")
    print(f"  Pipeline load: {load_time:.1f}s")
    print(f"  Generation: {gen_time:.1f}s")
    print(f"  Export: {export_time:.1f}s")
    print(f"  Total: {load_time + gen_time + export_time:.1f}s")
    print(f"[MEMORY] Peak GPU: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")

    # GPU summary
    monitor.print_summary()

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
    print(f"\nGenerated file: {os.path.abspath(args.output)}")
    print(f"GPU log: {os.path.abspath(args.gpu_log)}")
    print(f"\nVisual inspection required:")
    print(f"  1. Open {args.output} in a 3D viewer")
    print(f"  2. Compare with reference/sample_2026-01-24T055452.643.glb")
    print(f"  3. Score: Shape, Texture, Color, PBR (must be 8+/10 each)")

if __name__ == '__main__':
    main()
