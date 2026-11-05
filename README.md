# Genesis Forge

*Part of the research and design work collected at https://wordingone.github.io/.*

> **At a glance (read this first)**
>
> **What it is.** A Windows desktop app (Electron front end, FastAPI backend) that runs Microsoft's [TRELLIS](https://github.com/microsoft/TRELLIS) and [TRELLIS.2](https://github.com/microsoft/TRELLIS.2) models locally for text-to-3D and image-to-3D generation. It is a packaging and integration project: the 3D generation itself is Microsoft's research code and models.
>
> **What is whose.** `trellis/`, `trellis2/`, `cumesh/`, `flexgemm_source/` and `o_voxel/` are vendored from other projects, each under its own licence ([`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)). The code written for this repo is mainly the desktop app in `gui/` and the launch and helper scripts (`Start-GenesisForge.ps1`, `genesis-forge.bat`, `scripts/`, `tools/`).
>
> **Status.** Personal project from January-February 2026, not developed since: the code arrived in one commit on 2026-02-01, followed only by a clean-up on 2026-11-04 ([#1](https://github.com/wordingone/genesis-forge/pull/1): third-party notices, removal of local paths). Words such as "production-ready" in [`PROGRESS.md`](PROGRESS.md) are the working log of the time, not a tested claim. The repo has no record of testing on any machine other than the author's.
>
> **Who wrote it.** I designed and directed this work; most of the code was written by AI coding agents under my review.
>
> **Licence caution.** `cumesh/third_party/cubvh` is also under the NVIDIA Source Code License, which allows use for research or evaluation only (see the notice file). The licence for this repo's own code has not been chosen yet; the root `LICENSE` is Microsoft's TRELLIS MIT text.
>
> **Requirements in short.** Windows 10/11, an NVIDIA GPU with 16 GB+ VRAM, CUDA 12.x, Python 3.10, about 34 GB of model downloads. Two of the three CUDA extensions are committed pre-built for Python 3.10 on Windows; `cumesh` is not and must be built (see below).

---

Desktop application for local 3D asset generation from text and images.

Built on [TRELLIS](https://github.com/microsoft/TRELLIS) - Microsoft's Structured 3D Latents for Scalable and Versatile 3D Generation.

## Features

- **Text-to-3D**: Generate 3D models from text descriptions
- **Image-to-3D**: Generate 3D models from reference images with PBR materials
- **Multiple Formats**: Export as GLB, PLY, or render videos
- **Offline Operation**: All models run locally after initial download
- **Windows Native**: Pre-built `flex_gemm` and `o_voxel` CUDA extensions for Python 3.10 are committed; `cumesh` must be built locally

## Requirements

- **OS**: Windows 10/11 (64-bit)
- **GPU**: NVIDIA GPU with 16GB+ VRAM (RTX 3090, RTX 4090, A100, etc.)
- **CUDA**: CUDA Toolkit 12.x
- **Python**: Python 3.10 (cp310-win_amd64)
- **Visual Studio**: MSVC 2022 (for gsplat JIT compilation)
- **Node.js**: 18+ (for Electron)

## Quick Start

1. **Create Python Environment**
   ```powershell
   cd genesis-forge
   python -m venv venv
   .\venv\Scripts\activate
   pip install -r requirements.txt
   pip install -e .\flexgemm_source
   cd cumesh; pip install -e .; cd ..   # cumesh has no committed binary
   ```

2. **Install Electron Dependencies**
   ```powershell
   cd gui\electron
   npm install
   cd ..\..
   ```

3. **Download Models (~34GB)**
   ```powershell
   .\venv\Scripts\python.exe scripts\download_models.py
   ```

4. **Launch Application**
   ```powershell
   .\Start-GenesisForge.ps1
   ```

## Directory Structure

```
genesis-forge/
|-- gui/               # Application (FastAPI + Electron)
|   |-- backend/       # FastAPI server
|   |-- electron/      # Electron frontend
|-- trellis/           # TRELLIS 1 pipeline (text-to-3D)
|-- trellis2/          # TRELLIS.2 pipeline (image-to-3D with PBR)
|-- cumesh/            # CUDA mesh extension (pre-compiled)
|-- o_voxel/           # O-Voxel extension (pre-compiled)
|-- flexgemm_source/   # FlexGEMM sparse conv (editable install)
|-- configs/           # Pipeline configurations
|-- models/            # Downloaded models (user creates)
|-- outputs/           # Generated outputs
|-- scripts/           # Helper scripts
|-- tests/             # Test suite
|-- tools/             # Viewers and utilities
```

## Pre-compiled CUDA Extensions

The table lists the CUDA extensions the app needs. `flex_gemm` (`flexgemm_source/flex_gemm/kernels/cuda.cp310-win_amd64.pyd`) and `o_voxel` (`o_voxel/o_voxel/_C.cp310-win_amd64.pyd`) are committed pre-built for Python 3.10 + Windows + CUDA 12.x. `cumesh` is **not**: its `.gitignore` excludes the built file, so build it with `cd cumesh; pip install -e .` (or `cumesh/build_cumesh.ps1`, which assumes Visual Studio 2022 Community and CUDA 12.8 at their default paths).

| Extension | Purpose |
|-----------|---------|
| cumesh | CUDA mesh simplification |
| o_voxel | O-Voxel mesh extraction |
| flex_gemm | Sparse convolution with int64 indices |

If you have a different Python version or CUDA version, you may need to rebuild these extensions.

## Models

Genesis Forge uses the following models (downloaded to `models/`):

| Model | Pipeline | Size |
|-------|----------|------|
| TRELLIS-text-xlarge | Text-to-3D | ~8GB |
| TRELLIS-image-large | Image-to-3D (basic) | ~5GB |
| TRELLIS.2-4B | Image-to-3D (PBR) | ~16GB |
| DINOv2 | Image encoder | ~1GB |
| DPT | Depth estimation | ~0.5GB |

## Usage

### Full Application (Recommended)
```powershell
.\Start-GenesisForge.ps1
```

### Backend Only
```powershell
.\Start-GenesisForge.ps1 -BackendOnly
# API available at http://127.0.0.1:8000/docs
```

### Command Line
```powershell
# Batch launcher
.\genesis-forge.bat
.\genesis-forge.bat -backend
```

## Troubleshooting

### CUDA Extension Errors
Ensure you're using the project venv, not system Python:
```powershell
.\venv\Scripts\python.exe -c "import cumesh, o_voxel, flex_gemm; print('OK')"
```

### Textures Rendering Black
Install Visual Studio 2022 with C++ workload. The MSVC compiler is needed for gsplat JIT compilation.

### Out of Memory
- Close other GPU applications
- Reduce generation resolution in settings
- Use a GPU with more VRAM

### Model Download Issues
If offline mode blocks downloads, temporarily set:
```powershell
$env:HF_HUB_OFFLINE = '0'
.\venv\Scripts\python.exe scripts\download_models.py
```

## License

This repository vendors code from several projects, each under its own licence; see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). Most are MIT (TRELLIS, TRELLIS.2, CuMesh, FlexGEMM, xatlas), with Apache-2.0 (FlexiCubes) and MPL-2.0 (Eigen). `cumesh/third_party/cubvh` is additionally under the NVIDIA Source Code License, which limits use to research or evaluation.

The root [LICENSE](LICENSE) is Microsoft's MIT licence for TRELLIS. A licence for the code written for Genesis Forge itself has not been chosen yet. Model weights are downloaded at runtime and carry their own terms.

## Acknowledgments

- [Microsoft TRELLIS](https://github.com/microsoft/TRELLIS) - Core 3D generation technology
- [TRELLIS.2](https://github.com/microsoft/TRELLIS.2) - PBR material support
- [CuMesh](https://github.com/JeffreyXiang/CuMesh) and [FlexGEMM](https://github.com/JeffreyXiang/FlexGEMM) - CUDA mesh and sparse-convolution extensions
- The other vendored components listed in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)
