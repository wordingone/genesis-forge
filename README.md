# Genesis Forge

Desktop application for high-quality 3D asset generation from text and images.

Built on [TRELLIS](https://github.com/microsoft/TRELLIS) - Microsoft's Structured 3D Latents for Scalable and Versatile 3D Generation.

## Features

- **Text-to-3D**: Generate 3D models from text descriptions
- **Image-to-3D**: Generate 3D models from reference images with PBR materials
- **Multiple Formats**: Export as GLB, PLY, or render videos
- **Offline Operation**: All models run locally after initial download
- **Windows Native**: Pre-compiled CUDA extensions included

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

The following CUDA extensions are included pre-compiled for Python 3.10 + Windows + CUDA 12.x:

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
.\genesis-forge.bat -gradio
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

This project builds on [TRELLIS](https://github.com/microsoft/TRELLIS) by Microsoft, licensed under MIT.

See [LICENSE](LICENSE) for full terms.

## Acknowledgments

- [Microsoft TRELLIS](https://github.com/microsoft/TRELLIS) - Core 3D generation technology
- [TRELLIS.2](https://github.com/microsoft/TRELLIS.2) - PBR material support
