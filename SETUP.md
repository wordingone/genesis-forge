# Genesis Forge Setup Guide

Complete setup instructions for Genesis Forge on Windows.

## Prerequisites

Before starting, ensure you have:

1. **Windows 10/11 (64-bit)**

2. **NVIDIA GPU with 16GB+ VRAM**
   - Tested: RTX 3090, RTX 4090, A100
   - Minimum: RTX 3080 (may need reduced settings)

3. **CUDA Toolkit 12.x**
   - Download: https://developer.nvidia.com/cuda-downloads
   - Verify: `nvcc --version`

4. **Python 3.10**
   - Download: https://www.python.org/downloads/release/python-31011/
   - Use Windows installer (64-bit)
   - Add to PATH during installation

5. **Visual Studio 2022 with C++ workload**
   - Download: https://visualstudio.microsoft.com/downloads/
   - Select "Desktop development with C++"
   - Required for gsplat JIT compilation

6. **Node.js 18+**
   - Download: https://nodejs.org/
   - LTS version recommended

## Step-by-Step Setup

### Step 1: Create Python Virtual Environment

```powershell
cd genesis-forge

# Create venv using Python 3.10
python -m venv venv

# Activate
.\venv\Scripts\activate

# Verify Python version
python --version
# Should show: Python 3.10.x
```

### Step 2: Install Python Dependencies

```powershell
# Upgrade pip
python -m pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Install FlexGEMM (editable)
pip install -e .\flexgemm_source
```

### Step 3: Verify CUDA Extensions

The pre-compiled CUDA extensions should work immediately:

```powershell
python -c "import cumesh; print('cumesh OK')"
python -c "import o_voxel; print('o_voxel OK')"
python -c "import flex_gemm; print('flex_gemm OK')"
python -c "import spconv; print('spconv OK')"
python -c "import nvdiffrast; print('nvdiffrast OK')"
```

If any import fails, you may need to rebuild that extension (see Troubleshooting).

### Step 4: Install Electron Dependencies

```powershell
cd gui\electron
npm install
cd ..\..
```

### Step 5: Download Models

Models are downloaded from HuggingFace (~34GB total):

```powershell
# Ensure offline mode is disabled for download
$env:HF_HUB_OFFLINE = '0'

# Run download script
.\venv\Scripts\python.exe scripts\download_models.py
```

Models will be saved to `models/hub/`.

### Step 6: Test Backend

```powershell
.\venv\Scripts\python.exe -m uvicorn gui.backend.main:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000/docs - you should see the API documentation.

Press Ctrl+C to stop.

### Step 7: Test Electron

```powershell
cd gui\electron
npm start
```

The Genesis Forge window should open. Close it when done.

### Step 8: Launch Full Application

```powershell
.\Start-GenesisForge.ps1
```

This starts both backend and Electron together.

## Verification Checklist

After setup, verify:

- [ ] `python --version` shows 3.10.x
- [ ] All CUDA extension imports succeed
- [ ] Backend starts without errors
- [ ] Electron window opens and connects
- [ ] Test generation produces GLB output

## Troubleshooting

### Python Version Mismatch

If you have multiple Python versions:

```powershell
# Use specific Python 3.10
py -3.10 -m venv venv
```

### CUDA Extension Import Errors

If pre-compiled extensions fail:

1. Verify CUDA version matches (12.x)
2. Verify Python version matches (3.10)
3. Check that venv is activated

For rebuilding:
```powershell
cd cumesh
pip install -e .
cd ..
```

### Model Download Fails

If HuggingFace is blocked or slow:

1. Set `HF_HUB_OFFLINE=0` explicitly
2. Try a different network
3. Download models manually and place in `models/hub/`

### Backend Won't Start

Check logs in `outputs/logs/backend_*.log`

Common issues:
- Port 8000 in use (close other apps)
- Missing CUDA (verify nvcc)
- Missing models (run download script)

### Electron Won't Connect

1. Ensure backend is running first
2. Check if port 8000 is accessible
3. Look for errors in Electron console (npm start --dev)

### Textures Are Black

MSVC is missing. Install Visual Studio 2022 with C++ workload.

The vcvars64.bat path can be customized:
```powershell
$env:VCVARS64 = "C:\Path\To\vcvars64.bat"
.\Start-GenesisForge.ps1
```

## Environment Variables

Optional customization:

| Variable | Default | Purpose |
|----------|---------|---------|
| VCVARS64 | Auto-detect | MSVC environment script |
| ATTN_BACKEND | sdpa | Attention backend (sdpa/xformers) |
| SPCONV_ALGO | native | Sparse conv algorithm |
| HF_HUB_OFFLINE | 1 | Block HuggingFace downloads |

## Next Steps

After successful setup:

1. Try a test generation with the sample images in `assets/`
2. Check `outputs/generations/` for results
3. View GLB files with the built-in viewer at `tools/viewers/`

For issues, check `PROGRESS.md` for known issues and fixes.
