# Deployment Guide
## TRELLIS Forge v1.0

**Product**: TRELLIS Forge
**Version**: 1.0.0 (Initial Windows Release)
**Date**: 2026-01-20
**Status**: Production-Ready (Internal/Demo)

---

## 1. Overview

This document provides comprehensive deployment instructions for TRELLIS Forge, covering development, testing, and production environments. The current implementation targets Windows 11 with NVIDIA GPUs.

**Deployment Scope**:
- ✅ Development (localhost, single user)
- ✅ Internal demo (local network, controlled access)
- ⚠️ Production (public access, multiple users) - requires additional security measures

---

## 2. System Requirements

### 2.1 Hardware Requirements

#### Minimum Configuration
- **GPU**: NVIDIA RTX 3090 (24 GB VRAM)
- **CPU**: 8-core Intel Core i7 / AMD Ryzen 7
- **RAM**: 32 GB DDR4
- **Storage**: 50 GB SSD
  - 20 GB for model checkpoint
  - 20 GB for Python environment
  - 10 GB for temp files

#### Recommended Configuration (Current System)
- **GPU**: NVIDIA RTX 4090 (24 GB VRAM)
- **CPU**: 16-core Intel Core i9 / AMD Ryzen 9
- **RAM**: 64 GB DDR4
- **Storage**: 100 GB NVMe SSD

#### Production Configuration (Multi-User)
- **GPUs**: N × RTX 4090 (one per concurrent user)
- **CPU**: 32-core Xeon / EPYC
- **RAM**: 128 GB DDR4 ECC
- **Storage**: 500 GB NVMe SSD RAID

### 2.2 Software Requirements

#### Operating System
- **Supported**: Windows 11 (Build 22000 or later)
- **Tested**: Windows 11 Build 26200.7623
- **Production**: Windows Server 2022 (recommended)
- **Alternative**: Ubuntu 22.04 LTS (untested, may require modifications)

#### System Software
- **NVIDIA Driver**: Version ≥ 560 (CUDA 12.6+ support)
  - Current: 581.29 (supports CUDA 13.0)
  - Download: https://www.nvidia.com/Download/index.aspx
- **CUDA Toolkit**: Version 12.6 or newer
  - Current: 12.6
  - Download: https://developer.nvidia.com/cuda-downloads
- **Visual Studio 2022**: Community Edition or higher
  - Component: Desktop development with C++
  - MSVC v143 compiler
  - Download: https://visualstudio.microsoft.com/downloads/
- **Python**: Version 3.11.x
  - Current: 3.11.11
  - Download: https://www.python.org/downloads/

#### Network Requirements
- **Bandwidth**: 10 Mbps download per user (for model checkpoint)
- **Latency**: <100 ms (local network preferred)
- **Firewall**: Allow port 7860 (Gradio server)

---

## 3. Installation Guide

### 3.1 Prerequisites Installation

#### Step 1: Install NVIDIA Driver

```powershell
# Check current driver version
nvidia-smi

# If driver is old (< version 560), download and install:
# https://www.nvidia.com/Download/index.aspx
# Select: GeForce RTX 40 Series, RTX 4090, Windows 11 64-bit

# Verify installation
nvidia-smi
# Should show driver version ≥ 560 and CUDA version ≥ 12.6
```

#### Step 2: Install CUDA Toolkit 12.6

```powershell
# Download installer:
# https://developer.nvidia.com/cuda-12-6-0-download-archive
# Select: Windows, x86_64, 11, exe (local)

# Run installer (follow prompts)
# Default installation path: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6

# Verify installation
nvcc --version
# Should show: Cuda compilation tools, release 12.6
```

#### Step 3: Install Visual Studio 2022

```powershell
# Download Visual Studio Community 2022:
# https://visualstudio.microsoft.com/downloads/

# During installation, select:
# - Desktop development with C++
# - Windows 10 SDK (latest)
# - MSVC v143 - VS 2022 C++ x64/x86 build tools

# Verify installation
where cl
# Should show: C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\...\cl.exe
```

#### Step 4: Install Python 3.11

```powershell
# Download Python 3.11.11:
# https://www.python.org/downloads/release/python-31111/
# Select: Windows installer (64-bit)

# Run installer:
# - Check "Add Python 3.11 to PATH"
# - Choose "Install Now"

# Verify installation
python --version
# Should show: Python 3.11.11
```

### 3.2 Application Installation

#### Step 1: Clone Repository

```powershell
# Navigate to installation directory
cd <repo>

# If not already cloned:
# git clone <repository-url> spatial
cd spatial
```

#### Step 2: Create Virtual Environment

```powershell
# Create Python 3.11 virtual environment
python -m venv venv311

# Activate environment
.\venv311\Scripts\activate

# Verify Python version
python --version
# Should show: Python 3.11.x
```

#### Step 3: Install Dependencies

```powershell
# Upgrade pip
python -m pip install --upgrade pip

# Install PyTorch with CUDA 12.8 support
pip install torch==2.8.0+cu128 --index-url https://download.pytorch.org/whl/cu128

# Install core dependencies
pip install gradio==6.0.1
pip install transformers diffusers
pip install trimesh pygltflib open3d
pip install pillow opencv-python imageio imageio-ffmpeg
pip install rembg
pip install spconv-cu128
pip install xformers==0.0.33.post2
pip install gradio-litmodel3d
pip install easydict

# Fix numpy version for numba compatibility
pip install "numpy<2.4"
```

#### Step 4: Build gsplat from Source (CRITICAL)

**Create Build Script** `build_gsplat_final.bat`:

```batch
@echo off
echo ========================================
echo Building gsplat 1.5.3 for Windows
echo ========================================

REM Minimal PATH to avoid "input line too long" error
set PATH=C:\Windows\System32;C:\Windows;C:\Windows\System32\Wbem
set PATH=%PATH%;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin
set PATH=%PATH%;C:\Program Files (x86)\Microsoft Visual Studio\Installer

REM User TEMP folder to avoid access denied
set TEMP=C:\Users\%USERNAME%\AppData\Local\Temp
set TMP=C:\Users\%USERNAME%\AppData\Local\Temp

REM CUDA configuration
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6
set TORCH_CUDA_ARCH_LIST=8.9
set FORCE_CUDA=1

REM Fix for setuptools VC environment detection
set DISTUTILS_USE_SDK=1

echo Initializing Visual Studio environment...
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

echo.
echo Building gsplat from source...
pip install gsplat --no-binary gsplat --no-cache-dir

echo.
echo Build complete!
pause
```

**Run Build**:

```powershell
# Ensure virtual environment is activated
.\venv311\Scripts\activate

# Run build script
.\build_gsplat_final.bat

# Wait for completion (5-10 minutes)
# Success message: "Successfully built gsplat 1.5.3"

# Verify installation
python -c "import gsplat; print(gsplat.__version__)"
# Should show: 1.5.3
```

### 3.3 Model Checkpoint Download

**First-Time Setup**:

```powershell
# Activate virtual environment
.\venv311\Scripts\activate

# Navigate to trellis directory
cd trellis

# Run application (will download model on first run)
python app.py

# Download progress will be displayed:
# Downloading microsoft/TRELLIS-image-large...
# Downloading (14.5 GB): 100%|██████████| 14.5G/14.5G [XX:XX<00:00, XXMB/s]

# Model is cached at:
# C:\Users\<username>\.cache\huggingface\hub\models--microsoft--TRELLIS-image-large
```

**Offline Installation** (if download fails):

```powershell
# Pre-download model on machine with fast internet
# Then copy entire directory:
# From: C:\Users\<user>\.cache\huggingface\hub\models--microsoft--TRELLIS-image-large
# To: Target machine same path

# Verify model is available:
python -c "from transformers import AutoModel; AutoModel.from_pretrained('microsoft/TRELLIS-image-large', local_files_only=True)"
```

---

## 4. Configuration

### 4.1 Application Configuration

**Current Configuration** (app.py lines 18, 401-403):

```python
# Temporary directory for user files
TMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tmp')

# Model loading
pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
pipeline.cuda()

# Server launch
demo.launch()
```

**Optional Configuration** (modify app.py):

```python
# Server configuration
demo.launch(
    server_name="0.0.0.0",      # Listen on all interfaces (default: 127.0.0.1)
    server_port=7860,           # Port (default: 7860)
    share=False,                # Create public URL via Gradio (default: False)
    auth=("admin", "password"), # Basic authentication (default: None)
    ssl_certfile="cert.pem",    # HTTPS certificate (default: None)
    ssl_keyfile="key.pem",      # HTTPS key (default: None)
)
```

### 4.2 Environment Variables

**Optional Settings**:

```powershell
# Create .env file (requires python-dotenv)
# GPU selection
set CUDA_VISIBLE_DEVICES=0

# PyTorch memory allocation
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Gradio cache directory
set GRADIO_TEMP_DIR=D:\temp\gradio

# HuggingFace cache directory
set HF_HOME=D:\huggingface
```

### 4.3 Firewall Configuration

**Windows Firewall** (for network access):

```powershell
# Allow inbound connections on port 7860
New-NetFirewallRule -DisplayName "TRELLIS Forge" `
    -Direction Inbound `
    -LocalPort 7860 `
    -Protocol TCP `
    -Action Allow

# Verify rule
Get-NetFirewallRule -DisplayName "TRELLIS Forge"
```

---

## 5. Deployment Procedures

### 5.1 Development Deployment (Localhost)

**Purpose**: Single developer, testing on local machine

**Steps**:

```powershell
# 1. Navigate to application directory
cd <repo>

# 2. Activate virtual environment
..\venv311\Scripts\activate

# 3. Launch application
python app.py

# Output:
# Running on local URL:  http://127.0.0.1:7860
# To create a public link, set `share=True` in `launch()`.
```

**Access**:
- URL: `http://127.0.0.1:7860`
- Users: 1 (localhost only)
- Security: None required (not accessible externally)

**Shutdown**:
- Press `Ctrl+C` in terminal
- Cleanup: Temp files automatically removed on session end

### 5.2 Internal Demo Deployment (Local Network)

**Purpose**: Multiple users on local network, demo/testing

**Prerequisites**:
- Static IP or hostname (e.g., `192.168.1.100`)
- Firewall configured (port 7860 open)

**Modify app.py**:

```python
# Line 403: Change launch configuration
demo.launch(
    server_name="0.0.0.0",  # Listen on all interfaces
    server_port=7860,
    auth=("demo", "password123"),  # Basic authentication
)
```

**Launch**:

```powershell
# Activate environment
.\venv311\Scripts\activate

# Launch with network access
cd trellis
python app.py

# Output:
# Running on local URL:  http://0.0.0.0:7860
# Running on network URL: http://192.168.1.100:7860
```

**Access**:
- URL: `http://<machine-ip>:7860`
- Users: 1 concurrent (add queue for multiple)
- Security: Basic auth (username/password)

**Limitations**:
- No HTTPS (plaintext communication)
- No concurrent user support (single GPU)
- No load balancing
- No persistent storage

### 5.3 Production Deployment (Public Access)

**WARNING**: Current implementation NOT production-ready. Requires:
- HTTPS (reverse proxy)
- Queue management (concurrent users)
- Rate limiting (DoS protection)
- Input validation (security)
- Monitoring (observability)

**Recommended Architecture**:

```
Internet
    │
    ▼
[nginx Reverse Proxy]
    ├─ HTTPS termination (TLS 1.3)
    ├─ Rate limiting (10 req/hour per IP)
    ├─ Load balancing (if multiple workers)
    │
    ▼
[TRELLIS Forge Workers]
    ├─ Worker 1 (GPU 0): RTX 4090
    ├─ Worker 2 (GPU 1): RTX 4090
    └─ Worker N (GPU N): RTX 4090
    │
    ▼
[Redis Queue]
    ├─ Request queue
    └─ Result cache
    │
    ▼
[Persistent Storage]
    ├─ S3 / Azure Blob
    └─ User outputs (GLB, PLY)
```

**Production Deployment Steps** (not implemented in v1.0):

1. **Containerization** (Docker):
   ```dockerfile
   FROM nvidia/cuda:12.6-devel-ubuntu22.04
   RUN apt-get update && apt-get install -y python3.11
   COPY . /app
   RUN pip install -r requirements.txt
   RUN ./build_gsplat_final.sh
   CMD ["python3", "app.py"]
   ```

2. **Orchestration** (Kubernetes):
   - Deploy worker pods with GPU affinity
   - Horizontal pod autoscaler (based on queue depth)
   - Persistent volume for model checkpoint

3. **Security**:
   - OAuth2 authentication (Azure AD, Google)
   - API rate limiting (10 requests/hour per user)
   - Input sanitization (file type, size validation)

4. **Monitoring**:
   - Prometheus (metrics: requests/sec, VRAM usage, latency)
   - Grafana (dashboards)
   - ELK stack (log aggregation)

**For v1.0**: Production deployment NOT SUPPORTED. Use internal demo deployment only.

---

## 6. Verification Procedures

### 6.1 Installation Verification

**Test 1: Import Check**

```powershell
# Activate environment
.\venv311\Scripts\activate

# Test critical imports
python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"
# Expected: PyTorch: 2.8.0+cu128 CUDA: True

python -c "import gsplat; print('gsplat:', gsplat.__version__)"
# Expected: gsplat: 1.5.3

python -c "import gradio; print('Gradio:', gradio.__version__)"
# Expected: Gradio: 6.0.1
```

**Test 2: CUDA Verification**

```powershell
# Test CUDA runtime
python -c "import torch; print('CUDA devices:', torch.cuda.device_count()); print('Device name:', torch.cuda.get_device_name(0))"
# Expected:
# CUDA devices: 1
# Device name: NVIDIA GeForce RTX 4090
```

**Test 3: Model Loading**

```powershell
# Test pipeline initialization
python -c "from trellis.pipelines import TrellisImageTo3DPipeline; pipeline = TrellisImageTo3DPipeline.from_pretrained('microsoft/TRELLIS-image-large'); print('Pipeline loaded successfully')"
# Expected: Pipeline loaded successfully (may take 1-2 minutes)
```

### 6.2 Functional Verification

**Test 4: gsplat CUDA Test**

Create `test_gsplat_cuda.py`:

```python
import torch
import gsplat

device = "cuda"
N = 1000
means = torch.randn(N, 3, device=device)
quats = torch.randn(N, 4, device=device)
quats = quats / quats.norm(dim=-1, keepdim=True)
scales = torch.rand(N, 3, device=device) * 0.1
opacities = torch.rand(N, device=device)
colors = torch.rand(N, 3, device=device)

viewmats = torch.eye(4, device=device).unsqueeze(0)
Ks = torch.tensor([[500, 0, 256], [0, 500, 256], [0, 0, 1]], device=device, dtype=torch.float32).unsqueeze(0)

render_colors, render_alphas, meta = gsplat.rasterization(
    means=means, quats=quats, scales=scales, opacities=opacities,
    colors=colors, viewmats=viewmats, Ks=Ks, width=512, height=512,
)

print(f"Render output shape: {render_colors.shape}")
print(f"Non-zero pixels: {(render_colors.sum(dim=-1) > 0).sum().item()}")
print("✅ gsplat CUDA test PASSED")
```

Run:
```powershell
python test_gsplat_cuda.py
# Expected: ✅ gsplat CUDA test PASSED
```

**Test 5: End-to-End GLB Export**

Create `test_glb_export.py` (see PROGRESS.md for full version)

Run:
```powershell
python test_glb_export.py
# Expected: GLB file created (0.3-0.5 MB)
```

**Test 6: Application Launch**

```powershell
# Launch application
python app.py

# Should see:
# Running on local URL:  http://127.0.0.1:7860

# In browser: http://127.0.0.1:7860
# 1. Upload example image
# 2. Click "Generate" → Video preview appears (~4 seconds)
# 3. Click "Extract GLB" → 3D model appears in viewer (~30 seconds)
# 4. Click "Download GLB" → File downloads successfully

# ✅ Application functional
```

### 6.3 Performance Verification

**Test 7: Latency Benchmark**

```python
import time
from trellis.pipelines import TrellisImageTo3DPipeline
from PIL import Image

pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
pipeline.cuda()

image = Image.open("assets/example_image/chair_1.png")
image = pipeline.preprocess_image(image)

times = []
for i in range(3):
    start = time.time()
    outputs = pipeline.run(image, seed=i, formats=["gaussian", "mesh"],
                           preprocess_image=False,
                           sparse_structure_sampler_params={"steps": 12, "cfg_strength": 7.5},
                           slat_sampler_params={"steps": 12, "cfg_strength": 3.0})
    elapsed = time.time() - start
    times.append(elapsed)
    print(f"Run {i+1}: {elapsed:.2f}s")

print(f"\nAverage: {sum(times)/len(times):.2f}s")
# Expected: ~4 seconds on RTX 4090
```

### 6.4 Memory Verification

**Test 8: VRAM Usage**

```python
import torch
from trellis.pipelines import TrellisImageTo3DPipeline
from PIL import Image

torch.cuda.reset_peak_memory_stats()

pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
pipeline.cuda()
print(f"Model loaded: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

image = Image.open("assets/example_image/chair_1.png")
image = pipeline.preprocess_image(image)

outputs = pipeline.run(image, seed=0, formats=["gaussian", "mesh"],
                       preprocess_image=False,
                       sparse_structure_sampler_params={"steps": 12, "cfg_strength": 7.5},
                       slat_sampler_params={"steps": 12, "cfg_strength": 3.0})

print(f"Peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
# Expected: ~12 GB on RTX 4090
```

---

## 7. Operational Procedures

### 7.1 Starting the Application

**Standard Startup**:

```powershell
# 1. Open PowerShell
# 2. Navigate to application
cd <repo>

# 3. Activate environment
..\venv311\Scripts\activate

# 4. Launch application
python app.py

# 5. Wait for model loading (~30 seconds first time)
# 6. Access UI at http://127.0.0.1:7860
```

**Background Startup** (Windows Task Scheduler):

```powershell
# Create scheduled task to run on boot
$action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-ExecutionPolicy Bypass -File C:\path\to\start_trellis.ps1"

$trigger = New-ScheduledTaskTrigger -AtStartup

Register-ScheduledTask -TaskName "TRELLIS Forge" `
    -Action $action -Trigger $trigger -RunLevel Highest

# start_trellis.ps1 contents:
# cd <repo>
# .\venv311\Scripts\activate
# python app.py
```

### 7.2 Monitoring

**Process Monitoring** (PowerShell):

```powershell
# Check if application is running
Get-Process | Where-Object {$_.Name -eq "python"}

# Monitor VRAM usage
nvidia-smi --query-gpu=memory.used,memory.total --format=csv --loop=5
```

**Log Monitoring**:

```powershell
# Application logs (stdout)
# Currently: No log file, only console output

# Recommended: Redirect to log file
python app.py > logs\trellis_$(Get-Date -Format 'yyyyMMdd_HHmmss').log 2>&1
```

**Disk Space Monitoring**:

```powershell
# Check temp directory size
Get-ChildItem -Path "<repo>\tmp" -Recurse | Measure-Object -Property Length -Sum

# Clean old temp directories (if sessions not cleaned)
Get-ChildItem -Path "<repo>\tmp" | Where-Object {$_.LastWriteTime -lt (Get-Date).AddDays(-1)} | Remove-Item -Recurse -Force
```

### 7.3 Stopping the Application

**Graceful Shutdown**:

```powershell
# In application console window
# Press Ctrl+C

# Wait for cleanup (~5 seconds)
# Output: "Keyboard interrupt received, exiting."
```

**Force Stop**:

```powershell
# Find Python process
Get-Process python

# Kill specific process
Stop-Process -Name python -Force
```

### 7.4 Backup and Restore

**Backup Checklist**:

```powershell
# 1. Model checkpoint (14.5 GB)
# Location: C:\Users\<username>\.cache\huggingface\hub\models--microsoft--TRELLIS-image-large
# Backup: Copy entire directory

# 2. Application code
# Location: <repo>
# Backup: Git commit + push OR zip entire directory

# 3. Virtual environment (optional, can be recreated)
# Location: <repo>\venv
# Restore: Run installation steps again

# 4. User outputs (if saved)
# Location: <repo>\tmp
# Backup: Not recommended (temp files, deleted on session end)
```

**Restore Procedure**:

```powershell
# 1. Restore application code
cd <repo>
git clone <repository-url> spatial
# OR: Unzip backup

# 2. Restore model checkpoint
# Copy from backup to:
# C:\Users\<username>\.cache\huggingface\hub\models--microsoft--TRELLIS-image-large

# 3. Recreate virtual environment
cd spatial
python -m venv venv311
.\venv311\Scripts\activate
pip install -r requirements.txt
.\build_gsplat_final.bat

# 4. Verify installation
python test_gsplat_cuda.py
python app.py
```

---

## 8. Troubleshooting

### 8.1 Common Issues

#### Issue 1: gsplat build fails with "nvcc fatal: Could not set up environment"

**Symptoms**:
```
nvcc fatal : Could not set up the environment for Microsoft Visual Studio using 'vcvars64.bat'
```

**Diagnosis**:
- PATH length too long (>8000 chars)
- TEMP folder permission issues

**Solution**:
```batch
# Use minimal PATH in build_gsplat_final.bat:
set PATH=C:\Windows\System32;C:\Windows;C:\Windows\System32\Wbem
set PATH=%PATH%;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin
set PATH=%PATH%;C:\Program Files (x86)\Microsoft Visual Studio\Installer

# Use user TEMP folder:
set TEMP=C:\Users\%USERNAME%\AppData\Local\Temp
set TMP=C:\Users\%USERNAME%\AppData\Local\Temp

# Set DISTUTILS_USE_SDK:
set DISTUTILS_USE_SDK=1
```

#### Issue 2: "DLL load failed while importing gsplat_cuda"

**Symptoms**:
```
ImportError: DLL load failed while importing gsplat_cuda: The specified module could not be found.
```

**Diagnosis**:
- gsplat installed without CUDA extensions (pip found pre-built wheel without CUDA)

**Solution**:
```powershell
# Uninstall gsplat
pip uninstall gsplat -y

# Build from source with CUDA
.\build_gsplat_final.bat

# Verify CUDA extensions present
python -c "import gsplat; print(gsplat._C)"
# Should show: <module 'gsplat._C' from '...gsplat\_C.cp311-win_amd64.pyd'>
```

#### Issue 3: NumPy version conflict

**Symptoms**:
```
ImportError: Numba needs NumPy 2.3 or less. Got NumPy 2.4.
```

**Diagnosis**:
- gsplat installation upgraded numpy to 2.4.1
- numba (rembg dependency) requires ≤2.3

**Solution**:
```powershell
pip install "numpy<2.4"
# Downgrades to numpy 2.3.5
```

#### Issue 4: Out of Memory (OOM)

**Symptoms**:
```
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate X GB
```

**Diagnosis**:
- Image too large
- Multiple concurrent generations
- VRAM fragmentation

**Solution**:
```powershell
# Clear VRAM
python -c "import torch; torch.cuda.empty_cache()"

# Restart application (full VRAM reset)
# Ctrl+C → python app.py

# Reduce image resolution (modify pipeline preprocessing)
# Reduce sampling steps (12 → 6)
```

#### Issue 5: GLB export fails with black textures

**Symptoms**:
- GLB file created but textures are black
- Console shows: "gsplat rasterization failed: ..."

**Diagnosis**:
- gsplat CUDA kernel error
- Incorrect background color shape
- Incorrect radii format

**Solution**:
```powershell
# Verify gsplat_wrapper.py fixes applied:
# - Line 105: backgrounds = self.settings.bg (not .unsqueeze(0))
# - Lines 126-142: Radii format conversion [N_visible,2] → [N]

# Test gsplat directly:
python test_gsplat_cuda.py
# Should show: ✅ gsplat CUDA test PASSED

# If fails, rebuild gsplat:
pip uninstall gsplat -y
.\build_gsplat_final.bat
```

### 8.2 Diagnostic Tools

**Check GPU Status**:
```powershell
nvidia-smi
# Shows: VRAM usage, GPU utilization, temperature
```

**Check CUDA Version**:
```powershell
nvcc --version
# Shows: CUDA compiler version

python -c "import torch; print(torch.version.cuda)"
# Shows: PyTorch CUDA version
```

**Check Python Environment**:
```powershell
# List installed packages
pip list

# Check for conflicts
pip check
```

**Check File Permissions**:
```powershell
# Test write permission to temp directory
New-Item -Path "<repo>\tmp\test.txt" -ItemType File
Remove-Item -Path "<repo>\tmp\test.txt"
```

### 8.3 Log Analysis

**Enable Verbose Logging**:

```python
# Add to top of app.py
import logging
logging.basicConfig(level=logging.DEBUG)
```

**Common Error Patterns**:

| Error Message | Likely Cause | Solution |
|---------------|--------------|----------|
| "CUDA out of memory" | OOM | Restart app, reduce image size |
| "DLL load failed" | gsplat not built with CUDA | Rebuild gsplat |
| "Connection refused" | Firewall blocking | Allow port 7860 |
| "Session not found" | Session expired | Refresh browser |
| "File not found" | Temp file deleted | Regenerate 3D model |

---

## 9. Upgrade Procedures

### 9.1 Minor Version Upgrade (v1.0.x → v1.0.y)

**Steps**:

```powershell
# 1. Backup current installation
cd <repo>
git commit -am "Backup before upgrade"

# 2. Pull updates
git pull origin main

# 3. Update Python dependencies
.\venv311\Scripts\activate
pip install --upgrade -r requirements.txt

# 4. Test
python test_gsplat_cuda.py
python app.py
```

### 9.2 Major Version Upgrade (v1.x → v2.x)

**Steps**:

```powershell
# 1. Full backup
# Backup code, model checkpoint, and environment

# 2. Review changelog
# Check for breaking changes

# 3. Create new environment (recommended)
python -m venv venv_v2
.\venv_v2\Scripts\activate

# 4. Install dependencies
pip install -r requirements_v2.txt

# 5. Rebuild gsplat if needed
.\build_gsplat_final.bat

# 6. Migrate configuration
# Copy custom settings from old app.py to new

# 7. Test thoroughly
# Run all verification tests

# 8. Switch to new version
# Update launch scripts to use venv_v2
```

---

## 10. Maintenance

### 10.1 Regular Maintenance Tasks

**Daily** (if running 24/7):
- Check disk space (temp directory)
- Monitor VRAM usage
- Check error logs

**Weekly**:
- Clean temp directory (old sessions)
- Review error logs for patterns
- Update Windows security patches

**Monthly**:
- Update NVIDIA driver (if new version available)
- Update Python dependencies (minor versions)
- Backup model checkpoint

**Quarterly**:
- Full system backup
- Performance benchmarking (check for degradation)
- Security audit

### 10.2 Cleanup Procedures

**Temp Directory Cleanup**:

```powershell
# Manual cleanup
cd <repo>\tmp
Get-ChildItem | Remove-Item -Recurse -Force

# Automated cleanup (PowerShell script)
$tempDir = "<repo>\tmp"
Get-ChildItem -Path $tempDir | Where-Object {
    $_.LastWriteTime -lt (Get-Date).AddHours(-1)
} | Remove-Item -Recurse -Force
```

**Model Cache Cleanup** (if running low on space):

```powershell
# HuggingFace cache location
$cacheDir = "$env:USERPROFILE\.cache\huggingface\hub"

# List cached models
Get-ChildItem -Path $cacheDir

# Remove specific model cache (will re-download on next run)
Remove-Item -Path "$cacheDir\models--microsoft--TRELLIS-image-large" -Recurse -Force
```

**Python Package Cleanup**:

```powershell
# Remove unused packages
pip autoremove  # If pip-autoremove installed

# Clean pip cache
pip cache purge
```

---

## 11. Security Hardening (Production)

### 11.1 Network Security

**Reverse Proxy (nginx)**:

```nginx
# /etc/nginx/sites-available/trellis-forge
server {
    listen 443 ssl http2;
    server_name trellis.example.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=trellis:10m rate=10r/h;
    limit_req zone=trellis burst=5 nodelay;

    location / {
        proxy_pass http://127.0.0.1:7860;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }

    # File size limit
    client_max_body_size 20M;
}
```

### 11.2 Application Security

**Input Validation**:

```python
# Add to app.py before image_to_3d()
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB
MAX_DIMENSION = 2048
ALLOWED_FORMATS = ['PNG', 'JPEG', 'JPG']

def validate_image(image: PIL.Image):
    if image.format not in ALLOWED_FORMATS:
        raise gr.Error(f"Invalid format. Allowed: {ALLOWED_FORMATS}")

    if image.width > MAX_DIMENSION or image.height > MAX_DIMENSION:
        raise gr.Error(f"Image too large. Max dimension: {MAX_DIMENSION}")

    # Check file size (approximate)
    import io
    buf = io.BytesIO()
    image.save(buf, format='PNG')
    if buf.tell() > MAX_FILE_SIZE:
        raise gr.Error(f"File too large. Max size: {MAX_FILE_SIZE // 1024 // 1024} MB")

# Call in image_to_3d():
validate_image(image)
```

**Authentication**:

```python
# Basic authentication (app.py)
demo.launch(
    auth=[("admin", "secure_password_here")],
    auth_message="Enter credentials to access TRELLIS Forge"
)

# OAuth2 (requires gradio-oauth)
# pip install gradio-oauth
from gradio_oauth import OAuthButton

# Configure OAuth provider (Google, GitHub, etc.)
```

---

## 12. Disaster Recovery

### 12.1 Failure Scenarios

| Scenario | Impact | Recovery Time | Procedure |
|----------|--------|---------------|-----------|
| Application crash | Users cannot generate | 1 minute | Restart app: `python app.py` |
| GPU hang | Generation fails | 5 minutes | Reboot machine |
| Disk full | Cannot save outputs | 10 minutes | Clean temp directory, restart |
| Model corruption | Cannot load model | 1 hour | Re-download checkpoint |
| OS failure | Full outage | 4 hours | Restore from backup, reinstall |

### 12.2 Backup Strategy

**Critical Assets**:
1. **Application Code** (200 MB):
   - Frequency: Continuous (Git)
   - Location: GitHub repository
   - RTO: 5 minutes

2. **Model Checkpoint** (14.5 GB):
   - Frequency: Once (cached)
   - Location: External drive / cloud storage
   - RTO: 1 hour (re-download)

3. **Configuration** (1 KB):
   - Frequency: On change
   - Location: Git (app.py modifications)
   - RTO: 1 minute

### 12.3 Recovery Procedures

**Scenario: Application Crash**

```powershell
# 1. Check if still running
Get-Process python -ErrorAction SilentlyContinue

# 2. If hung, force kill
Stop-Process -Name python -Force

# 3. Restart
cd <repo>
.\venv311\Scripts\activate
python app.py
```

**Scenario: Disk Full**

```powershell
# 1. Check disk space
Get-PSDrive C | Select-Object Used,Free

# 2. Clean temp directory
Remove-Item -Path "<repo>\tmp\*" -Recurse -Force

# 3. Clean pip cache
pip cache purge

# 4. If still low, move model cache to larger drive
# (Requires changing HF_HOME environment variable)
```

**Scenario: Complete System Failure**

```powershell
# 1. Install prerequisites (NVIDIA driver, CUDA, VS2022, Python)
# See Section 3.1

# 2. Restore application code
git clone <repository-url> spatial

# 3. Restore model checkpoint (if backed up)
# Copy to C:\Users\<username>\.cache\huggingface\...

# 4. Recreate environment
python -m venv venv311
.\venv311\Scripts\activate
pip install -r requirements.txt
.\build_gsplat_final.bat

# 5. Verify
python test_gsplat_cuda.py
python app.py
```

---

## 13. Contact and Support

### 13.1 Issue Reporting

**Internal Issues**:
- Contact: Development team
- Email: <internal-email>
- Slack: #trellis-forge

**External Issues** (if open-sourced):
- GitHub Issues: https://github.com/<org>/trellis-forge/issues
- Documentation: https://<org>.github.io/trellis-forge
- Community: Discord / Forum

### 13.2 Escalation Path

**Level 1**: Application errors
- Check logs, restart application
- Reference: Section 8 (Troubleshooting)

**Level 2**: Environment issues
- Rebuild dependencies, verify CUDA
- Reference: Section 3 (Installation)

**Level 3**: Hardware issues
- NVIDIA driver update, GPU diagnostics
- Contact: IT support

**Level 4**: Model/architecture issues
- Contact: Development team
- May require code changes

---

**End of Deployment Guide**
