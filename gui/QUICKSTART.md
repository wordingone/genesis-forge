# TRELLIS GUI - Quick Start Guide

## Prerequisites

- ✅ TRELLIS installed and working
- ✅ Python 3.11 with venv at `B:\M\ArtificialArchitecture\spatial\venv311`
- ✅ Node.js 18+ installed
- ✅ CUDA 12.8 (for GPU acceleration)
- ✅ All dependencies installed (FastAPI, React, etc.)

## Starting the GUI (2 Steps)

### Step 1: Start the Backend Server

Open **Terminal 1** (PowerShell):

```powershell
cd B:\M\ArtificialArchitecture\spatial\gui\backend
B:\M\ArtificialArchitecture\spatial\venv311\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

**Expected output**:
```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [12345] using StatReload
INFO:     Started server process [67890]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

✅ Backend is ready when you see "Application startup complete"

### Step 2: Start the Frontend Server

Open **Terminal 2** (PowerShell):

```powershell
cd B:\M\ArtificialArchitecture\spatial\gui\frontend
npm run dev
```

**Expected output**:
```
  VITE v6.4.1  ready in 234 ms

  ➜  Local:   http://localhost:5173/
  ➜  Network: use --host to expose
  ➜  press h + enter to show help
```

✅ Frontend is ready - open browser to **http://localhost:5173/**

## Using the GUI

### Text-to-3D Generation

1. **Click "Text-to-3D" mode** (default)

2. **Enter a prompt** in the text area:
   ```
   A wooden chair shaped like an avocado
   ```

3. *(Optional)* **Open Advanced Settings**:
   - Click "Advanced Settings" button
   - Adjust parameters:
     - **Seed**: 42 (for reproducibility)
     - **Sparse Steps**: 12 (geometry detail)
     - **Sparse CFG**: 7.5 (prompt adherence)
     - **SLat Steps**: 12 (texture quality)
     - **SLat CFG**: 7.5 (appearance adherence)
     - **Simplification**: 95% (mesh reduction)
     - **Texture Size**: 1024 (resolution)

4. **Click "Generate 3D Model"**

5. **Watch progress** in the Generation Queue panel:
   - Status: "queued" → "processing" → "completed"
   - Progress bar shows 0-100%
   - Messages update during generation

6. **Download results** when complete:
   - Click **"GLB"** to download 3D model with textures
   - Click **"PLY"** to download Gaussian splat point cloud
   - View preview image inline

### Image-to-3D Generation

1. **Click "Image-to-3D" mode**

2. **Upload an image**:
   - Click the upload area
   - Select a PNG, JPG, or other image file
   - Preview appears

3. **Adjust parameters** (same as text-to-3D)

4. **Click "Generate 3D Model"**

5. **Download results** when complete

## Expected Performance

| Setting | RTX 4090 | RTX 3090 | RTX 4070 |
|---------|----------|----------|----------|
| Text-to-3D (default) | 2-3 min | 4-6 min | 6-10 min |
| Image-to-3D (default) | 3-5 min | 6-9 min | 8-14 min |
| High quality (steps=24) | 5-7 min | 10-15 min | 15-25 min |

**Memory usage**: ~8-12GB VRAM

**File sizes**:
- GLB: 100KB - 20MB (depends on complexity)
- PLY: 5MB - 50MB (Gaussian splats)
- Preview: ~100KB

## Viewing 3D Models

### Online Viewers (No Installation)
- **glTF Viewer**: https://gltf-viewer.donmccurdy.com/
  - Drag and drop GLB file
  - Interactive rotation, zoom, inspect materials

- **3D Viewer**: https://3dviewer.net/
  - Upload GLB or PLY
  - Multiple rendering modes

### Desktop Applications
- **Blender** (Free): File → Import → glTF 2.0
- **Meshlab** (Free): Open PLY files
- **Unity**: Drag GLB into Assets folder
- **Unreal Engine**: Import as Static Mesh

## Troubleshooting

### Backend won't start

**Error**: `ModuleNotFoundError: No module named 'trellis'`
```powershell
# Verify TRELLIS path
cd B:\M\ArtificialArchitecture\spatial\gui\backend
python test_startup.py
```

**Error**: `Port 8000 already in use`
```powershell
# Use different port
uvicorn main:app --host 127.0.0.1 --port 8001
```
Then update frontend `vite.config.ts` proxy target to `http://localhost:8001`

### Frontend won't start

**Error**: `npm: command not found`
- Install Node.js 18+ from https://nodejs.org/

**Error**: `Dependencies not found`
```powershell
cd B:\M\ArtificialArchitecture\spatial\gui\frontend
npm install
```

### Generation fails

**Error**: `CUDA out of memory`
- Reduce texture_size to 512
- Close other GPU applications
- Reduce sparse_steps and slat_steps to 8

**Error**: `Job status stuck at 'processing'`
- Check backend terminal for Python errors
- Restart backend server
- Check CUDA/PyTorch installation

### Downloads fail

**Error**: `404 Not Found`
- Verify job completed successfully (status = "completed")
- Check backend logs for file save errors
- Verify outputs directory exists and is writable

**Error**: `CORS error`
- Check backend is running on port 8000
- Verify frontend proxy in `vite.config.ts`
- Check browser console for details

## Advanced Usage

### Running Multiple Jobs

The backend processes jobs sequentially. You can queue multiple jobs:

1. Submit first job
2. Immediately submit second job (don't wait)
3. Both appear in queue
4. First job processes, then second

### Reproducible Results

To get identical outputs:

1. Set same **seed** value (e.g., 42)
2. Use same prompt or image
3. Use same parameter values
4. Output will be deterministic

### High-Quality Output

For publication-quality models:

```
Sparse Steps: 24
Sparse CFG: 9.0
SLat Steps: 24
SLat CFG: 9.0
Simplification: 98%
Texture Size: 2048
```

Warning: Takes 3-5x longer to generate

### Fast Preview

For quick iterations:

```
Sparse Steps: 6
Sparse CFG: 5.0
SLat Steps: 6
SLat CFG: 5.0
Simplification: 90%
Texture Size: 512
```

Warning: Lower quality output

## Stopping the Servers

### Stop Backend
Press `Ctrl+C` in Terminal 1 (backend)

### Stop Frontend
Press `Ctrl+C` in Terminal 2 (frontend)

## API Access (Advanced)

The backend provides a RESTful API. You can integrate it into your own applications:

```python
import requests

# Submit text-to-3D job
response = requests.post("http://localhost:8000/api/generate/text", json={
    "prompt": "A wooden chair shaped like an avocado",
    "seed": 42,
    "sparse_steps": 12,
    "sparse_cfg": 7.5,
    "slat_steps": 12,
    "slat_cfg": 7.5,
    "simplify": 0.95,
    "texture_size": 1024
})
job_id = response.json()["job_id"]

# Check status
status = requests.get(f"http://localhost:8000/api/jobs/{job_id}").json()
print(status["status"], status["progress"])

# Download GLB when complete
if status["status"] == "completed":
    with open("output.glb", "wb") as f:
        glb = requests.get(f"http://localhost:8000/api/download/{job_id}/glb")
        f.write(glb.content)
```

See `README.md` for full API documentation.

## Support

For issues with:
- **TRELLIS itself**: https://github.com/microsoft/TRELLIS/issues
- **This GUI**: Check console logs in both terminals for errors
- **CUDA/PyTorch**: Verify installation with `python -c "import torch; print(torch.cuda.is_available())"`

## Next Steps

- Experiment with different prompts
- Try image-to-3D with your own images
- Adjust parameters to understand their effects
- Import GLB files into Blender for further editing
- Use API for batch processing
