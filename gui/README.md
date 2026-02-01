# TRELLIS GUI

Full-stack web application providing intuitive control over TRELLIS text-to-3D and image-to-3D generation.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     React Frontend (Port 5173)              │
│  - Text-to-3D / Image-to-3D mode toggle                     │
│  - Full parameter control (seed, steps, CFG, etc.)          │
│  - Real-time job queue with progress tracking               │
│  - Preview images and file downloads                        │
└─────────────────────┬───────────────────────────────────────┘
                      │ HTTP + WebSocket
┌─────────────────────▼───────────────────────────────────────┐
│                   FastAPI Backend (Port 8000)               │
│  - Async job processing with background tasks               │
│  - TRELLIS pipeline caching for performance                 │
│  - File management (GLB, PLY, preview images)               │
│  - RESTful API with CORS support                            │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                    TRELLIS Pipeline                         │
│  - Text-to-3D: TrellisTextTo3DPipeline                      │
│  - Image-to-3D: TrellisImageTo3DPipeline                    │
│  - GLB export with PBR materials                            │
│  - Video rendering (optional)                               │
└─────────────────────────────────────────────────────────────┘
```

## Features

### Text-to-3D Generation
- Natural language prompts
- Full control over generation parameters:
  - Seed (reproducibility)
  - Sparse structure steps & CFG strength
  - SLat appearance steps & CFG strength
  - Mesh simplification ratio
  - Texture resolution (512/1024/2048)

### Image-to-3D Generation
- Upload any image (PNG, JPG, etc.)
- Same parameter control as text-to-3D
- Preview uploaded image before generation

### Job Management
- Real-time job queue display
- Progress tracking (0-100%)
- Status indicators (queued, processing, completed, failed)
- Job history with prompt/image preview
- Delete jobs
- Download outputs (GLB, PLY)

### Output Files
- **GLB**: 3D model with PBR materials (viewable in Blender, gltf-viewer.donmccurdy.com)
- **PLY**: Point cloud with Gaussian splats
- **Preview**: Rendered image of the 3D model

## Installation

### Backend Setup

1. **Navigate to backend directory**:
   ```bash
   cd spatial/gui/backend
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the server**:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

   The backend will be available at `http://localhost:8000`

### Frontend Setup

1. **Navigate to frontend directory**:
   ```bash
   cd spatial/gui/frontend
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Start development server**:
   ```bash
   npm run dev
   ```

   The frontend will be available at `http://localhost:5173`

## Usage

1. **Start both servers** (backend on 8000, frontend on 5173)

2. **Open browser** to `http://localhost:5173`

3. **Select mode**:
   - **Text-to-3D**: Enter a text prompt describing the 3D object
   - **Image-to-3D**: Upload an image to convert to 3D

4. **Adjust parameters** (optional):
   - Click "Advanced Settings" to reveal all controls
   - Modify seed, steps, CFG strength, simplification, texture size

5. **Generate**:
   - Click "Generate 3D Model"
   - Job appears in queue with "queued" status
   - Status updates to "processing" with progress bar
   - Once complete, preview image and download buttons appear

6. **Download outputs**:
   - Click "GLB" to download 3D model with textures
   - Click "PLY" to download Gaussian splat point cloud

## API Reference

### POST `/api/generate/text`
Generate 3D model from text prompt.

**Request body**:
```json
{
  "prompt": "A wooden chair shaped like an avocado",
  "seed": 42,
  "sparse_steps": 12,
  "sparse_cfg": 7.5,
  "slat_steps": 12,
  "slat_cfg": 7.5,
  "simplify": 0.95,
  "texture_size": 1024
}
```

**Response**:
```json
{
  "job_id": "uuid-string"
}
```

### POST `/api/generate/image`
Generate 3D model from uploaded image.

**Form data**:
- `file`: Image file (PNG, JPG, etc.)
- `seed`: 42
- `sparse_steps`: 12
- `sparse_cfg`: 7.5
- `slat_steps`: 12
- `slat_cfg`: 7.5
- `simplify`: 0.95
- `texture_size`: 1024

**Response**:
```json
{
  "job_id": "uuid-string"
}
```

### GET `/api/jobs`
List all jobs.

**Response**:
```json
{
  "jobs": [
    {
      "job_id": "uuid",
      "type": "text_to_3d",
      "status": "completed",
      "progress": 1.0,
      "message": "Exporting GLB",
      "outputs": {
        "glb": "path/to/file.glb",
        "ply": "path/to/file.ply",
        "preview": "path/to/preview.png"
      },
      "error": null,
      "created_at": "2024-01-19T12:00:00",
      "updated_at": "2024-01-19T12:05:00",
      "params": { "prompt": "...", ... }
    }
  ]
}
```

### GET `/api/jobs/{job_id}`
Get specific job status.

**Response**: Same as single job object above.

### DELETE `/api/jobs/{job_id}`
Delete a job and its output files.

**Response**:
```json
{
  "message": "Job deleted successfully"
}
```

### GET `/api/download/{job_id}/{file_type}`
Download output file (glb, ply, or preview).

**Response**: File download with appropriate content-type.

## Parameter Guide

### Seed
- Integer value for reproducible generation
- Same seed + prompt = same output
- Default: 42

### Sparse Structure
- **Steps**: Number of denoising steps for geometry generation (4-50)
  - Higher = more detailed geometry, slower generation
  - Default: 12
- **CFG Strength**: Classifier-free guidance for prompt adherence (1.0-15.0)
  - Higher = stronger prompt following, less diversity
  - Default: 7.5

### SLat Appearance
- **Steps**: Number of steps for appearance/texture generation (4-50)
  - Higher = better textures, slower generation
  - Default: 12
- **CFG Strength**: Guidance strength for appearance (1.0-15.0)
  - Default: 7.5

### Mesh Simplification
- Ratio of vertices to keep (0.0-1.0)
- 1.0 = no simplification (most detail, largest file)
- 0.5 = 50% reduction
- Default: 0.95 (5% reduction, good balance)

### Texture Size
- Resolution of output textures (512, 1024, 2048)
- Higher = better quality, larger files
- Default: 1024

## File Structure

```
gui/
├── README.md                  # This file
├── backend/
│   ├── main.py                # FastAPI application
│   ├── requirements.txt       # Python dependencies
│   └── outputs/               # Generated files (created at runtime)
└── frontend/
    ├── package.json           # Node dependencies
    ├── vite.config.ts         # Vite configuration
    ├── tsconfig.json          # TypeScript configuration
    ├── index.html             # HTML entry point
    └── src/
        ├── main.tsx           # React entry point
        ├── App.tsx            # Main application component
        ├── App.css            # Application styles
        └── index.css          # Global styles
```

## Troubleshooting

### Backend won't start
- Check Python environment has all dependencies installed
- Verify TRELLIS is properly installed in parent directory
- Check port 8000 is not in use

### Frontend won't start
- Run `npm install` to install dependencies
- Check Node.js version (requires 18+)
- Check port 5173 is not in use

### Generation fails
- Check backend logs for TRELLIS errors
- Verify CUDA is available (for GPU acceleration)
- Check output directory is writable
- Try reducing texture_size or simplify parameters

### Files won't download
- Check outputs directory exists and has files
- Verify job completed successfully (status = "completed")
- Check browser console for CORS errors

## Performance

- **Text-to-3D**: ~2-5 minutes on RTX 4090 (default settings)
- **Image-to-3D**: ~3-7 minutes on RTX 4090 (default settings)
- **Memory**: ~8-12GB VRAM required
- **File sizes**:
  - GLB: 100KB - 20MB (depends on complexity)
  - PLY: 5MB - 50MB (Gaussian splats)
  - Preview: ~100KB

## Development

### Backend hot reload
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend hot reload
```bash
npm run dev
```

### Build for production
```bash
cd frontend
npm run build
```

Output in `frontend/dist/` can be served with any static file server.

## License

Same as TRELLIS (MIT License).

## Credits

Built on Microsoft TRELLIS.2:
- Paper: https://arxiv.org/abs/2412.01506
- Code: https://github.com/microsoft/TRELLIS
- Model: https://huggingface.co/microsoft/TRELLIS-text-xlarge
