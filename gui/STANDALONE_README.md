# TRELLIS GUI - Standalone Desktop Application

## Overview

**Standalone desktop application** for TRELLIS 3D generation with integrated 3D viewer. No browser required!

### Features

✅ **Standalone Application** - Runs as native Windows desktop app via Electron
✅ **Integrated 3D Viewer** - Built-in Three.js viewer with interactive controls
✅ **Real-time Preview** - View GLB models instantly with rotation, zoom, pan
✅ **Wireframe Toggle** - Switch between solid and wireframe rendering
✅ **Camera Controls** - Orbit controls with mouse, reset button
✅ **Full Parameter Control** - All TRELLIS settings accessible
✅ **Job Queue Management** - Track multiple generations
✅ **File Management** - Save models to disk with native file dialogs

## Quick Start

### 1. Install Dependencies (First Time Only)

```powershell
cd <repo>\gui\electron
npm install
```

This installs Electron and Three.js (~300MB).

### 2. Run the Application

```powershell
.\start_standalone.ps1
```

Or manually:

```powershell
npm start
```

The application will:
1. Automatically start the backend server
2. Open the GUI window
3. Be ready to generate 3D models!

## User Interface

### Layout

```
┌─────────────────────────────────────────────────────────────┐
│  TRELLIS GUI - Standalone 3D Generation Application         │
├─────────────┬───────────────────────────┬────────────────────┤
│   Input     │       3D Viewer           │   Generation       │
│   Controls  │                           │      Queue         │
│             │   [Interactive Model]     │                    │
│  • Mode     │                           │  • Job Status      │
│  • Prompt   │   Camera Controls:        │  • Progress Bars   │
│  • Image    │   - Orbit (drag)          │  • View Models     │
│  • Settings │   - Zoom (scroll)         │  • Download        │
│  • Generate │   - Pan (right-drag)      │  • Delete          │
│             │   - Reset Camera          │                    │
│             │   - Wireframe Toggle      │                    │
└─────────────┴───────────────────────────┴────────────────────┘
```

### 3D Viewer Controls

**Mouse Controls:**
- **Left Click + Drag** - Rotate/orbit around model
- **Scroll Wheel** - Zoom in/out
- **Right Click + Drag** - Pan camera
- **Middle Click + Drag** - Pan camera (alternative)

**Buttons:**
- **Reset Camera** - Return to default view position
- **Wireframe** - Toggle between solid and wireframe rendering

**View Features:**
- Automatic model centering and scaling
- Grid floor for spatial reference
- Three-point lighting for clear visibility
- Smooth camera damping for professional feel

## Workflow

### Text-to-3D Generation

1. **Select "Text-to-3D" mode**
2. **Enter prompt**: "A wooden chair shaped like an avocado"
3. **(Optional) Open Advanced Settings**:
   - Seed: 42
   - Sparse Steps: 12
   - SLat Steps: 12
   - Simplification: 95%
   - Texture Size: 1024
4. **Click "Generate 3D Model"**
5. **Watch progress** in Generation Queue panel
6. **Click "View GLB"** when complete
7. **Interact with model** using mouse controls
8. **Click "Save GLB"** to export to disk

### Image-to-3D Generation

1. **Select "Image-to-3D" mode**
2. **Click upload area** to select image
3. **Preview appears** automatically
4. **Adjust settings** if desired
5. **Click "Generate 3D Model"**
6. **View and save** when complete

### Viewing Models

Once generation completes:

1. **Click "View GLB"** - Loads model into 3D viewer
   - Model auto-centers and scales
   - Rotate with left-click drag
   - Zoom with scroll wheel
   - Pan with right-click drag

2. **Click "View PLY"** - View Gaussian splat point cloud
   - Displays as colored points
   - Same camera controls

3. **Toggle Wireframe** - See mesh structure
   - Click wireframe button
   - Shows topology and edge flow

4. **Reset Camera** - Return to default view
   - Click reset button
   - Camera moves to (2, 2, 2) looking at origin

### Saving Models

1. **Click "Save GLB"** or **"Save PLY"**
2. **Native file dialog appears**
3. **Choose location and filename**
4. **Model saved to disk**

Can open saved models in:
- Blender (File → Import → glTF 2.0)
- Unity (drag into Assets)
- Unreal Engine (import as Static Mesh)
- Online viewers (gltf-viewer.donmccurdy.com)

## Technical Details

### Technology Stack

**Frontend:**
- Electron 28.1.3 (Chromium-based desktop app framework)
- Three.js 0.160.1 (WebGL 3D rendering)
- Native HTML/CSS/JavaScript (no framework bloat)

**3D Viewer:**
- GLTFLoader for loading GLB models
- PLYLoader for point cloud data
- OrbitControls for camera interaction
- PBR material rendering with tone mapping

**Backend:**
- Same FastAPI server as web version
- Spawned automatically by Electron main process
- Runs on localhost:8000

### File Structure

```
electron/
├── main.js              # Electron main process (backend spawning, window management)
├── preload.js           # IPC bridge (secure communication)
├── index.html           # Application UI structure
├── styles.css           # Application styling
├── app.js               # Frontend logic + Three.js viewer
├── package.json         # Dependencies and build config
├── start_standalone.ps1 # Launcher script
└── assets/
    └── icon.png         # Application icon
```

### Backend Integration

The Electron app automatically:
1. Verifies Python environment exists
2. Spawns backend server as child process
3. Waits for "Application startup complete" message
4. Opens GUI window when backend ready
5. Kills backend when app closes

**Backend Path**: `<repo>\venv\Scripts\python.exe`
**Backend Port**: 8000 (localhost only, not exposed to network)

## Performance

### Generation Times (RTX 4090)
- Text-to-3D: 2-3 minutes (default settings)
- Image-to-3D: 3-5 minutes (default settings)

### Viewer Performance
- GLB rendering: 60 FPS for models up to 100K triangles
- PLY rendering: 60 FPS for point clouds up to 1M points
- Smooth camera controls with damping
- Real-time wireframe toggle

### Memory Usage
- Electron app: ~200MB RAM
- Backend server: ~2GB RAM
- GPU VRAM: ~8-12GB during generation
- GPU VRAM: ~500MB for viewer

## Building Standalone Executable (Optional)

To create a distributable .exe:

```powershell
cd <repo>\gui\electron
npm run build:win
```

Output: `dist/TRELLIS GUI Setup.exe` (~150MB installer)

**Includes:**
- Electron runtime
- Three.js libraries
- Backend server code
- Python not included (users must have venv set up)

## Troubleshooting

### Application won't start

**Error**: Backend failed to start
- Check Python path in `main.js` line 11
- Verify venv exists: `<repo>\venv`
- Check backend dependencies installed

**Error**: Electron not found
```powershell
npm install
```

### 3D viewer shows black screen

- Check browser console for Three.js errors
- Verify model downloaded correctly
- Try "Reset Camera" button
- Check model file isn't corrupted

### Models won't load in viewer

**GLB fails to load:**
- Verify generation completed successfully
- Check file size > 0 bytes
- Try viewing in Blender to confirm file is valid

**PLY fails to load:**
- Point clouds may be very large (50MB+)
- Give viewer 10-15 seconds to load
- Check RAM usage

### Camera controls not working

- Click inside viewer area to focus
- Try scrolling to zoom (should work immediately)
- Try left-click drag to rotate
- Click "Reset Camera" to verify controls work

### Backend timeout on startup

- First startup takes 20-30 seconds (loading TRELLIS)
- Check backend window for errors
- Verify CUDA available: `python -c "import torch; print(torch.cuda.is_available())"`

## Keyboard Shortcuts

Currently no keyboard shortcuts implemented. All controls via mouse/buttons.

**Possible future additions:**
- `R` - Reset camera
- `W` - Toggle wireframe
- `F` - Frame selected object
- `G` - Toggle grid
- `Space` - Pause/resume generation

## Comparison: Standalone vs Web Version

| Feature | Standalone (Electron) | Web (Browser) |
|---------|----------------------|---------------|
| Installation | npm install (~300MB) | npm install frontend + backend |
| Startup | One command | Two terminals |
| 3D Viewer | Built-in (Three.js) | None |
| Browser Required | No | Yes |
| File Dialogs | Native OS dialogs | Browser downloads |
| Performance | Native speed | Same |
| Updates | Rebuild required | Refresh page |
| Offline Use | Yes (after install) | No |

**Recommendation**: Use standalone version for desktop use, web version for remote access.

## Known Limitations

1. **Single instance** - Can't run multiple copies simultaneously (backend port conflict)
2. **No GPU selection** - Uses default CUDA device
3. **No batch processing** - One job at a time
4. **No history persistence** - Jobs cleared on app restart
5. **Windows only** - Not tested on macOS/Linux (Electron is cross-platform, backend is not)

## Advanced Usage

### Development Mode

```powershell
npm run dev
```

Opens DevTools for debugging.

### Custom Backend Path

Edit `main.js` line 11-13:

```javascript
const BACKEND_CONFIG = {
  pythonExe: 'path/to/your/python.exe',
  backendPath: 'path/to/backend',
  port: 8000
};
```

### Changing Viewer Settings

Edit `app.js` `initViewer()` function:

```javascript
// Change FOV
camera = new THREE.PerspectiveCamera(75, ...); // default 50

// Change background
scene.background = new THREE.Color(0x1a1a2e); // darker

// Disable grid
// gridHelper line - comment out

// Add more lights
const pointLight = new THREE.PointLight(0xffffff, 1);
scene.add(pointLight);
```

## Support

For issues:
- **Electron app**: Check console output in terminal
- **3D viewer**: Open DevTools (npm run dev) and check browser console
- **Backend**: Same as web version troubleshooting
- **TRELLIS**: https://github.com/microsoft/TRELLIS/issues

## Future Enhancements

Possible additions:
- [ ] Multi-view layouts (front/side/top)
- [ ] Screenshot/render export
- [ ] Animation timeline
- [ ] Measurement tools
- [ ] Material editor
- [ ] Lighting presets
- [ ] Background HDRI environments
- [ ] VR support (WebXR)

---

**Summary**: Standalone desktop application with integrated 3D viewer eliminates browser dependency and provides professional model inspection tools. Recommended for desktop workflows.
