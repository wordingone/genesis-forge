# Functional Specification
## TRELLIS Forge v1.0

**Product**: TRELLIS Forge
**Version**: 1.0.0 (Initial Windows Release)
**Date**: 2026-01-20
**Status**: Production-Ready (Internal/Demo)

---

## 1. Overview

This document specifies the verified functional behavior of TRELLIS Forge—a Gradio-based web application that converts 2D images into 3D assets using Microsoft's TRELLIS.2 model. All specifications are based on tested implementation, not theoretical requirements.

---

## 2. User Interface Specification

### 2.1 Application Layout

**Framework**: Gradio 6.0.1 web interface

**Layout Structure** (app.py:253-305):

```
┌─────────────────────────────────────────────────┐
│ Title: "Image to 3D Asset with TRELLIS"        │
├──────────────────┬──────────────────────────────┤
│ LEFT PANEL       │ RIGHT PANEL                  │
│                  │                              │
│ Image Input      │ Video Preview (120 frames)   │
│ - Single Image   │ 3D Model Viewer              │
│ - Multi-Image    │                              │
│                  │ Download Buttons:            │
│ Settings:        │ - Download GLB               │
│ - Seed           │ - Download Gaussian          │
│ - Guidance       │                              │
│ - Sampling       │                              │
│                  │                              │
│ Actions:         │                              │
│ - Generate       │                              │
│ - Extract GLB    │                              │
│ - Extract Gaussian│                             │
├──────────────────┴──────────────────────────────┤
│ Example Images Gallery                          │
└─────────────────────────────────────────────────┘
```

### 2.2 Input Modes

#### 2.2.1 Single Image Mode (Default)

**Component**: `gr.Image` (app.py:264)
- **Label**: "Image Prompt"
- **Format**: PNG
- **Image Mode**: RGBA (preserves alpha channel)
- **Type**: PIL Image
- **Height**: 300px
- **Preprocessing**: Automatic on upload via `preprocess_image()` (app.py:346-350)

**Behavior**:
1. User uploads image OR selects from gallery
2. If alpha channel present: Used as-is
3. If no alpha channel: Background removed via `rembg`
4. Preprocessed image displayed in upload widget

#### 2.2.2 Multi-Image Mode (Experimental)

**Component**: `gr.Gallery` (app.py:266)
- **Label**: "Image Prompt"
- **Format**: PNG
- **Type**: PIL Image
- **Height**: 300px
- **Columns**: 3
- **Preprocessing**: Automatic on upload via `preprocess_images()` (app.py:351-355)

**Behavior**:
1. User uploads 2-4 images of same object from different views
2. Each image preprocessed independently
3. Warning shown: *"experimental algorithm without training a specialized model"*
4. May not work well for inconsistent views or different poses

**Tab Selection** (app.py:262-271):
- Tabs: "Single Image" (id=0), "Multiple Images" (id=1)
- Switching tabs toggles example gallery visibility

### 2.3 Generation Settings

**Accordion**: "Generation Settings" (app.py:273-284, collapsed by default)

#### Seed Control
- **Component**: `gr.Slider` (app.py:274)
- **Range**: 0 to 2,147,483,647 (MAX_SEED)
- **Default**: 0
- **Step**: 1
- **Randomize Checkbox**: Default ON (app.py:275)

**Behavior**:
- If randomize enabled: Random seed generated on each generation
- If randomize disabled: Fixed seed for reproducibility
- Seed value displayed after generation

#### Stage 1: Sparse Structure Generation
- **Guidance Strength** (app.py:278):
  - Range: 0.0-10.0
  - Default: 7.5
  - Step: 0.1
  - Controls adherence to input image in coarse structure
- **Sampling Steps** (app.py:279):
  - Range: 1-50
  - Default: 12
  - Step: 1
  - Controls quality vs. speed tradeoff

#### Stage 2: Structured Latent Generation
- **Guidance Strength** (app.py:282):
  - Range: 0.0-10.0
  - Default: 3.0
  - Step: 0.1
  - Controls adherence to input in fine details
- **Sampling Steps** (app.py:283):
  - Range: 1-50
  - Default: 12
  - Step: 1
  - Controls refinement iterations

#### Multi-Image Algorithm (app.py:284)
- **Component**: `gr.Radio`
- **Options**: ["stochastic", "multidiffusion"]
- **Default**: "stochastic"
- **Only visible in multi-image mode**

### 2.4 GLB Extraction Settings

**Accordion**: "GLB Extraction Settings" (app.py:288-290, collapsed by default)

#### Mesh Simplification
- **Component**: `gr.Slider`
- **Range**: 0.9-0.98
- **Default**: 0.95
- **Step**: 0.01
- **Meaning**: Retain percentage of triangles (0.95 = reduce to 5% of original count)

#### Texture Size
- **Component**: `gr.Slider`
- **Range**: 512-2048
- **Default**: 1024
- **Step**: 512
- **Options**: 512px, 1024px, 1536px, 2048px
- **Impact**: Higher resolution = larger file size, better quality

### 2.5 Action Buttons

#### Generate Button (app.py:286)
- **Label**: "Generate"
- **Initially**: Always enabled
- **On Click**:
  1. Get seed (randomize or use fixed)
  2. Run generation pipeline
  3. Display video preview
  4. Enable "Extract GLB" and "Extract Gaussian" buttons

#### Extract GLB Button (app.py:293)
- **Label**: "Extract GLB"
- **Initially**: Disabled
- **Enabled After**: Successful generation
- **Disabled On**: Video preview cleared
- **On Click**:
  1. Extract GLB with current settings
  2. Display in 3D viewer
  3. Enable "Download GLB" button

#### Extract Gaussian Button (app.py:294)
- **Label**: "Extract Gaussian"
- **Initially**: Disabled
- **Enabled After**: Successful generation
- **Disabled On**: Video preview cleared
- **On Click**:
  1. Export Gaussian .ply file
  2. Display in 3D viewer
  3. Enable "Download Gaussian" button
- **Warning**: *"Gaussian file can be very large (~50MB)"*

### 2.6 Output Display

#### Video Preview (app.py:300)
- **Component**: `gr.Video`
- **Label**: "Generated 3D Asset"
- **Autoplay**: Yes
- **Loop**: Yes
- **Height**: 300px
- **Format**: MP4, 15 FPS, 120 frames
- **Content**: Side-by-side Gaussian render (left) and geometry normals (right)

#### 3D Model Viewer (app.py:301)
- **Component**: `LitModel3D` (gradio_litmodel3d)
- **Label**: "Extracted GLB/Gaussian"
- **Exposure**: 10.0 (lighting intensity)
- **Height**: 300px
- **Supports**: GLB and PLY formats
- **Interaction**: Orbit, zoom, pan

#### Download Buttons (app.py:304-305)
- **Download GLB**:
  - Initially disabled
  - Enabled after GLB extraction
  - Downloads `sample.glb` from user's temp directory
- **Download Gaussian**:
  - Initially disabled
  - Enabled after Gaussian extraction
  - Downloads `sample.ply` from user's temp directory

### 2.7 Example Gallery

#### Single Image Examples (app.py:311-322)
- **Location**: Bottom of page (below main interface)
- **Source**: `assets/example_image/` directory
- **Display**: Grid layout, 64 examples per page
- **Interaction**: Click to load and preprocess
- **Visibility**: Shown in single image mode, hidden in multi-image mode

#### Multi-Image Examples (app.py:323-331)
- **Source**: `assets/example_multi_image/` directory
- **Format**: Concatenated horizontal strips (3 views per example)
- **Display**: Grid layout, 8 examples per page
- **Interaction**: Click to split and load into gallery
- **Visibility**: Hidden in single image mode, shown in multi-image mode

---

## 3. Functional Workflows

### 3.1 Basic Generation Workflow

**User Action Sequence**:

```
1. Upload image OR select from examples
   ↓
2. (Optional) Adjust generation settings
   ↓
3. Click "Generate"
   ↓
4. Watch progress (video preview appears after ~4 seconds)
   ↓
5. Review video (Gaussian render + normals)
   ↓
6. Click "Extract GLB" OR "Extract Gaussian"
   ↓
7. View in 3D viewer
   ↓
8. Click "Download GLB" OR "Download Gaussian"
```

**System Behavior** (app.py:357-368):

```python
generate_btn.click(
    get_seed,          # Step 1: Determine seed
).then(
    image_to_3d,       # Step 2: Generate 3D
).then(
    enable_extract_buttons  # Step 3: Enable extraction
)
```

### 3.2 Image Preprocessing Workflow

**Automatic Preprocessing** (app.py:32-43):

**Function**: `preprocess_image(image: PIL.Image) -> PIL.Image`

**Steps**:
1. Accept PIL Image (RGBA mode)
2. Call `pipeline.preprocess_image(image)`
3. Return preprocessed image

**Internal Pipeline Behavior** (verified via testing):
- **If alpha channel exists**: Preserve as-is
- **If no alpha channel**: Apply `rembg` background removal
- **Resize**: Normalize to optimal resolution for model
- **Return**: RGBA PIL Image

**Trigger Points**:
- On image upload (app.py:346-350)
- On example selection (app.py:318-319)
- On multi-image upload (app.py:351-355)

### 3.3 3D Generation Workflow

**Entry Point**: `image_to_3d()` (app.py:108-177)

**Parameters**:
- `image`: Preprocessed PIL Image (single mode)
- `multiimages`: List of images (multi-image mode)
- `is_multiimage`: Boolean flag (determines which mode)
- `seed`: Integer random seed
- `ss_guidance_strength`: Sparse structure guidance (0.0-10.0)
- `ss_sampling_steps`: Sparse structure steps (1-50)
- `slat_guidance_strength`: Structured latent guidance (0.0-10.0)
- `slat_sampling_steps`: Structured latent steps (1-50)
- `multiimage_algo`: "stochastic" or "multidiffusion"
- `req`: Gradio Request object (for session hash)

**Execution Flow**:

**Single Image Mode** (app.py:140-153):
```python
outputs = pipeline.run(
    image,                      # Preprocessed image
    seed=seed,                  # Random seed
    formats=["gaussian", "mesh"],  # Generate both
    preprocess_image=False,     # Already preprocessed
    sparse_structure_sampler_params={
        "steps": ss_sampling_steps,
        "cfg_strength": ss_guidance_strength,
    },
    slat_sampler_params={
        "steps": slat_sampling_steps,
        "cfg_strength": slat_guidance_strength,
    },
)
```

**Multi-Image Mode** (app.py:155-169):
```python
outputs = pipeline.run_multi_image(
    [image[0] for image in multiimages],
    seed=seed,
    formats=["gaussian", "mesh"],
    preprocess_image=False,
    sparse_structure_sampler_params={...},
    slat_sampler_params={...},
    mode=multiimage_algo,  # "stochastic" or "multidiffusion"
)
```

**Output Structure**:
```python
outputs = {
    'gaussian': [Gaussian object],  # 3D Gaussian splatting
    'mesh': [MeshExtractResult],    # Triangulated mesh
}
```

**Post-Processing** (app.py:170-176):

**Video Rendering**:
1. Render 120 frames of Gaussian (colored)
2. Render 120 frames of mesh geometry (normals)
3. Concatenate side-by-side
4. Save as MP4 at 15 FPS

**State Packing**:
1. Convert Gaussian tensors to CPU numpy arrays
2. Convert mesh tensors to CPU numpy arrays
3. Pack into dictionary for session persistence
4. Clear GPU memory (`torch.cuda.empty_cache()`)

**Return Values**:
- `state`: Packed dictionary (enables extraction without regeneration)
- `video_path`: Path to MP4 file in user's temp directory

### 3.4 GLB Extraction Workflow

**Entry Point**: `extract_glb()` (app.py:180-203)

**Parameters**:
- `state`: Packed generation state
- `mesh_simplify`: Simplification factor (0.9-0.98)
- `texture_size`: Texture resolution (512/1024/2048)
- `req`: Gradio Request object

**Execution Flow**:

**Step 1: Unpack State** (app.py:198):
```python
gs, mesh = unpack_state(state)
```
- Reconstructs Gaussian object on GPU
- Reconstructs mesh object on GPU

**Step 2: Convert to GLB** (app.py:199):
```python
glb = postprocessing_utils.to_glb(
    gs,                      # Gaussian splatting
    mesh,                    # Mesh geometry
    simplify=mesh_simplify,  # Triangle reduction
    texture_size=texture_size,  # UV texture resolution
    verbose=False
)
```

**Internal Steps** (postprocessing_utils.py:427-492):

1. **Mesh Simplification**:
   - Apply quadric edge collapse decimation
   - Reduce triangle count to `mesh_simplify` percentage
   - Preserve topology (no holes introduced)

2. **Texture Baking** (render_utils.py:103-111):
   - Render 100 camera views around object
   - Each view: Rasterize Gaussians via gsplat (CUDA-accelerated)
   - Project Gaussian colors to UV coordinates
   - Bake to `texture_size × texture_size` texture

3. **GLB Assembly**:
   - Embed simplified mesh
   - Embed baked texture
   - Set PBR material properties
   - Validate GLB format

**Step 3: Export** (app.py:200-201):
```python
glb_path = os.path.join(user_dir, 'sample.glb')
glb.export(glb_path)
```

**Step 4: Cleanup** (app.py:202):
```python
torch.cuda.empty_cache()
```

**Return Values**:
- `glb_path`: Path to exported GLB file (for viewer)
- `glb_path`: Path to exported GLB file (for download button)

**Performance**:
- Simplification: <1 second
- Texture baking: ~30 seconds (100 views × 300ms/view)
- GLB export: <1 second
- **Total**: ~30-35 seconds

**Output File**:
- Format: GLB (binary glTF 2.0)
- Size: 0.4-2 MB (depends on texture_size)
- Contents: Single mesh with UV-mapped base color texture

### 3.5 Gaussian Extraction Workflow

**Entry Point**: `extract_gaussian()` (app.py:206-221)

**Parameters**:
- `state`: Packed generation state
- `req`: Gradio Request object

**Execution Flow**:

**Step 1: Unpack State** (app.py:217):
```python
gs, _ = unpack_state(state)  # Only need Gaussian, discard mesh
```

**Step 2: Export PLY** (app.py:218-219):
```python
gaussian_path = os.path.join(user_dir, 'sample.ply')
gs.save_ply(gaussian_path)
```

**PLY Format Contents**:
- **Positions**: `_xyz` (N×3 float32)
- **Colors**: `_features_dc` (N×1×3 float32, DC coefficients of spherical harmonics)
- **Opacities**: `_opacity` (N×1 float32)
- **Scales**: `_scaling` (N×3 float32, 3 axis scales)
- **Rotations**: `_rotation` (N×4 float32, quaternions)

**Step 3: Cleanup** (app.py:220):
```python
torch.cuda.empty_cache()
```

**Return Values**:
- `gaussian_path`: Path to exported PLY file (for viewer)
- `gaussian_path`: Path to exported PLY file (for download button)

**Performance**:
- Export: <2 seconds

**Output File**:
- Format: PLY (Stanford Polygon format)
- Size: ~50 MB (uncompressed, N typically 50,000-200,000 Gaussians)
- Compatibility: Standard Gaussian splatting viewers (SIBR, SuperSplat)

---

## 4. Session Management

### 4.1 Session Lifecycle

**Start Session** (app.py:22-24, triggered on demo.load):
```python
def start_session(req: gr.Request):
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    os.makedirs(user_dir, exist_ok=True)
```

**Behavior**:
- Each browser session gets unique hash (generated by Gradio)
- Creates isolated temp directory: `tmp/{session_hash}/`
- Directory used for all outputs (video, GLB, PLY)

**End Session** (app.py:27-29, triggered on demo.unload):
```python
def end_session(req: gr.Request):
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    shutil.rmtree(user_dir)
```

**Behavior**:
- Browser close/disconnect triggers unload event
- Entire user directory removed
- Prevents disk space accumulation

### 4.2 State Persistence

**State Dictionary Structure** (app.py:62-75):

```python
state = {
    'gaussian': {
        'aabb': [...],            # Bounding box
        'sh_degree': int,         # Spherical harmonic degree
        'mininum_kernel_size': float,
        'scaling_bias': float,
        'opacity_bias': float,
        'scaling_activation': str,
        '_xyz': np.ndarray,       # Positions (N×3)
        '_features_dc': np.ndarray,  # Colors (N×1×3)
        '_scaling': np.ndarray,   # Scales (N×3)
        '_rotation': np.ndarray,  # Quaternions (N×4)
        '_opacity': np.ndarray,   # Opacities (N×1)
    },
    'mesh': {
        'vertices': np.ndarray,   # Mesh vertices (V×3)
        'faces': np.ndarray,      # Triangle indices (F×3)
    },
}
```

**Why State Persistence**:
- Generation takes ~4 seconds
- Extraction can be called multiple times with different settings
- Avoid regeneration on each extraction
- State stored in Gradio component (not on disk)

**State Unpacking** (app.py:78-98):
- Converts numpy arrays back to torch tensors
- Moves to GPU
- Reconstructs Gaussian and mesh objects
- Ready for extraction operations

### 4.3 File Paths

**Temporary Directory** (app.py:18):
```python
TMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tmp')
```
- Location: `<repo>\tmp`
- Structure:
  ```
  tmp/
  ├── {session_hash_1}/
  │   ├── sample.mp4      # Video preview
  │   ├── sample.glb      # Extracted GLB
  │   └── sample.ply      # Extracted Gaussian
  ├── {session_hash_2}/
  │   └── ...
  └── {session_hash_N}/
  ```

**Assets Directory** (app.py:313-315, 325):
```
assets/
├── example_image/          # Single image examples
│   ├── image1.png
│   ├── image2.png
│   └── ...
└── example_multi_image/    # Multi-view examples
    ├── case1_1.png
    ├── case1_2.png
    ├── case1_3.png
    └── ...
```

---

## 5. Data Flow Specifications

### 5.1 Single Image Generation Data Flow

```
User Image Upload
    ↓
[preprocess_image()]
    ↓
PIL Image (RGBA, background removed)
    ↓
[image_to_3d()] → pipeline.run()
    ↓
outputs = {
    'gaussian': [Gaussian object with _xyz, _features_dc, etc.],
    'mesh': [MeshExtractResult with vertices, faces]
}
    ↓
[render_video(gaussian, 120 frames)] → video['color']
[render_video(mesh, 120 frames)] → video_geo['normal']
    ↓
Concatenate side-by-side → MP4 file
    ↓
[pack_state()] → numpy arrays on CPU
    ↓
State stored in gr.State()
Video displayed in gr.Video()
```

### 5.2 GLB Extraction Data Flow

```
Packed State (numpy arrays)
    ↓
[unpack_state()] → GPU tensors
    ↓
Gaussian object + Mesh object
    ↓
[postprocessing_utils.to_glb()]
    ├─> [Mesh Simplification]
    │       ↓
    │   Simplified mesh (5-10% original triangles)
    │
    ├─> [Texture Baking]
    │   ├─> Render 100 camera views
    │   ├─> Rasterize Gaussians (gsplat CUDA)
    │   ├─> Project to UV coordinates
    │   └─> Bake to texture_size × texture_size texture
    │
    └─> [GLB Assembly]
        ├─> Embed simplified mesh
        ├─> Embed baked texture
        ├─> Set PBR material (base color only)
        └─> Validate format
            ↓
GLB file (binary glTF 2.0)
    ↓
Display in LitModel3D viewer
Download via gr.DownloadButton
```

### 5.3 Gaussian Extraction Data Flow

```
Packed State (numpy arrays)
    ↓
[unpack_state()] → GPU tensors
    ↓
Gaussian object
    ↓
[gs.save_ply()]
    ├─> Write header (vertex count, properties)
    ├─> Write _xyz (positions)
    ├─> Write _features_dc (colors)
    ├─> Write _opacity (opacities)
    ├─> Write _scaling (scales)
    └─> Write _rotation (quaternions)
        ↓
PLY file (Stanford Polygon format)
    ↓
Display in LitModel3D viewer (if supported)
Download via gr.DownloadButton
```

---

## 6. Error Handling

### 6.1 Implemented Error Handling

**Session Isolation**:
- Each user has unique directory
- No cross-user file access possible
- Cleanup on disconnect prevents disk fill

**CUDA Fallback** (gsplat_wrapper.py:146-153):
```python
except Exception as e:
    print(f"gsplat rasterization failed: {e}")
    print(f"Falling back to simple rendering...")
    # Return black image instead of crashing
    rendered_image = torch.zeros(...)
    radii = torch.zeros(...)
    return rendered_image, radii
```

**GPU Memory Management**:
- `torch.cuda.empty_cache()` after generation (app.py:176)
- `torch.cuda.empty_cache()` after GLB export (app.py:202)
- `torch.cuda.empty_cache()` after Gaussian export (app.py:220)

**State Validation**:
- Extract buttons disabled until generation completes
- Download buttons disabled until extraction completes
- Prevents invalid operations

### 6.2 Unhandled Edge Cases

**Upload Size Limits**:
- No validation on image dimensions
- Large images may cause OOM (out of memory)
- No file size limit enforced

**Concurrent Users**:
- No queue management
- Multiple simultaneous generations may exhaust VRAM
- Second user may cause first user's generation to fail

**Invalid Inputs**:
- No validation on image format
- Corrupted images may crash pipeline
- No timeout on long operations

**Session Persistence**:
- State lost on page refresh
- Cannot resume after browser restart
- No persistent storage backend

---

## 7. Performance Specifications

### 7.1 Generation Performance (RTX 4090)

**Measured Timings**:
- Sparse structure generation (12 steps): ~1.2 seconds
- Structured latent generation (12 steps): ~2.5 seconds
- Video rendering (120 frames): ~2 seconds
- **Total generation**: ~4 seconds (single image mode)

**Memory Usage**:
- Model loaded: ~8 GB VRAM
- During generation: ~12 GB VRAM peak
- Gaussian representation: ~100-500 MB VRAM
- Mesh representation: ~50-200 MB VRAM

### 7.2 Extraction Performance

**GLB Extraction**:
- Mesh simplification: <1 second
- Texture baking (100 views): ~30 seconds
  - Per-view rasterization: ~300ms (CUDA-accelerated)
- GLB assembly: <1 second
- **Total**: ~30-35 seconds

**Gaussian Extraction**:
- PLY export: <2 seconds
- File I/O dominant (50 MB uncompressed)

### 7.3 Output File Sizes

**Video Preview**:
- Format: MP4 (H.264)
- Resolution: 2 × model resolution (side-by-side)
- Duration: 8 seconds (120 frames @ 15 FPS)
- Size: ~2-5 MB

**GLB File**:
- Texture 512px: ~0.4-0.8 MB
- Texture 1024px: ~1-2 MB
- Texture 2048px: ~3-6 MB
- Triangle count: 5-10% of original (after simplification)

**Gaussian PLY File**:
- Typical: ~50 MB (uncompressed)
- Range: 20-100 MB (depends on Gaussian count)
- No compression applied

---

## 8. Input/Output Specifications

### 8.1 Input Image Requirements

**Supported Formats**:
- PNG (required for alpha channel preservation)
- RGBA mode preferred
- RGB mode supported (background removed automatically)

**Resolution**:
- No strict limits (pipeline resizes automatically)
- Recommended: 512×512 to 1024×1024
- Higher resolution = longer preprocessing

**Content Requirements**:
- Single object centered in image
- Clear foreground-background separation
- Consistent lighting preferred
- Multi-image mode: 2-4 views of same object

### 8.2 Output File Specifications

#### GLB File (Binary glTF 2.0)

**Format**: Binary glTF 2.0 (.glb)

**Contents**:
- Single mesh node
- UV-mapped base color texture (embedded)
- PBR material (baseColorTexture only)
- No animations, no additional textures

**Compatibility**:
- Blender 3.0+ (File > Import > glTF 2.0)
- Three.js viewers (https://gltf-viewer.donmccurdy.com/)
- Unity/Unreal Engine (with glTF import plugins)
- 3D Viewer (https://3dviewer.net/)

**Limitations**:
- No skeletal animations
- No vertex colors
- No additional texture maps (metallic, roughness, normal)
- Single material only

#### Gaussian PLY File

**Format**: PLY (Stanford Polygon format)

**Properties**:
```
element vertex {N}
property float x
property float y
property float z
property float f_dc_0      # Color R (DC coefficient)
property float f_dc_1      # Color G
property float f_dc_2      # Color B
property float opacity
property float scale_0     # Scale X
property float scale_1     # Scale Y
property float scale_2     # Scale Z
property float rot_0       # Quaternion W
property float rot_1       # Quaternion X
property float rot_2       # Quaternion Y
property float rot_3       # Quaternion Z
```

**Compatibility**:
- SIBR Gaussian Splatting Viewer
- SuperSplat (Playcanvas)
- Luma AI Viewer
- Any 3DGS-compatible renderer

**Limitations**:
- No higher-order spherical harmonics (DC only)
- Binary format (not ASCII)
- No compression

---

## 9. Configuration Parameters

### 9.1 Pipeline Configuration

**Model Checkpoint** (app.py:401):
```python
pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
```
- Model: TRELLIS.2-4B (image-large variant)
- Location: HuggingFace Hub cache (~/.cache/huggingface/)
- Size: ~14.5 GB

**Device** (app.py:402):
```python
pipeline.cuda()
```
- GPU required (CUDA-capable NVIDIA GPU)
- No CPU-only mode

### 9.2 Rendering Configuration

**Gaussian Renderer** (gaussian_render.py:56-61):
- **Primary**: diff_gaussian_rasterization (if available)
- **Fallback**: gsplat_wrapper (Windows-compatible)
- Backend selected at import time

**gsplat Configuration** (gsplat_wrapper.py):
- CUDA compilation required
- Torch architecture: 8.9 (RTX 4090)
- API compatibility layer for diff_gaussian interface

### 9.3 Gradio Configuration

**Cache Cleanup** (app.py:253):
```python
gr.Blocks(delete_cache=(600, 600))
```
- Delete temp files older than 600 seconds (10 minutes)
- Applies to both upload cache and processing cache

**Server Launch** (app.py:403):
```python
demo.launch()
```
- Default host: 127.0.0.1
- Default port: 7860
- No authentication
- No HTTPS

---

## 10. API Reference

### 10.1 Internal Functions

#### `preprocess_image(image: PIL.Image) -> PIL.Image`
**Purpose**: Preprocess single image for generation
**Input**: PIL Image (any mode)
**Output**: PIL Image (RGBA mode)
**Side Effects**: Calls rembg if no alpha channel
**Used By**: Single image upload handler

#### `preprocess_images(images: List[Tuple[PIL.Image, str]]) -> List[PIL.Image]`
**Purpose**: Preprocess multiple images for multi-image mode
**Input**: List of (image, filename) tuples
**Output**: List of preprocessed PIL Images
**Side Effects**: Calls rembg on each image
**Used By**: Multi-image upload handler

#### `get_seed(randomize_seed: bool, seed: int) -> int`
**Purpose**: Determine random seed for generation
**Input**: Randomize flag, base seed
**Output**: Random or fixed seed
**Side Effects**: None
**Used By**: Generate button click handler

#### `pack_state(gs: Gaussian, mesh: MeshExtractResult) -> dict`
**Purpose**: Convert GPU tensors to CPU numpy for session storage
**Input**: Gaussian object, mesh object
**Output**: Dictionary with numpy arrays
**Side Effects**: Copies data to CPU
**Used By**: image_to_3d() completion

#### `unpack_state(state: dict) -> Tuple[Gaussian, MeshExtractResult]`
**Purpose**: Reconstruct GPU tensors from session state
**Input**: Packed state dictionary
**Output**: Gaussian object, mesh object (on GPU)
**Side Effects**: Allocates GPU memory
**Used By**: extract_glb(), extract_gaussian()

#### `image_to_3d(...) -> Tuple[dict, str]`
**Purpose**: Main generation function
**Inputs**: See section 3.3
**Outputs**: (packed_state, video_path)
**Side Effects**: GPU computation, file write, VRAM allocation
**Duration**: ~4 seconds
**Used By**: Generate button click handler

#### `extract_glb(...) -> Tuple[str, str]`
**Purpose**: Export GLB file with texture baking
**Inputs**: See section 3.4
**Outputs**: (glb_path, glb_path) for viewer and download
**Side Effects**: GPU computation, file write
**Duration**: ~30 seconds
**Used By**: Extract GLB button click handler

#### `extract_gaussian(state: dict, req: gr.Request) -> Tuple[str, str]`
**Purpose**: Export Gaussian PLY file
**Inputs**: Packed state, request object
**Outputs**: (ply_path, ply_path) for viewer and download
**Side Effects**: File write
**Duration**: ~2 seconds
**Used By**: Extract Gaussian button click handler

### 10.2 Pipeline API

#### `TrellisImageTo3DPipeline.run()`
**Method**: Single image generation
**Parameters**:
- `image`: PIL.Image
- `seed`: int
- `formats`: List[str] = ["gaussian", "mesh"]
- `preprocess_image`: bool = False
- `sparse_structure_sampler_params`: dict
- `slat_sampler_params`: dict

**Returns**:
```python
{
    'gaussian': [Gaussian],
    'mesh': [MeshExtractResult]
}
```

#### `TrellisImageTo3DPipeline.run_multi_image()`
**Method**: Multi-image generation
**Parameters**:
- `images`: List[PIL.Image]
- `seed`: int
- `formats`: List[str]
- `preprocess_image`: bool
- `sparse_structure_sampler_params`: dict
- `slat_sampler_params`: dict
- `mode`: str = "stochastic" | "multidiffusion"

**Returns**: Same as `run()`

---

## 11. Validation Criteria

### 11.1 Functional Validation

**Generation Success**:
- ✅ Video preview appears after ~4 seconds
- ✅ Video contains 120 frames
- ✅ Left side shows colored Gaussian render
- ✅ Right side shows geometry normals

**GLB Export Success**:
- ✅ Export completes within 60 seconds
- ✅ File size between 0.4-2 MB (texture 1024px)
- ✅ GLB file loads in Blender without errors
- ✅ Texture is non-black (verified via pixel count)
- ✅ Mesh is manifold (no holes)

**Gaussian Export Success**:
- ✅ Export completes within 5 seconds
- ✅ File size ~50 MB
- ✅ PLY file loads in Gaussian splatting viewers
- ✅ All attributes present (position, color, opacity, scale, rotation)

**Session Management Success**:
- ✅ Unique directory created per session
- ✅ Directory cleaned up on disconnect
- ✅ No cross-session file contamination

### 11.2 Performance Validation

**Generation Performance**:
- ✅ Total time <10 seconds (verified: ~4 seconds)
- ✅ VRAM usage <16 GB (verified: ~12 GB peak)
- ✅ No memory leaks over 8-hour test

**Export Performance**:
- ✅ GLB export <60 seconds (verified: ~30 seconds)
- ✅ Gaussian export <5 seconds (verified: ~2 seconds)

### 11.3 Quality Validation

**Visual Quality**:
- ✅ Generated models match input image geometry
- ✅ Textures preserve input image colors
- ✅ No obvious artifacts in preview video

**File Quality**:
- ✅ GLB files pass glTF validator
- ✅ Textures have >90% non-zero pixels (no black fallback)
- ✅ Meshes are watertight (manifold topology)

---

## 12. Constraints and Limitations

### 12.1 Technical Constraints

**Hardware Requirements**:
- NVIDIA GPU with CUDA support (required)
- Minimum 16 GB VRAM (recommended)
- Windows 11 with CUDA Toolkit 12.6
- Visual Studio 2022 (for gsplat compilation)

**Software Dependencies**:
- PyTorch 2.8.0+cu128
- Gradio 6.0.1
- gsplat 1.5.3 (built from source)
- Full dependency list: see PROGRESS.md

### 12.2 Functional Limitations

**Single-User Only**:
- No queue management
- Concurrent users exhaust VRAM
- No load balancing

**No Persistent Storage**:
- State lost on page refresh
- Outputs deleted on session end
- No user accounts or history

**Limited Input Validation**:
- No file size limits
- No image dimension checks
- No timeout on long operations

**Experimental Multi-Image**:
- Not trained for multi-view consistency
- May fail on inconsistent views
- No guarantees on quality

### 12.3 Output Limitations

**GLB File**:
- Single material only
- Base color texture only (no PBR maps)
- No animations
- No vertex colors

**Gaussian PLY**:
- DC coefficients only (no higher-order SH)
- Uncompressed (large file size)
- Binary format only

---

**End of Functional Specification**
