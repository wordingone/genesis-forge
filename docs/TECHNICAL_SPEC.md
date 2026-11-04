# Technical Specification
## TRELLIS Forge v1.0

**Product**: TRELLIS Forge
**Version**: 1.0.0 (Initial Windows Release)
**Date**: 2026-01-20
**Status**: Production-Ready (Internal/Demo)

---

## 1. System Architecture Overview

TRELLIS Forge is a Python-based web application built on the Gradio framework, providing a user interface for Microsoft's TRELLIS.2 3D generation model. The system operates as a single-process application with GPU-accelerated inference.

**High-Level Architecture**:

```
┌─────────────────────────────────────────────────────────┐
│                    Browser Client                       │
│              (Gradio JavaScript Frontend)               │
└────────────────────┬────────────────────────────────────┘
                     │ HTTP/WebSocket
                     │
┌────────────────────┴────────────────────────────────────┐
│                   Gradio Server                         │
│              (Python Backend - app.py)                  │
│  ┌───────────────────────────────────────────────────┐  │
│  │          Session Management                       │  │
│  │  - User temp directories                          │  │
│  │  - State persistence                              │  │
│  │  - Cleanup handlers                               │  │
│  └────────────────┬──────────────────────────────────┘  │
│                   │                                      │
│  ┌────────────────┴──────────────────────────────────┐  │
│  │    TRELLIS Image-to-3D Pipeline                   │  │
│  │  - Preprocessor (rembg, resize)                   │  │
│  │  - Sparse structure sampler                       │  │
│  │  - Structured latent sampler                      │  │
│  │  - Gaussian decoder                               │  │
│  │  - Mesh extractor (FlexiCubes)                    │  │
│  └────────────────┬──────────────────────────────────┘  │
│                   │                                      │
│  ┌────────────────┴──────────────────────────────────┐  │
│  │         Rendering & Export                        │  │
│  │  - Gaussian Renderer (gsplat)                     │  │
│  │  - Video Renderer                                 │  │
│  │  - GLB Exporter (texture baking)                  │  │
│  │  - PLY Exporter                                   │  │
│  └────────────────┬──────────────────────────────────┘  │
└───────────────────┴─────────────────────────────────────┘
                    │
                    │ CUDA API
                    │
┌───────────────────┴─────────────────────────────────────┐
│              NVIDIA GPU (RTX 4090)                      │
│  - PyTorch CUDA Runtime                                 │
│  - gsplat CUDA Extensions                               │
│  - CUDA Toolkit 12.6                                    │
└─────────────────────────────────────────────────────────┘
```

---

## 2. Technology Stack

### 2.1 Core Framework

**Gradio 6.0.1**
- **Purpose**: Web application framework
- **Role**: UI components, HTTP server, WebSocket communication
- **Key Features Used**:
  - Component API (Image, Slider, Button, Video, Gallery)
  - Custom 3D viewer component (LitModel3D)
  - Session management (Request object)
  - State persistence (gr.State)
  - File download handlers (DownloadButton)

### 2.2 Deep Learning Stack

**PyTorch 2.8.0+cu128**
- **Purpose**: Deep learning framework
- **CUDA Version**: 12.8 binaries
- **Key Features Used**:
  - Tensor operations on GPU
  - Autograd (gradient computation)
  - Model loading and inference
  - Memory management (cuda.empty_cache)

**transformers (Hugging Face)**
- **Purpose**: Model loading and tokenization
- **Used By**: TRELLIS pipeline initialization
- **Key Features**:
  - from_pretrained() API
  - Model checkpoint caching

**diffusers (Hugging Face)**
- **Purpose**: Diffusion model components
- **Used By**: Flow matching transformers in TRELLIS
- **Key Features**:
  - Rectified flow sampling
  - Noise scheduling

### 2.3 3D Processing Libraries

**spconv-cu128**
- **Purpose**: Sparse convolution operations
- **Used By**: TRELLIS O-Voxel encoder/decoder
- **CUDA Version**: 12.8
- **Key Features**:
  - Sparse 3D convolutions
  - Efficient voxel processing

**xformers 0.0.33.post2**
- **Purpose**: Efficient attention mechanisms
- **Used By**: TRELLIS flow transformers
- **Fallback For**: flash-attn (Windows incompatible)
- **Key Features**:
  - Memory-efficient attention
  - Gradient checkpointing

**trimesh**
- **Purpose**: Mesh processing and manipulation
- **Used By**: GLB export, mesh simplification
- **Key Features**:
  - Quadric edge collapse decimation
  - Mesh validation
  - Format conversion

**pygltflib**
- **Purpose**: GLB/glTF file I/O
- **Used By**: GLB export
- **Key Features**:
  - Binary glTF writing
  - Texture embedding
  - Scene graph construction

**open3d**
- **Purpose**: 3D data structures and visualization
- **Used By**: Internal 3D utilities
- **Key Features**:
  - Point cloud processing
  - Mesh I/O

**kaolin**
- **Purpose**: NVIDIA 3D deep learning library
- **Used By**: TRELLIS rendering utilities
- **Key Features**:
  - Differentiable rendering primitives
  - 3D data loaders

### 2.4 Rendering Libraries

**gsplat 1.5.3** (CRITICAL - Custom Build)
- **Purpose**: CUDA-accelerated Gaussian splatting rasterization
- **Build**: Custom compiled from source on Windows
- **CUDA Version**: 12.6
- **Torch Architecture**: 8.9 (RTX 4090 Ada Lovelace)
- **Key Features**:
  - CUDA kernel rasterization
  - Anisotropic Gaussian splatting
  - Gradient computation for optimization
- **Critical Fix**: API compatibility wrapper (gsplat_wrapper.py)

**nvdiffrast**
- **Purpose**: Differentiable rasterization
- **Used By**: Advanced rendering features (optional)
- **Key Features**:
  - Triangle rasterization
  - Antialiasing

### 2.5 Image Processing Libraries

**pillow (PIL)**
- **Purpose**: Image I/O and manipulation
- **Used By**: Image upload, preprocessing, format conversion
- **Key Features**:
  - RGBA mode support
  - Resize, crop operations
  - Format conversions

**opencv-python (cv2)**
- **Purpose**: Computer vision operations
- **Used By**: Image preprocessing
- **Key Features**:
  - Color space conversions
  - Image filtering

**rembg**
- **Purpose**: Background removal via U2Net model
- **Used By**: Automatic alpha matte generation
- **Dependencies**: numba, pymatting
- **Key Features**:
  - Automatic foreground segmentation
  - Alpha mask generation

**imageio**
- **Purpose**: Video I/O
- **Used By**: Video preview export
- **Key Features**:
  - MP4 encoding (via imageio-ffmpeg)
  - Frame sequence writing

### 2.6 System Dependencies

**CUDA Toolkit 12.6**
- **Purpose**: GPU computing infrastructure
- **Location**: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6`
- **Components Used**:
  - nvcc compiler
  - cuDNN libraries
  - CUDA runtime

**Visual Studio 2022 Community**
- **Purpose**: C++ compiler for CUDA extensions
- **Component**: MSVC v143
- **Used By**: gsplat compilation
- **Required For**: PyTorch C++ extensions (torch.utils.cpp_extension)

**NVIDIA Driver 581.29**
- **CUDA Support**: 13.0
- **GPU**: RTX 4090
- **Features Used**:
  - CUDA 12.6 runtime
  - Tensor Cores (for xformers attention)

---

## 3. Data Models

### 3.1 Gaussian Representation

**Class**: `Gaussian` (trellis/representations/gaussian/)

**Core Attributes**:
```python
class Gaussian:
    # Initialization parameters
    aabb: torch.Tensor              # [2, 3] bounding box (min, max)
    sh_degree: int                  # Spherical harmonic degree (0 for DC only)
    mininum_kernel_size: float      # Minimum Gaussian size
    scaling_bias: float             # Bias added to scale
    opacity_bias: float             # Bias added to opacity
    scaling_activation: str         # 'exp' or 'softplus'

    # Learned parameters (N Gaussians)
    _xyz: torch.Tensor              # [N, 3] positions
    _features_dc: torch.Tensor      # [N, 1, 3] DC color coefficients
    _scaling: torch.Tensor          # [N, 3] log-space scale (x, y, z)
    _rotation: torch.Tensor         # [N, 4] quaternions (w, x, y, z)
    _opacity: torch.Tensor          # [N, 1] logit-space opacity
```

**Derived Properties**:
```python
@property
def get_xyz(self) -> torch.Tensor:
    """Returns positions [N, 3]"""
    return self._xyz

@property
def get_scaling(self) -> torch.Tensor:
    """Returns exp(_scaling) [N, 3]"""
    return self.scaling_activation(self._scaling + self.scaling_bias)

@property
def get_rotation(self) -> torch.Tensor:
    """Returns normalized quaternions [N, 4]"""
    return self._rotation / self._rotation.norm(dim=-1, keepdim=True)

@property
def get_opacity(self) -> torch.Tensor:
    """Returns sigmoid(_opacity) [N, 1]"""
    return torch.sigmoid(self._opacity + self.opacity_bias)
```

**Storage Format (PLY)**:
```
ply
format binary_little_endian 1.0
element vertex {N}
property float x
property float y
property float z
property float f_dc_0
property float f_dc_1
property float f_dc_2
property float opacity
property float scale_0
property float scale_1
property float scale_2
property float rot_0
property float rot_1
property float rot_2
property float rot_3
end_header
{binary data}
```

### 3.2 Mesh Representation

**Class**: `MeshExtractResult` (easydict)

**Structure**:
```python
MeshExtractResult = edict({
    'vertices': torch.Tensor,  # [V, 3] vertex positions
    'faces': torch.Tensor,     # [F, 3] triangle indices (int32)
})
```

**Constraints**:
- Vertices normalized to unit cube
- Faces use 0-indexed vertex references
- Topology typically manifold (watertight) after extraction

**Storage Format (in GLB)**:
- Embedded as glTF primitive
- Vertex positions in `POSITION` accessor
- Triangle indices in `INDICES` accessor
- UV coordinates generated during texture baking

### 3.3 Session State Model

**Packed State Structure** (app.py:61-75):

```python
StateDict = {
    'gaussian': {
        # Initialization parameters
        'aabb': List[List[float]],              # [[xmin, ymin, zmin], [xmax, ymax, zmax]]
        'sh_degree': int,                       # Typically 0
        'mininum_kernel_size': float,           # Typically 0.0
        'scaling_bias': float,                  # Typically 0.01
        'opacity_bias': float,                  # Typically 0.0
        'scaling_activation': str,              # 'exp'

        # Learned parameters (CPU numpy arrays)
        '_xyz': np.ndarray,                     # [N, 3] float32
        '_features_dc': np.ndarray,             # [N, 1, 3] float32
        '_scaling': np.ndarray,                 # [N, 3] float32
        '_rotation': np.ndarray,                # [N, 4] float32
        '_opacity': np.ndarray,                 # [N, 1] float32
    },
    'mesh': {
        'vertices': np.ndarray,                 # [V, 3] float32
        'faces': np.ndarray,                    # [F, 3] int64
    },
}
```

**Memory Footprint**:
- Gaussian (N=100,000): ~10 MB
- Mesh (V=50,000, F=100,000): ~3 MB
- Total state: ~15-20 MB typical

**Persistence**:
- Stored in `gr.State()` component
- Lives in server memory (not serialized to disk)
- Lost on page refresh

### 3.4 File Formats

#### 3.4.1 GLB File Structure

**Format**: Binary glTF 2.0

**Components**:
```json
{
  "asset": {"version": "2.0"},
  "scene": 0,
  "scenes": [{"nodes": [0]}],
  "nodes": [{"mesh": 0}],
  "meshes": [{
    "primitives": [{
      "attributes": {
        "POSITION": 0,    // Vertex positions
        "TEXCOORD_0": 1   // UV coordinates
      },
      "indices": 2,       // Triangle indices
      "material": 0
    }]
  }],
  "materials": [{
    "pbrMetallicRoughness": {
      "baseColorTexture": {"index": 0},
      "metallicFactor": 0.0,
      "roughnessFactor": 1.0
    }
  }],
  "textures": [{"source": 0}],
  "images": [{
    "mimeType": "image/png",
    "bufferView": 3     // Embedded texture
  }],
  "accessors": [...],   // Data accessors
  "bufferViews": [...], // Data views
  "buffers": [{
    "byteLength": X     // All data embedded
  }]
}
```

**Binary Layout**:
```
[12-byte header]
[JSON chunk: scene graph]
[BIN chunk: vertex data + texture data]
```

#### 3.4.2 PLY File Structure

**Format**: Binary Little-Endian PLY

**Header**:
```
ply\n
format binary_little_endian 1.0\n
element vertex {N}\n
property float x\n
property float y\n
property float z\n
property float f_dc_0\n
property float f_dc_1\n
property float f_dc_2\n
property float opacity\n
property float scale_0\n
property float scale_1\n
property float scale_2\n
property float rot_0\n
property float rot_1\n
property float rot_2\n
property float rot_3\n
end_header\n
```

**Binary Data**:
- N vertices × 13 floats × 4 bytes = 52N bytes
- No compression
- Sequential layout (position, color, opacity, scale, rotation)

---

## 4. Core Algorithms

### 4.1 Image Preprocessing

**Pipeline** (pipeline.preprocess_image):

```python
def preprocess_image(image: PIL.Image) -> PIL.Image:
    """
    1. Alpha channel check
       - If RGBA: extract alpha mask
       - If RGB: generate alpha via rembg U2Net model

    2. Background removal
       - Apply alpha mask
       - Convert to RGBA with transparent background

    3. Resize
       - Target: 512×512 or nearest power-of-2
       - Method: LANCZOS (high-quality downsampling)

    4. Normalization
       - Center object in frame
       - Maintain aspect ratio

    Returns: PIL.Image (RGBA, 512×512)
    """
```

**rembg Background Removal**:
- Model: U2Net (pretrained on salient object detection)
- Input: RGB image
- Output: Grayscale mask [0-255]
- Threshold: 128 (binary mask)
- Post-processing: Morphological closing (fill holes)

### 4.2 3D Generation Algorithm

**High-Level Pipeline** (TrellisImageTo3DPipeline.run):

```
Input: Preprocessed RGBA image (512×512)
    ↓
[Stage 1: Sparse Structure Generation]
    ├─> Encode image to latent (VAE encoder)
    ├─> Sample sparse voxel structure (flow matching)
    │   • Initial noise: N(0, I)
    │   • Flow steps: 12 (configurable)
    │   • CFG strength: 7.5 (guidance scale)
    │   • Timesteps: linear [0, 1]
    ├─> Decode to sparse voxel grid (1024³ sparse)
    ↓
[Stage 2: Structured Latent Generation]
    ├─> Condition on sparse structure
    ├─> Sample structured latent (flow matching)
    │   • Flow steps: 12 (configurable)
    │   • CFG strength: 3.0 (lower for refinement)
    ├─> Decode to dual-grid O-Voxel
    │   • Geometry grid: Sharp features, topology
    │   • Appearance grid: Color, material properties
    ↓
[Gaussian Extraction]
    ├─> Sample points from O-Voxel
    ├─> Initialize Gaussian attributes:
    │   • Positions: Voxel centers
    │   • Colors: Appearance grid values
    │   • Scales: Voxel size
    │   • Rotations: Identity quaternions
    │   • Opacities: Occupancy probabilities
    ↓
[Mesh Extraction (FlexiCubes)]
    ├─> Extract isosurface from geometry grid
    │   • Adaptive grid resolution
    │   • Marching cubes variant
    ├─> Optimize vertex positions
    │   • Gradient-based refinement
    │   • Preserve sharp features
    ├─> Triangulation
    │   • Manifold topology
    │   • Consistent winding
    ↓
Output: Gaussian + Mesh
```

**Flow Matching Details**:

Rectified flow ODE:
```
dx/dt = v_θ(x, t)
```

where `v_θ` is a neural network predicting the velocity field.

**Sampling** (Euler integration):
```python
x_t = x_0  # Start from noise
for t in linspace(0, 1, num_steps):
    v = model(x_t, t, condition)
    x_t = x_t + v * dt
return x_t  # Final latent
```

**Classifier-Free Guidance** (CFG):
```python
v_guided = v_uncond + cfg_strength * (v_cond - v_uncond)
```
- `v_uncond`: Velocity without image conditioning
- `v_cond`: Velocity with image conditioning
- `cfg_strength`: Guidance scale (higher = closer to image)

### 4.3 Gaussian Rasterization Algorithm

**Entry Point**: `GaussianRenderer.render()` (gaussian_render.py:56)

**Fallback Chain**:
1. Try: `diff_gaussian_rasterization` (not available on Windows)
2. Fallback: `gsplat_wrapper` (CUDA-accelerated)
3. Final fallback: Black image (on CUDA error)

**gsplat Rasterization** (gsplat_wrapper.py:108-121):

```python
def rasterize(means3D, quats, scales, opacities, colors, viewmat, K, W, H):
    """
    CUDA kernel rasterization of anisotropic 3D Gaussians

    Inputs:
        means3D: [N, 3] Gaussian centers
        quats: [N, 4] rotation quaternions (normalized)
        scales: [N, 3] axis-aligned scales
        opacities: [N] opacity values [0, 1]
        colors: [N, 3] RGB colors or [N, C_sh] SH coefficients
        viewmat: [4, 4] world-to-camera matrix
        K: [3, 3] intrinsic matrix
        W, H: Image width, height

    Algorithm:
        1. Project Gaussians to 2D (covariance projection)
           Σ_2D = J * R * Σ_3D * R^T * J^T
           where J is Jacobian of projection, R is rotation

        2. Tile-based rendering:
           - Divide image into 16×16 tiles
           - Assign Gaussians to overlapping tiles
           - Sort by depth within each tile

        3. Per-pixel blending (front-to-back):
           For each pixel:
             T = 1  # Transmittance
             C = 0  # Accumulated color
             For each Gaussian in sorted order:
               α = opacity * exp(-0.5 * Σ_2D^(-1) * (p - μ)^2)
               C += T * α * color
               T *= (1 - α)
               if T < ε: break  # Early termination
             return C

    Outputs:
        render_colors: [1, H, W, 3] rendered image
        render_alphas: [1, H, W, 1] alpha channel
        meta: {
            'radii': [N_visible, 2] (x_radius, y_radius) in pixels
            'gaussian_ids': [N_visible] indices of visible Gaussians
        }
    """
```

**API Compatibility Wrapper** (gsplat_wrapper.py:126-142):

gsplat returns `radii` as `[N_visible, 2]`, but diff_gaussian expects `[N]`:

```python
# Initialize full array with zeros
radii = torch.zeros(N, dtype=torch.int32, device=device)

# Fill in visible Gaussians
if 'radii' in meta and 'gaussian_ids' in meta:
    visible_ids = meta['gaussian_ids']        # [N_visible]
    visible_radii = meta['radii']             # [N_visible, 2]
    max_radii = visible_radii.max(dim=1).values  # [N_visible] - take max(x,y)
    radii[visible_ids] = max_radii            # Assign to full array

# Now radii is [N] as expected by diff_gaussian API
```

### 4.4 Texture Baking Algorithm

**Entry Point**: `postprocessing_utils.to_glb()` (postprocessing_utils.py:427-492)

**High-Level Flow**:

```python
def to_glb(gaussian, mesh, simplify=0.95, texture_size=1024):
    """
    1. Mesh simplification
       - Quadric error metric decimation
       - Target: (1 - simplify) × original triangles
       - Preserve topology

    2. UV generation
       - Smart UV projection (minimize distortion)
       - Parameterization: conformal or area-preserving

    3. Multi-view rendering (100 camera positions)
       - Camera placement: Fibonacci sphere sampling
       - For each view:
           a. Set camera pose (orbit around object)
           b. Rasterize Gaussians to image (gsplat CUDA)
           c. Project pixels to UV coordinates
           d. Accumulate colors in texture atlas

    4. Texture compositing
       - Average overlapping projections
       - Fill holes (inpainting if needed)
       - Resize to texture_size × texture_size

    5. GLB assembly
       - Embed simplified mesh
       - Embed baked texture
       - Create PBR material (base color only)
       - Validate glTF structure

    Returns: GLB object ready for export
    """
```

**Camera Placement** (Fibonacci Sphere):

```python
def fibonacci_sphere(n_views=100):
    """
    Evenly distribute n_views cameras on sphere

    Golden ratio method:
    """
    golden_ratio = (1 + sqrt(5)) / 2
    indices = np.arange(n_views)

    # Latitude
    theta = np.arccos(1 - 2 * (indices + 0.5) / n_views)

    # Longitude
    phi = 2 * np.pi * indices / golden_ratio

    # Cartesian coordinates
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    return np.stack([x, y, z], axis=-1)  # [100, 3]
```

**Texture Accumulation**:

```python
# For each pixel in rendered view:
for u, v in uv_coordinates:
    rendered_color = render_colors[y, x]  # From Gaussian rasterization
    texture_atlas[v, u] += rendered_color
    weight_atlas[v, u] += 1

# Average accumulation:
texture_atlas /= (weight_atlas + epsilon)
```

### 4.5 Mesh Simplification Algorithm

**Method**: Quadric Error Metrics (QEM) Edge Collapse

**Implementation**: `trimesh.simplify.simplify_quadric_decimation()`

**Algorithm**:

```
1. Compute quadric error matrix for each vertex
   Q_v = sum over adjacent faces f:
           K_f^T * K_f
   where K_f is plane equation [a, b, c, d] of face f
       (ax + by + cz + d = 0)

2. Compute error for each edge collapse
   For edge (v1, v2):
     Q_new = Q_v1 + Q_v2
     v_new = argmin(v^T * Q_new * v)  # Optimal position
     error = v_new^T * Q_new * v_new

3. Priority queue of edges sorted by error

4. Iteratively collapse lowest-error edge:
   a. Remove edge (v1, v2)
   b. Replace v1, v2 with v_new
   c. Update affected faces
   d. Recompute errors for neighboring edges
   e. Repeat until target triangle count reached

5. Topology preservation:
   - Reject collapses that would create non-manifold vertices
   - Reject collapses that would flip face normals
```

**Performance**:
- Complexity: O(E log E) where E = edge count
- Typical: 100,000 triangles → 5,000 triangles in <1 second

---

## 5. Rendering Pipeline

### 5.1 Video Preview Rendering

**Entry Point**: `render_utils.render_video()` (render_utils.py)

**Process**:

```python
def render_video(representation, num_frames=120):
    """
    Render 120-frame turntable video

    For Gaussian:
        1. Create camera orbit (circular path, 360°)
        2. For each frame:
            a. Set camera pose
            b. Rasterize Gaussians (gsplat CUDA)
            c. Return [H, W, 3] RGB image
        3. Stack frames → [120, H, W, 3]

    For Mesh:
        1. Create camera orbit
        2. For each frame:
            a. Set camera pose
            b. Render geometry normals (phong shading)
            c. Color-code normals: (nx, ny, nz) → RGB
            d. Return [H, W, 3] RGB image
        3. Stack frames → [120, H, W, 3]

    Returns: {'color': frames}
    """
```

**Camera Orbit**:
```python
# Circular path at fixed elevation
radius = 2.0
elevation = 30° (from horizontal)
azimuth = linspace(0°, 360°, 120)  # Full rotation

for i, az in enumerate(azimuth):
    x = radius * cos(az) * cos(elevation)
    y = radius * sin(az) * cos(elevation)
    z = radius * sin(elevation)

    camera_pos = [x, y, z]
    camera_target = [0, 0, 0]  # Look at origin
    camera_up = [0, 0, 1]      # Z-up convention
```

**Video Encoding** (app.py:173-174):

```python
# Concatenate Gaussian and normals side-by-side
video = [np.concatenate([video[i], video_geo[i]], axis=1)
         for i in range(len(video))]

# Save as MP4 (H.264)
imageio.mimsave(video_path, video, fps=15)
# Uses imageio-ffmpeg backend for H.264 encoding
```

### 5.2 3D Viewer Integration

**Component**: `gradio_litmodel3d.LitModel3D`

**Backend**: Three.js (JavaScript 3D library)

**Supported Formats**:
- GLB (binary glTF 2.0)
- PLY (polygon mesh or point cloud)
- OBJ (Wavefront)
- STL (stereolithography)

**Viewer Features**:
- Orbit controls (mouse drag to rotate)
- Zoom (mouse wheel)
- Pan (right-click drag)
- Lighting: Three-point lighting with `exposure` parameter
- Material: PBR rendering (metallic-roughness workflow)

**Loading Process**:
1. Gradio sends file path to frontend
2. JavaScript fetches file via HTTP
3. Three.js GLTFLoader or PLYLoader parses file
4. Scene constructed with mesh, lights, camera
5. Animation loop renders at 60 FPS

---

## 6. Database and Storage

### 6.1 Model Checkpoint Storage

**HuggingFace Hub Cache**:
- **Location**: `~/.cache/huggingface/hub/models--microsoft--TRELLIS-image-large`
- **Structure**:
  ```
  models--microsoft--TRELLIS-image-large/
  ├── snapshots/
  │   └── {commit_hash}/
  │       ├── dinov2_vitb14_pretrain.pth      # DINOv2 encoder
  │       ├── sparse_decoder.safetensors      # Sparse VAE decoder
  │       ├── sparse_encoder.safetensors      # Sparse VAE encoder
  │       ├── slat_decoder.safetensors        # SLAT decoder
  │       ├── slat_encoder.safetensors        # SLAT encoder
  │       ├── flow_model_sparse.safetensors   # Flow transformer (sparse)
  │       ├── flow_model_slat.safetensors     # Flow transformer (SLAT)
  │       └── config.json                     # Model configuration
  └── refs/
      └── main → snapshots/{commit_hash}
  ```
- **Total Size**: ~14.5 GB
- **Download**: One-time on first run (via `transformers.from_pretrained()`)

**Model Loading** (app.py:401):
```python
pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
```
- Checks cache first (offline mode if available)
- Downloads missing files from HuggingFace Hub
- Loads all components into memory (~8 GB VRAM)

### 6.2 Temporary File Storage

**User Directories**:
- **Base**: `<repo>\tmp`
- **Per-Session**: `tmp/{session_hash}/`
- **Lifecycle**: Created on session start, deleted on session end

**File Types**:
- `sample.mp4`: Video preview (2-5 MB)
- `sample.glb`: Extracted GLB (0.4-2 MB)
- `sample.ply`: Extracted Gaussian (20-100 MB)

**Cleanup Policy**:
- **On disconnect**: Entire directory removed (demo.unload)
- **Gradio cache**: Files older than 10 minutes auto-deleted
- **Manual cleanup**: None implemented

**Disk Usage**:
- Per user: ~25-110 MB (if all formats exported)
- Concurrent users: N × 110 MB
- Risk: No disk space monitoring

### 6.3 Asset Storage

**Example Images**:
- **Location**: `assets/example_image/`
- **Format**: PNG (RGBA or RGB)
- **Count**: Variable (current: ~20 images)
- **Size**: ~5 MB total

**Multi-Image Examples**:
- **Location**: `assets/example_multi_image/`
- **Naming**: `{case}_{view}.png` (e.g., `chair_1.png`, `chair_2.png`)
- **Count**: ~5 cases × 3 views = 15 images
- **Size**: ~3 MB total

---

## 7. Security Considerations

### 7.1 Current Security Posture

**Authentication**: None
- No user accounts
- No login required
- Open access to `http://127.0.0.1:7860`

**Authorization**: None
- All users have full access to all features
- No role-based access control

**Input Validation**: Minimal
- File type: Not validated (relies on Gradio component)
- File size: Not validated (OOM risk)
- Image dimensions: Not validated (may cause crashes)

**Session Isolation**: Strong
- Unique temp directory per session
- Session hash generated by Gradio (random UUID)
- File paths isolated (no path traversal possible)

**Network Security**: Weak
- HTTP only (no HTTPS)
- No CORS configuration
- No rate limiting
- Local deployment only (127.0.0.1)

### 7.2 Vulnerability Assessment

**High Risk**:
1. **Denial of Service (DoS)**:
   - No rate limiting on generation requests
   - Single large image can exhaust VRAM
   - Multiple concurrent users can crash application
   - Mitigation: Production deployment needs queue system

2. **Resource Exhaustion**:
   - No upload size limit
   - No timeout on generation
   - No disk space monitoring
   - Mitigation: Add limits before public deployment

**Medium Risk**:
3. **Malformed Input**:
   - Corrupted images may crash pipeline
   - No exception handling in upload path
   - Mitigation: Add try-except wrappers

4. **Path Traversal** (Mitigated):
   - Session hash is UUID (no user control)
   - os.path.join used correctly
   - Risk: LOW (proper session isolation)

**Low Risk**:
5. **Code Injection**:
   - No user input executed as code
   - No SQL database (no SQL injection)
   - Risk: LOW

### 7.3 Security Recommendations

**For Production Deployment**:

1. **Authentication**:
   - Implement OAuth2 or API key system
   - Use Gradio's `auth` parameter for basic auth
   - Example: `demo.launch(auth=("user", "password"))`

2. **HTTPS**:
   - Use reverse proxy (nginx) with TLS certificate
   - Configure HSTS headers
   - Redirect HTTP → HTTPS

3. **Input Validation**:
   ```python
   MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB
   MAX_DIMENSION = 2048
   ALLOWED_FORMATS = ['PNG', 'JPG', 'JPEG']

   def validate_upload(image):
       if image.format not in ALLOWED_FORMATS:
           raise ValueError("Invalid format")
       if image.width > MAX_DIMENSION or image.height > MAX_DIMENSION:
           raise ValueError("Image too large")
       # ... additional checks
   ```

4. **Rate Limiting**:
   - Per-IP request limits (e.g., 10 requests/hour)
   - Queue system for generation requests
   - Timeout on operations (e.g., 60s generation limit)

5. **Resource Limits**:
   - Monitor VRAM usage (reject if >80% full)
   - Monitor disk space (reject uploads if <1 GB free)
   - Process timeout (kill after 120 seconds)

6. **Logging**:
   - Access logs (IP, timestamp, endpoint)
   - Error logs (stack traces, failures)
   - Security events (failed auth, rate limit hits)

---

## 8. Performance Optimization

### 8.1 GPU Optimization

**CUDA Memory Management**:
```python
# Aggressive cleanup after operations
torch.cuda.empty_cache()  # Return cached memory to GPU

# PyTorch allocator configuration
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
# Prevents memory fragmentation
```

**Batch Processing** (Not Implemented):
- Current: One image at a time
- Potential: Batch multiple user requests
- Constraint: VRAM limit (24 GB RTX 4090)

**Mixed Precision** (Partially Used):
- Model weights: float32 (no quantization)
- Intermediate activations: float32
- Potential: Use float16 for 2× speedup, ~1.4× VRAM reduction

**Model Optimizations**:
- xformers attention: 30-40% faster than vanilla attention
- Flash attention: Not available on Windows (50% faster potential)

### 8.2 CPU Optimization

**Image Preprocessing**:
- rembg uses GPU (U2Net model on CUDA)
- PIL resize uses SIMD (pillow-simd available)
- Bottleneck: rembg inference (~500ms)

**File I/O**:
- PLY export: Single-threaded, uncompressed
- Potential: Parallel write, zlib compression
- Current bottleneck: Disk write speed (50 MB @ ~25 MB/s)

**Video Encoding**:
- imageio-ffmpeg: Uses CPU H.264 encoder
- Potential: Use NVENC (NVIDIA GPU encoder)
- Current: 120 frames @ 15 FPS = ~2 seconds encode time

### 8.3 Network Optimization

**Gradio WebSocket**:
- Minimal overhead (local deployment)
- State updates: JSON serialization (lightweight)

**File Downloads**:
- Direct file serve (no chunking)
- Potential: Chunked transfer encoding for large files

**Video Streaming** (Not Implemented):
- Current: Full MP4 download before playback
- Potential: HLS/DASH streaming for progressive load

---

## 9. Scalability Considerations

### 9.1 Current Limitations

**Single-User Design**:
- One GPU serves one request at a time
- No queue management
- VRAM exhaustion on concurrent requests

**Single-Process Architecture**:
- Gradio runs in main thread
- Pipeline inference in main thread
- No parallelism across users

**No Load Balancing**:
- Cannot distribute across multiple GPUs
- Cannot distribute across multiple machines

### 9.2 Scaling Strategies

**Vertical Scaling** (Single Machine):

1. **Multi-GPU Support**:
   ```python
   # Round-robin GPU assignment
   gpu_id = user_id % num_gpus
   device = f'cuda:{gpu_id}'
   pipeline.to(device)
   ```
   - Requires: Multiple RTX 4090s
   - Throughput: Linear scaling (2 GPUs = 2× users)

2. **Request Queue**:
   ```python
   from queue import Queue
   import threading

   request_queue = Queue(maxsize=10)

   def worker():
       while True:
           task = request_queue.get()
           result = pipeline.run(task['image'], ...)
           task['callback'](result)

   # Background thread processes queue
   threading.Thread(target=worker, daemon=True).start()
   ```

**Horizontal Scaling** (Multiple Machines):

1. **Load Balancer + Worker Pool**:
   ```
   [nginx Load Balancer]
       ├─> [Worker 1: GPU Server]
       ├─> [Worker 2: GPU Server]
       └─> [Worker N: GPU Server]
   ```
   - Each worker: Independent Gradio instance
   - Session affinity required (sticky sessions)

2. **Centralized Queue (Redis/RabbitMQ)**:
   ```
   [Frontend Server] → [Redis Queue] ← [Worker Pool]
   ```
   - Frontend: Accept uploads, enqueue tasks
   - Workers: Poll queue, process, return results
   - Requires: WebSocket reconnection logic

### 9.3 Deployment Architectures

**Development** (Current):
```
[Single Process]
  ├─ Gradio Server
  ├─ Pipeline Inference
  └─ File Storage
```
- Users: 1
- Throughput: ~15 generations/hour (4s each)

**Production - Small Scale**:
```
[Docker Container]
  ├─ nginx (HTTPS termination)
  ├─ Gradio App (gunicorn workers)
  └─ GPU access (NVIDIA Docker)

Storage: Mounted volume
Queue: In-memory
```
- Users: 1-3 concurrent (with queue)
- Throughput: ~20 generations/hour (with queue waits)

**Production - Large Scale**:
```
[Load Balancer]
  ├─> [API Server Pool]
  │     └─> [Redis Queue]
  └─> [Worker Pool] (GPU machines)
        ├─> Worker 1 (RTX 4090)
        ├─> Worker 2 (RTX 4090)
        └─> Worker N (RTX 4090)

Storage: S3/Cloud Storage
Monitoring: Prometheus + Grafana
```
- Users: N workers × 1 = N concurrent
- Throughput: N × 15 generations/hour

---

## 10. Error Handling and Logging

### 10.1 Exception Handling

**Current Implementation**:

**gsplat Fallback** (gsplat_wrapper.py:146-153):
```python
try:
    render_colors, render_alphas, meta = gsplat.rasterization(...)
except Exception as e:
    print(f"gsplat rasterization failed: {e}")
    print(f"Falling back to simple rendering...")
    # Return black image instead of crashing
    return black_image, zero_radii
```

**No Exception Handling**:
- Image upload (may crash on corrupted files)
- Generation pipeline (may crash on OOM)
- GLB export (may crash on invalid geometry)
- File I/O (may crash on permission errors)

**Gradio Error Display**:
- Uncaught exceptions shown in UI as error box
- No custom error messages
- Stack trace visible to user (information leak)

### 10.2 Logging

**Current Logging**:
- `print()` statements to console
- No structured logging
- No log files
- No log levels (DEBUG, INFO, WARNING, ERROR)

**Recommended Logging**:
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler('trellis_forge.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Usage:
logger.info(f"Generation started for session {session_hash}")
logger.warning(f"VRAM usage at 90%: {vram_used} MB / {vram_total} MB")
logger.error(f"GLB export failed: {e}", exc_info=True)
```

### 10.3 Monitoring

**Not Implemented**:
- VRAM usage tracking
- Request rate monitoring
- Error rate tracking
- Performance metrics (latency, throughput)

**Recommended Metrics**:
```python
# Prometheus metrics example
from prometheus_client import Counter, Histogram, Gauge

generation_requests = Counter('generation_requests_total', 'Total generations')
generation_duration = Histogram('generation_duration_seconds', 'Generation time')
vram_usage = Gauge('vram_usage_bytes', 'GPU memory used')

# Usage:
with generation_duration.time():
    outputs = pipeline.run(...)
generation_requests.inc()
vram_usage.set(torch.cuda.memory_allocated())
```

---

## 11. Build and Compilation

### 11.1 gsplat Custom Build

**Critical Dependency**: gsplat 1.5.3 must be built from source on Windows.

**Build Script**: `build_gsplat_final.bat`

**Build Environment**:
```batch
# Minimal PATH (avoid "input line too long" error)
set PATH=C:\Windows\System32;C:\Windows;C:\Windows\System32\Wbem
set PATH=%PATH%;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin
set PATH=%PATH%;C:\Program Files (x86)\Microsoft Visual Studio\Installer

# User TEMP folder (avoid access denied)
set TEMP=C:\Users\Admin\AppData\Local\Temp
set TMP=C:\Users\Admin\AppData\Local\Temp

# CUDA configuration
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6
set TORCH_CUDA_ARCH_LIST=8.9  # RTX 4090 Ada Lovelace

# Force CUDA build
set FORCE_CUDA=1

# Fix for setuptools VC environment detection
set DISTUTILS_USE_SDK=1
```

**Compilation Process**:
1. Initialize Visual Studio 2022 environment:
   ```batch
   call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
   ```

2. Compile CUDA extensions:
   ```batch
   pip install gsplat --no-binary gsplat --no-cache-dir
   ```
   - Downloads gsplat source from PyPI
   - Runs `setup.py build_ext`
   - Compiles `.cu` files with nvcc
   - Links against PyTorch CUDA libraries
   - Installs `.pyd` extension modules

**Build Output**:
```
venv311\Lib\site-packages\gsplat\
├── __init__.py
├── _C.cp311-win_amd64.pyd       # CUDA extension
├── cuda_legacy\
│   └── cuda_legacy.cp311-win_amd64.pyd
└── rendering\
    └── rendering.cp311-win_amd64.pyd
```

**Verification**:
```python
import gsplat
print(gsplat.__version__)  # 1.5.3
print(gsplat._C)           # <module 'gsplat._C' from '...pyd'>
```

### 11.2 Dependency Installation

**Standard Install** (works on Windows):
```bash
pip install torch==2.8.0+cu128 --index-url https://download.pytorch.org/whl/cu128
pip install gradio==6.0.1
pip install transformers diffusers
pip install trimesh pygltflib open3d
pip install pillow opencv-python imageio imageio-ffmpeg
pip install rembg
pip install spconv-cu128
pip install xformers==0.0.33.post2
pip install gradio-litmodel3d
pip install easydict
```

**Version Constraints**:
- PyTorch 2.8.0 (kaolin requires ≥2.8)
- numpy <2.4 (numba requires ≤2.3)
- Gradio 6.0.1 (LitModel3D compatibility)
- xformers 0.0.33.post2 (PyTorch 2.8 compatible)

**Windows-Specific Issues**:
- flash-attn: Not available (use xformers fallback)
- diff-gaussian-rasterization: Build fails (use gsplat)
- numba: Requires numpy ≤2.3 (conflict with gsplat)

---

## 12. Testing Infrastructure

### 12.1 Unit Tests

**Not Implemented**:
- No pytest suite
- No unit tests for functions
- No integration tests

**Recommended Tests**:
```python
# tests/test_preprocessing.py
def test_preprocess_removes_background():
    image = PIL.Image.open("test_assets/rgb_image.png")
    result = preprocess_image(image)
    assert result.mode == "RGBA"
    assert has_alpha_channel(result)

# tests/test_generation.py
def test_generation_produces_gaussian():
    image = load_test_image()
    outputs = pipeline.run(image, seed=42, ...)
    assert 'gaussian' in outputs
    assert outputs['gaussian'][0]._xyz.shape[0] > 0

# tests/test_export.py
def test_glb_export_creates_file():
    state = load_test_state()
    glb_path, _ = extract_glb(state, ...)
    assert os.path.exists(glb_path)
    assert os.path.getsize(glb_path) > 0
```

### 12.2 Integration Tests

**Manual Tests Run** (see PROGRESS.md):

1. **test_gsplat_cuda.py**: ✅ PASSED
   - Tests CUDA rasterization with 1000 Gaussians
   - Verifies non-zero output

2. **test_glb_export.py**: ✅ PASSED
   - End-to-end generation and GLB export
   - Verifies file creation and size

3. **debug_gsplat_wrapper.py**: ✅ PASSED
   - Tests wrapper API compatibility
   - Verifies correct shapes and formats

**Recommended Integration Tests**:
- Full single-image workflow
- Full multi-image workflow
- Session lifecycle (start → generate → extract → cleanup)
- Concurrent user isolation
- Error recovery (OOM, CUDA error)

### 12.3 Performance Tests

**Not Implemented**:
- No latency benchmarks
- No throughput tests
- No memory profiling

**Recommended Benchmarks**:
```python
# benchmark/latency.py
import time

def benchmark_generation(n_runs=10):
    times = []
    for i in range(n_runs):
        start = time.time()
        outputs = pipeline.run(test_image, seed=i, ...)
        elapsed = time.time() - start
        times.append(elapsed)

    print(f"Mean: {np.mean(times):.2f}s")
    print(f"Std: {np.std(times):.2f}s")
    print(f"Min: {np.min(times):.2f}s")
    print(f"Max: {np.max(times):.2f}s")
```

---

## 13. Configuration Management

### 13.1 Environment Variables

**Currently Used**:
- `CUDA_HOME`: Path to CUDA toolkit (build only)
- `TORCH_CUDA_ARCH_LIST`: GPU architecture (build only)
- `FORCE_CUDA`: Force CUDA build (build only)
- `DISTUTILS_USE_SDK`: Fix setuptools (build only)

**Potential Runtime Config**:
```python
# .env file
GRADIO_SERVER_PORT=7860
GRADIO_SERVER_NAME=127.0.0.1
CUDA_VISIBLE_DEVICES=0  # Select GPU
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
LOG_LEVEL=INFO
MAX_UPLOAD_SIZE=20971520  # 20 MB
SESSION_TIMEOUT=3600  # 1 hour
```

### 13.2 Model Configuration

**Hardcoded**:
- Model name: "microsoft/TRELLIS-image-large"
- Device: cuda (no CPU fallback)
- Precision: float32 (no mixed precision)

**Potential Config File** (config.yaml):
```yaml
model:
  name: microsoft/TRELLIS-image-large
  device: cuda
  dtype: float32
  cache_dir: ~/.cache/huggingface

generation:
  default_seed: 0
  default_ss_guidance: 7.5
  default_ss_steps: 12
  default_slat_guidance: 3.0
  default_slat_steps: 12

export:
  default_simplify: 0.95
  default_texture_size: 1024
  video_fps: 15
  video_frames: 120

server:
  host: 127.0.0.1
  port: 7860
  auth: false
```

---

## 14. Deployment Requirements

### 14.1 Hardware Requirements

**Minimum**:
- GPU: NVIDIA RTX 3090 (24 GB VRAM)
- CPU: 8-core Intel/AMD
- RAM: 32 GB
- Storage: 50 GB SSD (20 GB model + 30 GB workspace)

**Recommended**:
- GPU: NVIDIA RTX 4090 (24 GB VRAM)
- CPU: 16-core Intel/AMD
- RAM: 64 GB
- Storage: 100 GB NVMe SSD

**Multi-User Setup**:
- GPUs: N × RTX 4090 (one per concurrent user)
- RAM: 64 GB + (N × 16 GB)
- Storage: 100 GB + (N × 50 GB scratch space)

### 14.2 Software Requirements

**Operating System**:
- Windows 11 (current)
- Windows Server 2022 (production)
- Ubuntu 22.04 LTS (Linux alternative, untested)

**System Software**:
- NVIDIA Driver ≥ 560 (CUDA 12.6+ support)
- CUDA Toolkit 12.6 or newer
- Visual Studio 2022 (C++ compiler)
- Python 3.11.x

**Network**:
- Bandwidth: 10 Mbps per concurrent user (download)
- Latency: <100ms (local network preferred)

### 14.3 Production Deployment Checklist

**Infrastructure**:
- [ ] Reverse proxy (nginx) with HTTPS
- [ ] SSL certificate (Let's Encrypt or commercial)
- [ ] Firewall rules (allow 443, deny 7860 external access)
- [ ] Monitoring (Prometheus + Grafana)
- [ ] Log aggregation (ELK stack or CloudWatch)
- [ ] Backup strategy (model checkpoints, code)

**Application**:
- [ ] Authentication (OAuth2 or API keys)
- [ ] Rate limiting (10 requests/hour per IP)
- [ ] Input validation (file size, dimensions, format)
- [ ] Error handling (try-except wrappers)
- [ ] Queue system (Redis + Celery or RabbitMQ)
- [ ] Session timeout (1 hour max)
- [ ] Disk space monitoring (reject uploads if <5 GB free)
- [ ] VRAM monitoring (reject requests if >80% used)

**Testing**:
- [ ] Load testing (Apache Bench or Locust)
- [ ] Security audit (OWASP Top 10)
- [ ] Penetration testing
- [ ] Disaster recovery drill

**Documentation**:
- [ ] Deployment guide
- [ ] Operations runbook
- [ ] Incident response plan
- [ ] API documentation (if exposing API)

---

**End of Technical Specification**
