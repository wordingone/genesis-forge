# Product Requirements Document
## TRELLIS Forge

**Product Name**: TRELLIS Forge
**Version**: 1.0.0
**Status**: Production-Ready (Internal/Demo)
**Date**: 2026-01-20

---

## 1. Product Vision

TRELLIS Forge transforms 2D images into production-ready 3D assets through a simple web interface, making advanced 3D generation accessible to artists, designers, and content creators without requiring 3D modeling expertise or command-line tools.

---

## 2. Target Users

### Primary
- **Game Asset Creators**: Need rapid prototyping of 3D assets from concept art
- **3D Designers**: Convert reference images to editable 3D meshes
- **AR/VR Content Developers**: Generate 3D models for immersive experiences
- **Product Designers**: Visualize concepts in 3D from sketches/photos

### Secondary
- **Educators**: Teaching 3D modeling concepts
- **Researchers**: Experimenting with 3D generation models
- **Hobbyists**: Creating 3D content for personal projects

---

## 3. Core Requirements

### 3.1 Functional Requirements

#### FR-1: Image Upload & Preprocessing
**Priority**: P0 (Critical)
- **REQ-1.1**: Accept PNG/RGBA images via web upload
- **REQ-1.2**: Automatically remove background if no alpha channel present
- **REQ-1.3**: Provide gallery of example images for quick testing
- **REQ-1.4**: Support multi-image input (experimental feature)
- **REQ-1.5**: Resize images to optimal resolution automatically

**Acceptance Criteria**:
- Upload completes within 2 seconds for images <10MB
- Background removal produces clean alpha masks >90% accuracy
- Example images load instantly on page load

#### FR-2: 3D Generation
**Priority**: P0 (Critical)
- **REQ-2.1**: Generate 3D Gaussian splatting representation from single image
- **REQ-2.2**: Generate triangulated mesh from Gaussian representation
- **REQ-2.3**: Display 120-frame preview video (Gaussian + normals) within 10 seconds
- **REQ-2.4**: Allow adjustment of generation parameters (guidance, steps)
- **REQ-2.5**: Support reproducible generation via seed control

**Acceptance Criteria**:
- Total generation time <10 seconds on RTX 4090
- Generated meshes have manifold topology (watertight)
- Preview video plays smoothly at 15 FPS

#### FR-3: GLB Export
**Priority**: P0 (Critical)
- **REQ-3.1**: Export generated mesh as GLB file with embedded texture
- **REQ-3.2**: Allow mesh simplification control (0.9-0.98 reduction)
- **REQ-3.3**: Allow texture resolution control (512/1024/2048px)
- **REQ-3.4**: Bake Gaussian splatting colors to UV-mapped texture
- **REQ-3.5**: Display 3D preview of GLB before download

**Acceptance Criteria**:
- GLB export completes within 60 seconds
- Exported GLB opens in Blender/Three.js without errors
- Texture quality matches preview video
- File size <5MB for typical assets (1024px texture)

#### FR-4: Gaussian Export
**Priority**: P1 (High)
- **REQ-4.1**: Export Gaussian splatting as .ply file
- **REQ-4.2**: Include all Gaussian attributes (position, color, opacity, scale, rotation)
- **REQ-4.3**: Compatible with standard Gaussian splatting viewers

**Acceptance Criteria**:
- .ply file loads in SIBR viewer without errors
- All Gaussians render correctly

#### FR-5: Session Management
**Priority**: P0 (Critical)
- **REQ-5.1**: Create isolated temporary directory per user session
- **REQ-5.2**: Clean up temporary files on session end
- **REQ-5.3**: Prevent cross-user file access
- **REQ-5.4**: Handle browser disconnection gracefully

**Acceptance Criteria**:
- No files leaked between sessions
- Disk usage returns to baseline after session cleanup
- No orphaned processes after user disconnect

### 3.2 Non-Functional Requirements

#### NFR-1: Performance
- **REQ-P1**: Total generation time <10 seconds (P0)
- **REQ-P2**: GLB export time <60 seconds (P0)
- **REQ-P3**: UI response time <100ms for controls (P1)
- **REQ-P4**: Support 1 concurrent user per GPU (P0)

#### NFR-2: Reliability
- **REQ-R1**: 99% uptime during demo hours (P0)
- **REQ-R2**: Graceful degradation on CUDA errors (P1)
- **REQ-R3**: Automatic cleanup of crashed sessions (P1)
- **REQ-R4**: No memory leaks over 8-hour operation (P0)

#### NFR-3: Usability
- **REQ-U1**: No training required for basic use (P0)
- **REQ-U2**: Visual feedback for all operations (P0)
- **REQ-U3**: Clear error messages on failure (P1)
- **REQ-U4**: Mobile-friendly UI layout (P2)

#### NFR-4: Security
- **REQ-S1**: Validate upload file types (P0)
- **REQ-S2**: Limit upload size to 20MB (P0)
- **REQ-S3**: Sanitize file paths (P0)
- **REQ-S4**: Rate limiting on generation requests (P1)

#### NFR-5: Compatibility
- **REQ-C1**: Works on Windows 11 with NVIDIA GPUs (P0)
- **REQ-C2**: Requires GPU with ≥16GB VRAM (P0)
- **REQ-C3**: Supports modern browsers (Chrome, Firefox, Edge) (P0)
- **REQ-C4**: Exported GLB compatible with Blender 3.0+ (P0)

---

## 4. User Stories

### US-1: Basic Image-to-3D Workflow
**As a** game asset creator
**I want to** upload a concept art image and get a 3D mesh
**So that** I can quickly prototype game characters/objects

**Acceptance Criteria**:
- Upload image via drag-and-drop
- Generation completes in <10 seconds
- Download GLB file that imports into Unity/Unreal

### US-2: Iteration with Parameters
**As a** 3D designer
**I want to** adjust generation quality settings
**So that** I can control the detail/speed tradeoff

**Acceptance Criteria**:
- Sliders for guidance strength and sampling steps
- Preview updates reflect parameter changes
- Settings persist across generations in same session

### US-3: Reproducible Generation
**As a** researcher
**I want to** generate the same 3D model from the same image repeatedly
**So that** I can compare different post-processing techniques

**Acceptance Criteria**:
- Manual seed input field
- Same seed + image = identical output
- Seed value displayed in UI after generation

### US-4: Multi-View Input (Experimental)
**As an** artist with multiple reference photos
**I want to** generate 3D from multiple angles
**So that** I get more accurate geometry

**Acceptance Criteria**:
- Upload 2-4 images of same object
- Algorithm merges views into coherent 3D
- Warning that results may vary

---

## 5. Out of Scope (v1.0)

- **Batch Processing**: Processing multiple images simultaneously
- **API Access**: RESTful API for programmatic access
- **Collaboration**: Multi-user editing/sharing
- **Animation**: Rigging, skinning, or animation generation
- **Custom Training**: Fine-tuning model on user data
- **Cloud Deployment**: Production-ready hosting infrastructure
- **Payment/Licensing**: Commercial licensing system

---

## 6. Success Metrics

### 6.1 Technical Metrics
- **Generation Success Rate**: >95% of uploads produce valid 3D output
- **GLB Export Success Rate**: >98% of generated meshes export successfully
- **Average Generation Time**: <8 seconds on RTX 4090
- **VRAM Efficiency**: Peak usage <16GB

### 6.2 User Metrics
- **Time to First Export**: <2 minutes for new users
- **User Satisfaction**: >80% rate quality as "good" or "excellent"
- **Error Rate**: <5% of sessions encounter errors

### 6.3 Quality Metrics
- **Mesh Topology**: >90% manifold (watertight)
- **Texture Coverage**: >95% of visible surfaces textured
- **File Compatibility**: 100% of GLB files open in Blender

---

## 7. Constraints & Assumptions

### Constraints
- **Hardware**: Requires NVIDIA GPU with CUDA support
- **Platform**: Windows-only in v1.0 (Linux via WSL2)
- **Model Size**: 14.5 GB checkpoint requires fast storage
- **Network**: HuggingFace model download required on first run

### Assumptions
- Users have basic computer literacy
- Users understand 2D→3D conversion has limitations
- Target use case is concept iteration, not production-final assets
- Single-user deployment (no concurrent users on same GPU)

---

## 8. Dependencies

### Critical
- Microsoft TRELLIS.2 model (checkpoint)
- gsplat 1.5.3 with CUDA support (custom-built)
- PyTorch 2.8+ with CUDA 12.6+
- NVIDIA GPU drivers supporting CUDA 12.6+

### Important
- Gradio 6.0+ (UI framework)
- rembg (background removal)
- trimesh (mesh processing)

### Optional
- nvdiffrast (rendering optimization)
- flash-attention (inference speedup)

---

## 9. Risks & Mitigation

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| CUDA compilation failure | High | Medium | Provide prebuilt wheels, detailed build docs |
| Model checkpoint unavailable | Critical | Low | Cache checkpoint locally, mirror hosting |
| VRAM exhaustion | High | Medium | Queue system, memory monitoring |
| Browser compatibility | Medium | Low | Test on Chrome/Firefox/Edge |
| Generated mesh quality issues | Medium | Medium | Expose quality parameters, user education |

---

## 10. Roadmap (Future Versions)

### v1.1 (Planned)
- Linux native support
- Docker deployment
- Basic API endpoints
- Batch processing

### v1.2 (Planned)
- Multi-GPU support
- Queue management
- Enhanced error handling
- Performance profiling

### v2.0 (Vision)
- Custom model fine-tuning
- Animation generation
- Collaborative editing
- Cloud deployment

---

**End of PRD**
