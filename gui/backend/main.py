"""
Genesis Forge Backend - FastAPI Server
Provides full control over TRELLIS text-to-3D and image-to-3D generation

IMPORTANT: This application runs FULLY OFFLINE after initial model download.
All models must be pre-cached using scripts/download_models.py
"""
import os
import sys
from pathlib import Path

# ===== CUDA ALLOCATOR CONFIG - Must be set BEFORE any torch imports =====
# Enable expandable_segments for VMM-based allocation (eliminates fragmentation)
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# ===== LOCAL MODEL CACHE - Must be set BEFORE any HuggingFace imports =====
# Use local models/ directory for HuggingFace cache instead of C:\Users\...
_TRELLIS_ROOT = Path(__file__).parent.parent.parent
_LOCAL_MODELS_DIR = _TRELLIS_ROOT / "models"
_LOCAL_MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Set HuggingFace cache to local directory
# All HuggingFace caches point to models/hub where models are stored
os.environ['HF_HOME'] = str(_LOCAL_MODELS_DIR)
os.environ['HF_HUB_CACHE'] = str(_LOCAL_MODELS_DIR / "hub")
os.environ['HUGGINGFACE_HUB_CACHE'] = str(_LOCAL_MODELS_DIR / "hub")
os.environ['TRANSFORMERS_CACHE'] = str(_LOCAL_MODELS_DIR / "hub")

# Set torch hub cache for DINOv2 model
os.environ['TORCH_HOME'] = str(_LOCAL_MODELS_DIR / "torch_hub")

# ===== OFFLINE MODE - Must be set BEFORE any HuggingFace imports =====
# This ensures the application never tries to download from HuggingFace
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import base64
import io
import uuid
import asyncio
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime

# ===== Session Debug Logging Setup =====
# Create fresh log file per session with timestamp
_LOG_DIR = Path(__file__).parent.parent.parent / "outputs" / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)

_SESSION_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
_SESSION_LOG_FILE = _LOG_DIR / f"backend_{_SESSION_TIMESTAMP}.log"

# Configure root logger to capture all debug output
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler(_SESSION_LOG_FILE, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

# Create module logger
logger = logging.getLogger("trellis.backend")
logger.info(f"=== TRELLIS Backend Session Started ===")
logger.info(f"Session log: {_SESSION_LOG_FILE}")

# Capture print statements containing debug tags and log them
class _LoggingPrintCapture:
    """Captures print statements containing debug tags and logs them."""
    def __init__(self, original_stdout):
        self.original_stdout = original_stdout
        self.debug_tags = ['[GSPLAT', '[TEXTURE', '[DEBUG', '[ERROR', '[WARN', '[JOB', '[API', '[SPARSE', '[SLAT', '[MESH']
        self._logger = logging.getLogger("trellis.print")

    def write(self, text):
        self.original_stdout.write(text)
        # Also log debug-tagged prints to file
        if any(tag in text for tag in self.debug_tags):
            clean_text = text.strip()
            if clean_text:
                self._logger.debug(clean_text)

    def flush(self):
        self.original_stdout.flush()

sys.stdout = _LoggingPrintCapture(sys.stdout)

# Add TRELLIS to path
# main.py is at gui/backend/main.py
# parent.parent.parent goes: main.py -> backend -> gui -> trellis (root)
# The trellis package is at trellis/trellis/, so we need the root in path
trellis_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(trellis_root))

# Add o_voxel to path (CUDA extension for TRELLIS.2 mesh extraction)
# o_voxel package is at genesis-forge/o_voxel/o_voxel/, parent dir needed in path
o_voxel_parent = trellis_root / "o_voxel"
if o_voxel_parent.exists():
    sys.path.insert(0, str(o_voxel_parent))

# Set environment
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
import gc
import torch
from PIL import Image


def clear_cuda_memory():
    """Aggressively clear CUDA memory to prevent OOM errors.

    This should be called before each generation to ensure maximum available VRAM.
    The sequence is important:
    1. gc.collect() - Release Python references to tensors
    2. cuda.empty_cache() - Release PyTorch's cached memory back to CUDA
    3. cuda.synchronize() - Ensure all CUDA operations complete before proceeding
    """
    if torch.cuda.is_available():
        gc.collect()  # Collect Python garbage (unreferenced tensors)
        torch.cuda.empty_cache()  # Release cached memory to CUDA driver
        torch.cuda.synchronize()  # Wait for all CUDA operations to complete


def clear_pipeline_cache():
    """Clear the pipeline cache to free GPU memory held by models.

    This is a more aggressive memory recovery that unloads all models from GPU.
    Use this after OOM errors to allow the next generation to succeed.
    Note: Next generation will be slower as models must be reloaded.
    """
    global pipelines
    for key in pipelines:
        if pipelines[key] is not None:
            # Move model to CPU first to free GPU memory
            try:
                pipelines[key].cpu()
            except Exception:
                pass
            pipelines[key] = None
    clear_cuda_memory()
    print("[MEMORY] Pipeline cache cleared - models will reload on next generation")

# ===== PERFORMANCE OPTIMIZATION: Enable cuDNN benchmark =====
# This enables kernel auto-tuning for convolution operations
torch.backends.cudnn.benchmark = True

# ===== INFERENCE MODE: Disable gradient computation =====
# Matches test script behavior for deterministic results
torch.set_grad_enabled(False)

# ===== CUDA DETERMINISM: Ensure reproducible results =====
# Required for identical output between test script and GUI
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True, warn_only=True)

from trellis.pipelines import TrellisTextTo3DPipeline, TrellisImageTo3DPipeline, TrellisImageTo3DPipelineV2, Trellis2ImageTo3DPipeline
from trellis.utils import postprocessing_utils, render_utils

# Initialize FastAPI
app = FastAPI(
    title="TRELLIS GUI Backend",
    description="Full control over TRELLIS 3D generation",
    version="1.0.0"
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for Electron (file://) and local development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Storage paths
OUTPUT_DIR = Path(__file__).parent.parent.parent / "outputs"
UPLOAD_DIR = Path(__file__).parent.parent.parent / "uploads"
OUTPUT_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)

# Pipeline cache
pipelines = {
    "text_to_3d": None,
    "image_to_3d": None,
    "image_to_3d_v2": None,  # TRELLIS.2 pipeline
}

# Job tracking (in-memory, cleared on restart)
jobs: Dict[str, Dict[str, Any]] = {}


# ===== Startup/Shutdown Lifecycle =====

@app.on_event("startup")
async def startup_event():
    """Clean state on server startup"""
    global jobs, pipelines

    # Clear any stale job state
    jobs.clear()

    # Reset pipeline cache
    pipelines["text_to_3d"] = None
    pipelines["image_to_3d"] = None
    pipelines["image_to_3d_v2"] = None

    # Clear CUDA memory aggressively
    clear_cuda_memory()

    logger.info("Server startup: state cleared, ready for new session")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on server shutdown"""
    global jobs, pipelines

    # Clear jobs
    jobs.clear()

    # Release pipeline memory
    pipelines["text_to_3d"] = None
    pipelines["image_to_3d"] = None
    pipelines["image_to_3d_v2"] = None

    # Clear CUDA memory aggressively
    clear_cuda_memory()

    logger.info("Server shutdown: resources released")


# ===== Pydantic Models =====

class TextTo3DRequest(BaseModel):
    prompt: str = Field(..., description="Text description of 3D object")
    seed: int = Field(42, description="Random seed for reproducibility")
    sparse_steps: int = Field(12, description="Sparse structure sampling steps")
    sparse_cfg: float = Field(7.5, description="Sparse structure CFG strength")
    slat_steps: int = Field(12, description="SLat sampling steps")
    slat_cfg: float = Field(3.0, description="SLat CFG strength")
    simplify: float = Field(0.95, description="Mesh simplification ratio (0=none, 0.95=aggressive). Lower = more faces = fewer holes")
    texture_size: int = Field(1024, description="Texture resolution (512, 1024, 2048)")
    pbr: bool = Field(False, description="Enable PBR materials (metallic/roughness textures)")
    metallic: float = Field(0.0, description="Default metallic value (0-1). Only used if pbr=true")
    roughness: float = Field(1.0, description="Default roughness value (0-1). Only used if pbr=true")


class ImageTo3DRequest(BaseModel):
    """TRELLIS.2 Image-to-3D request parameters. Defaults match official HuggingFace demo."""
    seed: int = Field(42, description="Random seed")

    # Stage 1: Sparse Structure Generation
    sparse_steps: int = Field(12, description="Sparse structure sampling steps")
    sparse_cfg: float = Field(7.5, description="Sparse structure guidance strength")
    guidance_rescale_sparse: float = Field(0.7, description="Sparse structure guidance rescale")
    rescale_t_sparse: float = Field(5.0, description="Sparse structure rescale T")

    # Stage 2: Shape Generation
    slat_steps: int = Field(12, description="Shape generation sampling steps")
    slat_cfg: float = Field(7.5, description="Shape generation guidance strength")
    guidance_rescale_shape: float = Field(0.5, description="Shape generation guidance rescale")
    rescale_t_shape: float = Field(3.0, description="Shape generation rescale T")

    # Stage 3: Material Generation
    tex_steps: int = Field(12, description="Material generation sampling steps")
    tex_cfg: float = Field(1.0, description="Material generation guidance strength (official: 1.0)")
    guidance_rescale_material: float = Field(0.0, description="Material generation guidance rescale")
    rescale_t_material: float = Field(3.0, description="Material generation rescale T")

    # Output settings
    decimation_target: int = Field(300000, description="Target face count for mesh decimation (official: 300000)")
    simplify: float = Field(0.95, description="Mesh simplification ratio (0=none, 0.95=aggressive)")
    texture_size: int = Field(2048, description="Texture resolution (1024, 2048, 4096)")

    # Legacy/advanced
    pbr: bool = Field(False, description="Enable PBR materials (metallic/roughness textures)")
    metallic: float = Field(0.0, description="Default metallic value (0-1). Only used if pbr=true")
    roughness: float = Field(1.0, description="Default roughness value (0-1). Only used if pbr=true")

    # TRELLIS.2 pipeline settings
    pipeline_version: str = Field("v2", description="Pipeline version: 'v1' (TRELLIS.1) or 'v2' (TRELLIS.2)")
    resolution: int = Field(1024, description="TRELLIS.2 output resolution (512, 1024, 1536). Official default: 1024")


class JobStatus(BaseModel):
    job_id: str
    status: str  # "queued", "processing", "completed", "failed"
    progress: float  # 0.0 to 1.0
    message: str
    outputs: Optional[Dict[str, str]] = None
    error: Optional[str] = None
    created_at: str
    updated_at: str


# ===== Helper Functions =====

def load_pipeline(pipeline_type: str):
    """Lazy load pipelines"""
    if pipelines[pipeline_type] is None:
        print(f"Loading {pipeline_type} pipeline...")
        if pipeline_type == "text_to_3d":
            pipelines[pipeline_type] = TrellisTextTo3DPipeline.from_pretrained(
                "microsoft/TRELLIS-text-xlarge"
            )
        elif pipeline_type == "image_to_3d":
            pipelines[pipeline_type] = TrellisImageTo3DPipeline.from_pretrained(
                "microsoft/TRELLIS-image-large"
            )
        elif pipeline_type == "image_to_3d_v2":
            # TRELLIS.2 pipeline - use local model path first, then HuggingFace
            trellis2_local = _TRELLIS_ROOT / "models" / "hub" / "TRELLIS.2-4B"
            try:
                if trellis2_local.exists():
                    print(f"[TRELLIS.2] Loading from local path: {trellis2_local}")
                    pipelines[pipeline_type] = Trellis2ImageTo3DPipeline.from_pretrained(
                        str(trellis2_local)
                    )
                else:
                    print(f"[TRELLIS.2] Loading from HuggingFace: microsoft/TRELLIS.2-4B")
                    pipelines[pipeline_type] = Trellis2ImageTo3DPipeline.from_pretrained(
                        "microsoft/TRELLIS.2-4B"
                    )
                print(f"[TRELLIS.2] Loaded native TRELLIS.2-4B model")
            except Exception as e:
                # Fall back to V1 model with V2 pipeline (for compatibility)
                print(f"[TRELLIS.2] TRELLIS.2-4B not available ({e}), falling back to TRELLIS.1")
                import traceback
                traceback.print_exc()
                pipelines[pipeline_type] = TrellisImageTo3DPipeline.from_pretrained(
                    "microsoft/TRELLIS-image-large"
                )

        if torch.cuda.is_available():
            # Force low_vram mode for TRELLIS.2 to prevent OOM during generation
            # Even 24GB GPUs can OOM when flow models + intermediate tensors peak together
            # low_vram mode offloads models to CPU when not in use, sacrificing ~10-15% speed
            # for reliable memory management
            if pipeline_type == "image_to_3d_v2" and hasattr(pipelines[pipeline_type], 'low_vram'):
                pipelines[pipeline_type].low_vram = True
                print(f"[TRELLIS.2] Forcing low_vram=True for reliable memory management")
            pipelines[pipeline_type].cuda()

        print(f"{pipeline_type} pipeline loaded!")

    return pipelines[pipeline_type]


async def run_text_to_3d_job(job_id: str, params: TextTo3DRequest):
    """Background task for text-to-3D generation"""
    try:
        # Update status IMMEDIATELY and yield to event loop
        print(f"[JOB {job_id[:8]}] Setting status to PROCESSING")
        jobs[job_id]["status"] = "processing"
        jobs[job_id]["progress"] = 0.1
        jobs[job_id]["message"] = "Loading pipeline..."
        jobs[job_id]["updated_at"] = datetime.now().isoformat()
        print(f"[JOB {job_id[:8]}] Status is now: {jobs[job_id]['status']}")
        await asyncio.sleep(0)  # Yield to allow API requests to be processed

        # Load pipeline (run in executor to not block event loop)
        loop = asyncio.get_event_loop()
        pipeline = await loop.run_in_executor(None, load_pipeline, "text_to_3d")

        jobs[job_id]["progress"] = 0.2
        jobs[job_id]["message"] = "Generating 3D structure..."
        jobs[job_id]["updated_at"] = datetime.now().isoformat()
        await asyncio.sleep(0)  # Yield again

        # Log generation parameters for debugging
        print(f"[JOB {job_id[:8]}] PROMPT: '{params.prompt}'")
        print(f"[JOB {job_id[:8]}] SEED: {params.seed}")

        # Clear CUDA memory aggressively before generation to prevent OOM
        clear_cuda_memory()
        print(f"[JOB {job_id[:8]}] CUDA memory cleared")

        # Run generation in executor (blocking operation)
        def run_pipeline():
            return pipeline.run(
                params.prompt,
                seed=params.seed,
                sparse_structure_sampler_params={
                    "steps": params.sparse_steps,
                    "cfg_strength": params.sparse_cfg,
                },
                slat_sampler_params={
                    "steps": params.slat_steps,
                    "cfg_strength": params.slat_cfg,
                },
            )
        outputs = await loop.run_in_executor(None, run_pipeline)

        jobs[job_id]["progress"] = 0.7
        jobs[job_id]["message"] = "Exporting files..."
        jobs[job_id]["updated_at"] = datetime.now().isoformat()
        await asyncio.sleep(0)  # Yield

        # Save outputs
        output_prefix = OUTPUT_DIR / job_id
        output_files = {}

        # PLY file (run in executor)
        ply_path = f"{output_prefix}_gaussian.ply"
        await loop.run_in_executor(None, outputs['gaussian'][0].save_ply, ply_path)
        output_files["ply"] = ply_path

        jobs[job_id]["progress"] = 0.8
        jobs[job_id]["message"] = "Baking textures..."
        jobs[job_id]["updated_at"] = datetime.now().isoformat()
        await asyncio.sleep(0)  # Yield

        # GLB file (run in executor - this is the slow texture baking step)
        try:
            print(f"[TEXTURE DEBUG] Starting GLB export with texture_size={params.texture_size}, pbr={params.pbr}")
            print(f"[TEXTURE DEBUG] Gaussian representation type: {type(outputs['gaussian'][0])}")
            print(f"[TEXTURE DEBUG] Mesh type: {type(outputs['mesh'][0])}")

            def export_glb():
                try:
                    print("[TEXTURE DEBUG] Inside export_glb executor thread")
                    import sys
                    sys.stdout.flush()
                    if params.pbr:
                        # Use PBR export with metallic/roughness textures
                        print(f"[TEXTURE DEBUG] Using PBR export: metallic={params.metallic}, roughness={params.roughness}")
                        result = postprocessing_utils.to_glb_pbr(
                            outputs['gaussian'][0],
                            outputs['mesh'][0],
                            simplify=params.simplify,
                            texture_size=params.texture_size,
                            metallic=params.metallic,
                            roughness=params.roughness,
                            debug=True,
                            verbose=True,
                        )
                    else:
                        # Use standard export (albedo only)
                        result = postprocessing_utils.to_glb(
                            outputs['gaussian'][0],
                            outputs['mesh'][0],
                            simplify=params.simplify,
                            texture_size=params.texture_size,
                            verbose=True,
                        )
                    print("[TEXTURE DEBUG] GLB export returned successfully")
                    sys.stdout.flush()
                    return result
                except Exception as inner_e:
                    print(f"[TEXTURE ERROR] Exception inside executor: {inner_e}")
                    import traceback
                    traceback.print_exc()
                    sys.stdout.flush()
                    raise
            glb = await loop.run_in_executor(None, export_glb)

            print(f"[TEXTURE DEBUG] GLB export complete")
            # Handle both single PBRMaterial and multi-material cases
            if hasattr(glb.visual, 'material'):
                mat = glb.visual.material
                if hasattr(mat, 'materials'):
                    # Multi-material case
                    print(f"[TEXTURE DEBUG] GLB has {len(mat.materials)} materials")
                    for i, m in enumerate(mat.materials):
                        print(f"[TEXTURE DEBUG] Material {i}: {type(m).__name__}")
                else:
                    # Single PBRMaterial case
                    print(f"[TEXTURE DEBUG] GLB has single material: {type(mat).__name__}")
                    if hasattr(mat, 'baseColorTexture') and mat.baseColorTexture is not None:
                        print(f"[TEXTURE DEBUG]   - baseColorTexture present")
            else:
                print(f"[TEXTURE DEBUG] GLB has no material")

            glb_path = f"{output_prefix}_model.glb"
            glb.export(glb_path)
            print(f"[TEXTURE DEBUG] GLB saved to: {glb_path}")

            # Verify file size (should be >1MB if textures are included)
            import os
            file_size_mb = os.path.getsize(glb_path) / (1024 * 1024)
            print(f"[TEXTURE DEBUG] GLB file size: {file_size_mb:.2f} MB")

            output_files["glb"] = glb_path
        except Exception as e:
            print(f"GLB export failed: {e}")
            import traceback
            traceback.print_exc()
            output_files["glb"] = None
            output_files["glb_error"] = str(e)

        # Preview images
        try:
            video_frames = render_utils.render_video(outputs['gaussian'][0], num_frames=1)
            preview_img = Image.fromarray(video_frames['color'][0])
            preview_path = f"{output_prefix}_preview.png"
            preview_img.save(preview_path)
            output_files["preview"] = preview_path
        except Exception as e:
            print(f"Preview generation failed: {e}")
            output_files["preview"] = None

        # Success - check if GLB export failed
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["progress"] = 1.0
        if output_files.get("glb") is None:
            jobs[job_id]["message"] = "Generation complete (GLB export failed)"
            jobs[job_id]["glb_error"] = output_files.get("glb_error", "Unknown error")
        else:
            jobs[job_id]["message"] = "Generation complete!"
        jobs[job_id]["outputs"] = output_files
        jobs[job_id]["updated_at"] = datetime.now().isoformat()

    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        jobs[job_id]["message"] = f"Failed: {str(e)}"
        jobs[job_id]["updated_at"] = datetime.now().isoformat()
        import traceback
        traceback.print_exc()
        # Clear CUDA memory after failure to prevent OOM on next attempt
        # For OOM errors, also clear the pipeline cache to free model memory
        if "out of memory" in str(e).lower():
            print(f"[JOB {job_id[:8]}] OOM detected - clearing pipeline cache")
            clear_pipeline_cache()
        else:
            clear_cuda_memory()
            print(f"[JOB {job_id[:8]}] CUDA memory cleared after failure")


async def run_image_to_3d_job(job_id: str, image_path: str, params: ImageTo3DRequest):
    """Background task for image-to-3D generation (supports V1 and V2 pipelines)"""
    try:
        # Determine which pipeline to use
        use_v2 = params.pipeline_version == "v2"
        pipeline_key = "image_to_3d_v2" if use_v2 else "image_to_3d"

        # Update status IMMEDIATELY and yield to event loop
        jobs[job_id]["status"] = "processing"
        jobs[job_id]["progress"] = 0.1
        jobs[job_id]["message"] = f"Loading {'TRELLIS.2' if use_v2 else 'TRELLIS.1'} pipeline..."
        jobs[job_id]["updated_at"] = datetime.now().isoformat()
        await asyncio.sleep(0)  # Yield to allow API requests

        loop = asyncio.get_event_loop()
        pipeline = await loop.run_in_executor(None, load_pipeline, pipeline_key)

        jobs[job_id]["progress"] = 0.2
        jobs[job_id]["message"] = "Processing image..."
        jobs[job_id]["updated_at"] = datetime.now().isoformat()
        await asyncio.sleep(0)

        # Load image - preserve alpha channel if present (important for preprocess_image)
        # If RGBA with transparency, pipeline uses existing alpha; if RGB, runs background removal
        image = Image.open(image_path)

        # Clear CUDA memory aggressively before generation to prevent OOM
        clear_cuda_memory()
        print(f"[JOB {job_id[:8]}] CUDA memory cleared")

        # Run generation in executor
        def run_pipeline():
            # Check if pipeline is TRELLIS.2 (Trellis2ImageTo3DPipeline)
            is_trellis2 = isinstance(pipeline, Trellis2ImageTo3DPipeline)

            if use_v2 and is_trellis2:
                # Native TRELLIS.2-4B pipeline with 3 stages
                # Pipeline types based on resolution: 512, 1024_cascade, 1536_cascade
                resolution_to_pipeline = {512: '512', 1024: '1024_cascade', 1536: '1536_cascade'}
                pipeline_type = resolution_to_pipeline.get(params.resolution, '1024_cascade')
                print(f"[JOB {job_id[:8]}] Using native TRELLIS.2-4B pipeline ({pipeline_type}) with resolution={params.resolution}")
                return pipeline.run(
                    image,
                    seed=params.seed,
                    sparse_structure_sampler_params={
                        "steps": params.sparse_steps,
                        "guidance_strength": params.sparse_cfg,
                        "guidance_rescale": params.guidance_rescale_sparse,
                        "rescale_t": params.rescale_t_sparse,
                    },
                    shape_slat_sampler_params={
                        "steps": params.slat_steps,
                        "guidance_strength": params.slat_cfg,
                        "guidance_rescale": params.guidance_rescale_shape,
                        "rescale_t": params.rescale_t_shape,
                    },
                    tex_slat_sampler_params={
                        "steps": params.tex_steps,
                        "guidance_strength": params.tex_cfg,
                        "guidance_rescale": params.guidance_rescale_material,
                        "rescale_t": params.rescale_t_material,
                    },
                    pipeline_type=pipeline_type,
                )
            elif use_v2:
                # Fallback V2 pipeline (old API)
                print(f"[JOB {job_id[:8]}] Using fallback V2 pipeline (not native TRELLIS.2)")
                return pipeline.run(
                    image,
                    seed=params.seed,
                    sparse_structure_sampler_params={
                        "steps": params.sparse_steps,
                        "cfg_strength": params.sparse_cfg,
                    },
                    slat_sampler_params={
                        "steps": params.slat_steps,
                        "cfg_strength": params.slat_cfg,
                    },
                )
            else:
                # TRELLIS.1 pipeline (original)
                print(f"[JOB {job_id[:8]}] Using TRELLIS.1 pipeline")
                return pipeline.run(
                    image,
                    seed=params.seed,
                    sparse_structure_sampler_params={
                        "steps": params.sparse_steps,
                        "cfg_strength": params.sparse_cfg,
                    },
                    slat_sampler_params={
                        "steps": params.slat_steps,
                        "cfg_strength": params.slat_cfg,
                    },
                )
        outputs = await loop.run_in_executor(None, run_pipeline)

        jobs[job_id]["progress"] = 0.7
        jobs[job_id]["message"] = "Exporting files..."
        jobs[job_id]["updated_at"] = datetime.now().isoformat()
        await asyncio.sleep(0)

        # Save outputs
        output_prefix = OUTPUT_DIR / job_id
        output_files = {}

        # Check if outputs is TRELLIS.2 format (List[MeshWithVoxel]) or TRELLIS.1 format (dict)
        from trellis2.representations.mesh.base import MeshWithVoxel
        is_trellis2_output = isinstance(outputs, list) and len(outputs) > 0 and isinstance(outputs[0], MeshWithVoxel)

        if is_trellis2_output:
            # TRELLIS.2 returns List[MeshWithVoxel] - no gaussian output
            print(f"[JOB {job_id[:8]}] TRELLIS.2 output: {len(outputs)} MeshWithVoxel objects")
            output_files["ply"] = None  # TRELLIS.2 doesn't output gaussian splats
        else:
            # TRELLIS.1 returns dict with 'gaussian' and 'mesh' keys
            ply_path = f"{output_prefix}_gaussian.ply"
            await loop.run_in_executor(None, outputs['gaussian'][0].save_ply, ply_path)
            output_files["ply"] = ply_path

        jobs[job_id]["progress"] = 0.8
        jobs[job_id]["message"] = "Baking textures..."
        jobs[job_id]["updated_at"] = datetime.now().isoformat()
        await asyncio.sleep(0)

        try:
            print(f"[TEXTURE DEBUG] Image-to-3D: Starting GLB export with texture_size={params.texture_size}, pbr={params.pbr}, v2={use_v2}")

            def export_glb():
                if is_trellis2_output:
                    # TRELLIS.2: Use MeshWithVoxel.export() - same as test script
                    # This avoids CUDA threading issues that cause hangs with o_voxel.postprocess.to_glb()
                    print(f"[TEXTURE DEBUG] Using MeshWithVoxel.export() with decimation_target={params.decimation_target}, texture_size={params.texture_size}")
                    mesh = outputs[0]  # First mesh in batch
                    glb_path = f"{output_prefix}_model.glb"
                    mesh.export(
                        glb_path,
                        texture_size=params.texture_size,
                        decimation_target=params.decimation_target,
                        remesh=True,  # Use Dual Contouring remeshing to fix fragmentation
                        remesh_band=1.0,
                        remesh_project=0.0,
                        verbose=True,
                    )
                    return glb_path
                elif params.pbr:
                    # TRELLIS.1: Use PBR export with metallic/roughness textures
                    print(f"[TEXTURE DEBUG] Image-to-3D using PBR export: metallic={params.metallic}, roughness={params.roughness}")
                    return postprocessing_utils.to_glb_pbr(
                        outputs['gaussian'][0],
                        outputs['mesh'][0],
                        simplify=params.simplify,
                        texture_size=params.texture_size,
                        metallic=params.metallic,
                        roughness=params.roughness,
                        debug=True,
                        verbose=True,
                    )
                else:
                    # TRELLIS.1: Use standard export (albedo only)
                    return postprocessing_utils.to_glb(
                        outputs['gaussian'][0],
                        outputs['mesh'][0],
                        simplify=params.simplify,
                        texture_size=params.texture_size,
                        verbose=True,
                    )
            glb_result = await loop.run_in_executor(None, export_glb)

            # Handle different return types
            if is_trellis2_output:
                # TRELLIS.2 export() writes directly to file
                glb_path = glb_result
            else:
                # TRELLIS.1 returns trimesh object that needs export
                glb_path = f"{output_prefix}_model.glb"
                glb_result.export(glb_path)

            # Verify file size
            import os
            file_size_mb = os.path.getsize(glb_path) / (1024 * 1024)
            print(f"[TEXTURE DEBUG] Image-to-3D GLB file size: {file_size_mb:.2f} MB (PBR={params.pbr}, V2={use_v2})")

            output_files["glb"] = glb_path
        except Exception as e:
            print(f"GLB export failed: {e}")
            import traceback
            traceback.print_exc()
            output_files["glb"] = None
            output_files["glb_error"] = str(e)

        try:
            if is_trellis2_output:
                # TRELLIS.2: No gaussian, skip preview (mesh rendering requires different approach)
                print(f"[JOB {job_id[:8]}] TRELLIS.2 preview: skipping (no gaussian output)")
                output_files["preview"] = None
            else:
                # TRELLIS.1: Use gaussian rendering
                video_frames = render_utils.render_video(outputs['gaussian'][0], num_frames=1)
                preview_img = Image.fromarray(video_frames['color'][0])
                preview_path = f"{output_prefix}_preview.png"
                preview_img.save(preview_path)
                output_files["preview"] = preview_path
        except Exception as e:
            print(f"Preview generation failed: {e}")
            output_files["preview"] = None

        # Success - check if GLB export failed
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["progress"] = 1.0
        if output_files.get("glb") is None:
            jobs[job_id]["message"] = "Generation complete (GLB export failed)"
            jobs[job_id]["glb_error"] = output_files.get("glb_error", "Unknown error")
        else:
            jobs[job_id]["message"] = "Generation complete!"
        jobs[job_id]["outputs"] = output_files
        jobs[job_id]["updated_at"] = datetime.now().isoformat()

    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        jobs[job_id]["message"] = f"Failed: {str(e)}"
        jobs[job_id]["updated_at"] = datetime.now().isoformat()
        import traceback
        traceback.print_exc()
        # Clear CUDA memory after failure to prevent OOM on next attempt
        # For OOM errors, also clear the pipeline cache to free model memory
        if "out of memory" in str(e).lower():
            print(f"[JOB {job_id[:8]}] OOM detected - clearing pipeline cache")
            clear_pipeline_cache()
        else:
            clear_cuda_memory()
            print(f"[JOB {job_id[:8]}] CUDA memory cleared after failure")


# ===== API Endpoints =====

@app.get("/")
async def root():
    """Health check"""
    return {
        "status": "online",
        "version": "1.0.0",
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


@app.post("/api/generate/text")
async def generate_from_text(request: TextTo3DRequest, background_tasks: BackgroundTasks):
    """Generate 3D model from text prompt"""
    job_id = str(uuid.uuid4())

    # Create job
    jobs[job_id] = {
        "job_id": job_id,
        "type": "text_to_3d",
        "status": "queued",
        "progress": 0.0,
        "message": "Job queued",
        "outputs": None,
        "error": None,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "params": request.dict(),
    }

    # Start background task
    background_tasks.add_task(run_text_to_3d_job, job_id, request)

    return {"job_id": job_id}


@app.post("/api/generate/image")
async def generate_from_image(
    file: UploadFile = File(...),
    # All defaults match ImageTo3DRequest and official HuggingFace demo
    seed: int = 42,
    # Stage 1: Sparse Structure (matches pipeline.json sparse_structure_sampler)
    sparse_steps: int = 12,
    sparse_cfg: float = 7.5,
    guidance_rescale_sparse: float = 0.7,
    rescale_t_sparse: float = 5.0,
    # Stage 2: Shape Generation (matches pipeline.json shape_slat_sampler)
    slat_steps: int = 12,
    slat_cfg: float = 7.5,
    guidance_rescale_shape: float = 0.5,
    rescale_t_shape: float = 3.0,
    # Stage 3: Material Generation (matches pipeline.json tex_slat_sampler)
    tex_steps: int = 12,
    tex_cfg: float = 1.0,
    guidance_rescale_material: float = 0.0,
    rescale_t_material: float = 3.0,
    # Output settings (match base.py MeshWithVoxel.export defaults)
    decimation_target: int = 300000,
    simplify: float = 0.95,
    texture_size: int = 2048,
    # Pipeline settings
    pipeline_version: str = "v2",
    resolution: int = 1024,
):
    """Generate 3D model from image (TRELLIS.2 with PBR textures)"""
    job_id = str(uuid.uuid4())

    # Save uploaded image
    image_path = UPLOAD_DIR / f"{job_id}.png"
    with open(image_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Create params with all TRELLIS.2 parameters
    params = ImageTo3DRequest(
        seed=seed,
        # Stage 1
        sparse_steps=sparse_steps,
        sparse_cfg=sparse_cfg,
        guidance_rescale_sparse=guidance_rescale_sparse,
        rescale_t_sparse=rescale_t_sparse,
        # Stage 2
        slat_steps=slat_steps,
        slat_cfg=slat_cfg,
        guidance_rescale_shape=guidance_rescale_shape,
        rescale_t_shape=rescale_t_shape,
        # Stage 3
        tex_steps=tex_steps,
        tex_cfg=tex_cfg,
        guidance_rescale_material=guidance_rescale_material,
        rescale_t_material=rescale_t_material,
        # Output
        decimation_target=decimation_target,
        simplify=simplify,
        texture_size=texture_size,
        # Pipeline
        pipeline_version=pipeline_version,
        resolution=resolution,
    )

    # Create job
    jobs[job_id] = {
        "job_id": job_id,
        "type": "image_to_3d",
        "status": "queued",
        "progress": 0.0,
        "message": "Job queued",
        "outputs": None,
        "error": None,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "params": params.dict(),
    }

    # Start background task
    asyncio.create_task(run_image_to_3d_job(job_id, str(image_path), params))

    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Get status of a generation job"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    return jobs[job_id]


@app.get("/api/jobs")
async def list_jobs():
    """List all jobs"""
    # Debug: log job statuses on each poll
    statuses = [f"{j['job_id'][:8]}:{j['status']}" for j in jobs.values()]
    if statuses:
        print(f"[API /jobs] Returning {len(jobs)} jobs: {', '.join(statuses)}")
    return {"jobs": list(jobs.values())}


@app.get("/api/download/{job_id}/{file_type}")
async def download_file(job_id: str, file_type: str):
    """Download generated file (glb, ply, or preview)"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not completed")

    if file_type not in job["outputs"]:
        raise HTTPException(status_code=404, detail=f"File type {file_type} not found")

    file_path = job["outputs"][file_type]
    if file_path is None or not Path(file_path).exists():
        raise HTTPException(status_code=404, detail=f"File not found: {file_type}")

    # Set correct MIME types for 3D files
    media_types = {
        "glb": "model/gltf-binary",
        "ply": "application/octet-stream",
        "png": "image/png"
    }
    media_type = media_types.get(file_type, "application/octet-stream")

    return FileResponse(file_path, filename=f"{job_id}.{file_type}", media_type=media_type)


@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a job and its outputs"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]

    # Delete output files
    if job.get("outputs"):
        for file_path in job["outputs"].values():
            if file_path and Path(file_path).exists():
                Path(file_path).unlink()

    # Delete job
    del jobs[job_id]

    return {"message": "Job deleted"}


@app.delete("/api/jobs")
async def clear_all_jobs():
    """Clear all jobs and their outputs"""
    global jobs

    deleted_count = len(jobs)

    # Delete all output files
    for job in jobs.values():
        if job.get("outputs"):
            for file_path in job["outputs"].values():
                if file_path and Path(file_path).exists():
                    try:
                        Path(file_path).unlink()
                    except Exception as e:
                        logger.warning(f"Failed to delete {file_path}: {e}")

    # Clear jobs dict
    jobs.clear()

    # Clear CUDA memory aggressively
    clear_cuda_memory()

    logger.info(f"Cleared {deleted_count} jobs")
    return {"message": f"Cleared {deleted_count} jobs"}


if __name__ == "__main__":
    import uvicorn
    print("=" * 70)
    print("TRELLIS GUI Backend Server")
    print("=" * 70)
    print(f"Session Log: {_SESSION_LOG_FILE}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Output Directory: {OUTPUT_DIR.absolute()}")
    print(f"Upload Directory: {UPLOAD_DIR.absolute()}")
    print("=" * 70)
    print("\nStarting server on http://localhost:8000")
    print("API Docs: http://localhost:8000/docs")
    print("\n")

    logger.info(f"Server starting - CUDA: {torch.cuda.is_available()}")

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
