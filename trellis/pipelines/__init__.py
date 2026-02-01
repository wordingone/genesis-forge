# Lazy imports to allow environment variables to be set before transformers is loaded
# This is critical for offline mode with custom cache directories
from . import samplers

# These classes are imported lazily via __getattr__ to avoid early transformers import
__all__ = [
    'TrellisImageTo3DPipeline',
    'Trellis2ImageTo3DPipeline',
    'TrellisImageTo3DPipelineV2',
    'TrellisTextTo3DPipeline',
    'samplers',
    'from_pretrained',
]

_lazy_imports = {}

def __getattr__(name):
    if name == 'TrellisImageTo3DPipeline':
        if name not in _lazy_imports:
            from .trellis_image_to_3d import TrellisImageTo3DPipeline
            _lazy_imports[name] = TrellisImageTo3DPipeline
        return _lazy_imports[name]
    elif name == 'Trellis2ImageTo3DPipeline':
        if name not in _lazy_imports:
            # Import from trellis2 module (correct TRELLIS.2 implementation)
            from trellis2.pipelines.trellis2_image_to_3d import Trellis2ImageTo3DPipeline
            _lazy_imports[name] = Trellis2ImageTo3DPipeline
        return _lazy_imports[name]
    elif name == 'TrellisImageTo3DPipelineV2':
        # Backward compatibility alias
        return __getattr__('Trellis2ImageTo3DPipeline')
    elif name == 'TrellisTextTo3DPipeline':
        if name not in _lazy_imports:
            from .trellis_text_to_3d import TrellisTextTo3DPipeline
            _lazy_imports[name] = TrellisTextTo3DPipeline
        return _lazy_imports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def from_pretrained(path: str):
    """
    Load a pipeline from a model folder or a Hugging Face model hub.

    Args:
        path: The path to the model. Can be either local path or a Hugging Face model name.
    """
    import os
    import json
    is_local = os.path.exists(f"{path}/pipeline.json")

    if is_local:
        config_file = f"{path}/pipeline.json"
    else:
        from huggingface_hub import hf_hub_download
        config_file = hf_hub_download(path, "pipeline.json")

    with open(config_file, 'r') as f:
        config = json.load(f)
    return globals()[config['name']].from_pretrained(path)
