"""Test TRELLIS.2 pipeline loading."""
import os
import sys

# Set HF cache to local path
os.environ['HF_HOME'] = 'B:/M/ArtificialArchitecture/spatial/trellis-forge/models/hub'
os.environ['TRANSFORMERS_CACHE'] = 'B:/M/ArtificialArchitecture/spatial/trellis-forge/models/hub'

# Add trellis2 to path
sys.path.insert(0, 'B:/M/ArtificialArchitecture/spatial/trellis-forge')

print("Testing TRELLIS.2 pipeline loading...")

try:
    from trellis2.pipelines import Trellis2ImageTo3DPipeline

    model_path = "B:/M/ArtificialArchitecture/spatial/trellis-forge/models/hub/TRELLIS.2-4B"
    print(f"Loading from: {model_path}")

    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(model_path)
    print("SUCCESS: Pipeline loaded!")
    print(f"Models: {list(pipeline.models.keys())}")

except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()
