"""Download BiRefNet model weights to local cache."""
import os
os.environ['HF_HOME'] = os.path.abspath('./models')

from transformers import AutoModelForImageSegmentation

print("[BiRefNet] Downloading ZhengPeng7/BiRefNet model...")
model = AutoModelForImageSegmentation.from_pretrained(
    'ZhengPeng7/BiRefNet', trust_remote_code=True
)
print(f"[BiRefNet] Model type: {type(model).__name__}")
print(f"[BiRefNet] Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
print(f"[BiRefNet] Download complete. Model cached in ./models/")
