import torch
vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(f"VRAM: {vram} GB")
print(f">= 24: {vram >= 24}")
print(f">= 23: {vram >= 23}")
