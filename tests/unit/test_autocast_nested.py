import torch

print("Testing nested autocast behavior:")

print(f"\n1. Outside all contexts: {torch.is_autocast_enabled()}")

with torch.autocast('cuda', dtype=torch.bfloat16, enabled=True):
    print(f"2. Inside outer autocast: {torch.is_autocast_enabled()}")

    with torch.autocast('cuda', enabled=False):
        print(f"3. Inside nested disabled autocast: {torch.is_autocast_enabled()}")

    print(f"4. After nested disabled autocast (still inside outer): {torch.is_autocast_enabled()}")

print(f"5. After outer autocast: {torch.is_autocast_enabled()}")
