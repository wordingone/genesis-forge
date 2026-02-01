"""Test if nested autocast properly resumes after disabled context."""
import torch

print("Testing nested autocast resume behavior:")

def simulate_sample_shape_slat_cascade():
    print(f"\n1. Before outer autocast: {torch.is_autocast_enabled()}")

    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=True):
        print(f"2. Inside outer autocast (before LR sampling): {torch.is_autocast_enabled()}")

        # Simulate LR sampling
        # ...

        # Simulate upsample with disabled autocast
        print("   [Entering upsample with disabled autocast]")
        with torch.autocast('cuda', enabled=False):
            print(f"3. Inside disabled context (upsample): {torch.is_autocast_enabled()}")
        print(f"4. After disabled context exits (should resume): {torch.is_autocast_enabled()}")

        # Simulate HR sampling
        print("   [Doing HR sampling here]")
        print(f"5. During HR sampling: {torch.is_autocast_enabled()}")

    print(f"6. After outer autocast: {torch.is_autocast_enabled()}")

# Call from outside any context
simulate_sample_shape_slat_cascade()

print("\n---")
print("Now simulating as called from run_generation_test.py (nested call):")

def outer_caller():
    print(f"\nA. Before outer autocast in caller: {torch.is_autocast_enabled()}")
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=True):
        print(f"B. Inside caller's autocast: {torch.is_autocast_enabled()}")
        simulate_sample_shape_slat_cascade()
        print(f"C. After cascade returns: {torch.is_autocast_enabled()}")

outer_caller()
