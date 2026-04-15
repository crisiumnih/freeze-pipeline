#!/usr/bin/env python3
"""
Test SAM2 environment setup
"""
import numpy as np
import torch
import sys
from pathlib import Path

print("Testing SAM2 environment...")
print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Test SAM2 import
try:
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    print("✓ SAM2 imported successfully")
except ImportError as e:
    print(f"✗ Failed to import SAM2: {e}")
    sys.exit(1)

# Test model loading
try:
    checkpoint_path = Path(__file__).parent / "checkpoints/sam2.1_hiera_large.pt"
    if not checkpoint_path.exists():
        print(f"✗ Checkpoint not found: {checkpoint_path}")
        print("  Run setup.sh to download checkpoint")
        sys.exit(1)

    print("\nLoading SAM2 model...")
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    sam2 = build_sam2(model_cfg, str(checkpoint_path), device="cuda")
    mask_generator = SAM2AutomaticMaskGenerator(sam2)
    print("✓ SAM2 model loaded successfully")
except Exception as e:
    print(f"✗ Failed to load model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test inference
try:
    print("\nTesting inference...")
    # Create a dummy image
    import cv2
    dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    masks = mask_generator.generate(dummy_image)
    print(f"✓ Inference successful! Generated {len(masks)} masks")

except Exception as e:
    print(f"✗ Inference failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*50)
print("✓ All tests passed! SAM2 environment is ready.")
print("="*50)
