#!/usr/bin/env python3
"""
Test DINOv2 environment setup
"""
import numpy as np
import torch
from PIL import Image
import sys

print("Testing DINOv2 environment...")
print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Test model loading
try:
    print("\nLoading DINOv2 model...")
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
    model = model.cuda() if torch.cuda.is_available() else model
    model.eval()
    print("✓ DINOv2 model loaded successfully")
except Exception as e:
    print(f"✗ Failed to load model: {e}")
    sys.exit(1)

# Test inference
try:
    print("\nTesting inference...")
    # Create a dummy image
    img = Image.new('RGB', (224, 224), color=(128, 128, 128))

    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize(518),
        transforms.CenterCrop(518),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    x = transform(img).unsqueeze(0)
    x = x.cuda() if torch.cuda.is_available() else x

    with torch.no_grad():
        output = model.get_intermediate_layers(x, n=1)[0]

    print(f"✓ Inference successful! Output shape: {output.shape}")
    print(f"  Expected: [1, 1370, 1024] (CLS + 37×37 patches)")

    patch_tokens = output[:, 1:, :]
    print(f"  Patch tokens shape: {patch_tokens.shape}")

except Exception as e:
    print(f"✗ Inference failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*50)
print("✓ All tests passed! DINOv2 environment is ready.")
print("="*50)
