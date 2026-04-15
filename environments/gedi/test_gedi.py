#!/usr/bin/env python3
"""
Test GeDi environment setup
"""
import numpy as np
import torch
import open3d as o3d
import sys
from pathlib import Path

# Add GeDi to path
GEDI_PATH = Path(__file__).resolve().parent.parent.parent / 'GeDi'
sys.path.insert(0, str(GEDI_PATH))
print(f"GeDi path: {GEDI_PATH}", file=sys.stderr)

print("Testing GeDi environment...")
print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Open3D version: {o3d.__version__}")

# Test imports
try:
    from gedi import GeDi
    print("✓ GeDi model imported successfully")
except ImportError as e:
    print(f"✗ Failed to import GeDi model: {e}")
    sys.exit(1)

try:
    import pointnet2_ops
    print("✓ PointNet2 ops imported successfully")
except ImportError as e:
    print(f"✗ Failed to import PointNet2 ops: {e}")
    sys.exit(1)

# Test model loading
try:
    checkpoint_path = GEDI_PATH / 'data/chkpts/3dmatch/chkpt.tar'
    if not checkpoint_path.exists():
        print(f"✗ Checkpoint not found: {checkpoint_path}")
        print("  Run: cd GeDi && python download_data.py")
        sys.exit(1)

    config = {
        'dim': 32,
        'samples_per_batch': 500,
        'samples_per_patch_lrf': 4000,
        'samples_per_patch_out': 512,
        'r_lrf': .5,
        'fchkpt_gedi_net': str(checkpoint_path)
    }
    model = GeDi(config=config)
    print("✓ GeDi model loaded successfully")
except Exception as e:
    print(f"✗ Failed to load model: {e}")
    sys.exit(1)

# Test inference
try:
    # Create a simple test point cloud
    test_points = np.random.randn(100, 3).astype(np.float32)
    pts_tensor = torch.tensor(test_points).cuda() if torch.cuda.is_available() else torch.tensor(test_points)

    with torch.no_grad():
        features = model.compute(pts=pts_tensor, pcd=pts_tensor)

    print(f"✓ Inference successful! Features shape: {features.shape}")
    assert features.shape == (100, 32), f"Expected shape (100, 32), got {features.shape}"
    print("✓ Feature dimension correct (32)")

except Exception as e:
    print(f"✗ Inference failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*50)
print("✓ All tests passed! GeDi environment is ready.")
print("="*50)
