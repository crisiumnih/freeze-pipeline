# FreeZe v2 - Complete Setup Guide

This guide will walk you through setting up the entire FreeZe pipeline from scratch.

## Prerequisites

### System Requirements
- **OS**: Linux (tested on Ubuntu 20.04+)
- **GPU**: NVIDIA GPU with CUDA support (tested on RTX 4090)
- **CUDA**: Version 11.x or 12.x
- **RAM**: 16GB+ recommended
- **Disk**: ~20GB for models and dependencies

### Required Software
1. **Python**: Need both Python 3.8 and 3.10+
2. **uv**: Modern Python package installer
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
3. **Git**: For cloning repositories
4. **CUDA Toolkit**: For GPU acceleration

---

## Step-by-Step Setup

### Step 1: Navigate to Project Directory

```bash
cd /home/sra/Prajwal/freeze_v2
```

### Step 2: Run Automated Setup

```bash
./setup_all.sh
```

This script will:
1. Clone GeDi repository
2. Create GeDi environment (Python 3.8)
   - Download PyTorch 1.8.1 wheel
   - Download Open3D 0.15.1 wheel
   - Build PointNet2 ops
   - Download GeDi pretrained model
3. Create DINOv2 environment (Python 3.10)
   - Install modern PyTorch
   - Download DINOv2 model (auto on first use)
4. Create SAM2 environment (Python 3.10)
   - Install SAM2
   - Download SAM2.1 checkpoint
5. Setup main freeze package

**Expected time**: 10-15 minutes (depending on internet speed)

### Step 3: Verify Installation

```bash
./test_all.sh
```

You should see:
```
✓ All tests passed! GeDi environment is ready.
✓ All tests passed! DINOv2 environment is ready.
✓ All tests passed! SAM2 environment is ready.
```

### Step 4: Test Client Wrappers

```bash
cd freeze
source .venv/bin/activate
python ../scripts/test_clients.py
```

Expected output:
```
GeDi         ✓ PASS
DINOv2       ✓ PASS
SAM2         ✓ PASS
✓ All client tests passed!
```

---

## Download HOPE Dataset

### Option 1: Direct Download (Recommended)

1. Visit: https://github.com/swtyree/hope-dataset
2. Download the dataset (RGB-D images + 3D models)
3. Extract to: `freeze_v2/data/hope/`

Expected structure:
```
data/hope/
├── models/
│   ├── AlphabetSoup.ply
│   ├── BBQSauce.ply
│   ├── Butter.ply
│   └── ... (28 objects total)
├── rgb/
│   ├── 000001.png
│   ├── 000002.png
│   └── ...
├── depth/
│   ├── 000001.png
│   ├── 000002.png
│   └── ...
└── poses/
    └── ... (ground truth poses)
```

### Option 2: Use Script (TODO)

```bash
python scripts/download_hope.py
```

---

## Troubleshooting

### GeDi Environment Issues

**Problem**: PointNet2 compilation fails
```
Solution:
cd GeDi/backbones/pointnet2_ops_lib
# Check CUDA architecture matches your GPU
# RTX 4090 = sm_89, but compile as sm_86
sed -i 's/TORCH_CUDA_ARCH_LIST.*/TORCH_CUDA_ARCH_LIST"] = "8.6"/' setup.py
rm -rf build
uv pip install --no-build-isolation .
```

**Problem**: PyTorch 1.8.1 wheel not found
```
Solution:
cd environments/gedi
wget https://github.com/isl-org/open3d_downloads/releases/download/torch1.8.1/torch-1.8.1-cp38-cp38-linux_x86_64.whl
```

### DINOv2 Environment Issues

**Problem**: Model download fails
```
Solution:
# Download manually
mkdir -p ~/.cache/torch/hub/checkpoints/
cd ~/.cache/torch/hub/checkpoints/
wget https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth
```

### SAM2 Environment Issues

**Problem**: SAM2 installation fails
```
Solution:
cd environments/sam2
source .venv/bin/activate
pip install git+https://github.com/facebookresearch/segment-anything-2.git
```

**Problem**: Checkpoint download fails
```
Solution:
cd environments/sam2/checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```

### General Issues

**Problem**: "uv not found"
```
Solution:
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc  # or ~/.zshrc
```

**Problem**: CUDA out of memory
```
Solution:
# Reduce batch size or image resolution in config
# Or use a smaller SAM2 model: sam2.1_hiera_small.pt
```

---

## Verifying Setup

### Quick Verification Checklist

- [ ] `./test_all.sh` passes all tests
- [ ] `python scripts/test_clients.py` shows all ✓ PASS
- [ ] GeDi environment at `environments/gedi/.venv/`
- [ ] DINOv2 environment at `environments/dinov2/.venv/`
- [ ] SAM2 environment at `environments/sam2/.venv/`
- [ ] Main freeze package at `freeze/.venv/`
- [ ] GeDi source code at `GeDi/`
- [ ] HOPE dataset at `data/hope/`

### Test Individual Components

**Test GeDi:**
```bash
cd environments/gedi
source .venv/bin/activate
python test_gedi.py
```

**Test DINOv2:**
```bash
cd environments/dinov2
source .venv/bin/activate
python test_dinov2.py
```

**Test SAM2:**
```bash
cd environments/sam2
source .venv/bin/activate
python test_sam2.py
```

---

## Next Steps

Once setup is complete:

1. **Read the README**: `less README.md`
2. **Explore the structure**: `tree -L 2 -I '.venv|__pycache__'`
3. **Run a demo** (coming soon): `python scripts/demo.py`
4. **Process a query object** (coming soon): `python scripts/process_query.py`
5. **Estimate pose** (coming soon): `python scripts/estimate_pose.py`

---

## Manual Setup (Alternative)

If automated setup fails, you can set up each environment manually:

### GeDi Environment
```bash
cd environments/gedi
uv venv --python 3.8 .venv
source .venv/bin/activate

# Download wheels
wget https://github.com/isl-org/open3d_downloads/releases/download/torch1.8.1/torch-1.8.1-cp38-cp38-linux_x86_64.whl
wget https://github.com/isl-org/open3d_downloads/releases/download/torch1.8.1/open3d-0.15.1-cp38-cp38-manylinux_2_27_x86_64.whl

# Install
uv pip install ./torch-1.8.1-cp38-cp38-linux_x86_64.whl
uv pip install ./open3d-0.15.1-cp38-cp38-manylinux_2_27_x86_64.whl
uv pip install -r requirements.txt

# Build PointNet2
cd ../../GeDi/backbones/pointnet2_ops_lib
sed -i 's/TORCH_CUDA_ARCH_LIST.*/TORCH_CUDA_ARCH_LIST"] = "8.6"/' setup.py
rm -rf build
uv pip install --no-build-isolation .

# Download model
cd ../../..
cd GeDi
python download_data.py
```

### DINOv2 Environment
```bash
cd environments/dinov2
uv venv --python 3.10 .venv
source .venv/bin/activate
uv pip install -e .
```

### SAM2 Environment
```bash
cd environments/sam2
uv venv --python 3.10 .venv
source .venv/bin/activate
uv pip install torch torchvision numpy pillow opencv-python
uv pip install git+https://github.com/facebookresearch/segment-anything-2.git

mkdir -p checkpoints
cd checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```

### Main Package
```bash
cd freeze
uv venv --python 3.10 .venv
source .venv/bin/activate
uv pip install -e .
```

---


