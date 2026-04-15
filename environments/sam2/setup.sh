#!/bin/bash
set -e

echo "Setting up SAM2 environment (Python 3.10+)..."

# Create virtual environment
uv venv --python 3.10 .venv
source .venv/bin/activate

echo "Installing dependencies..."
uv pip install torch torchvision numpy pillow opencv-python

echo "Installing SAM2..."
uv pip install git+https://github.com/facebookresearch/segment-anything-2.git

echo "Downloading SAM2 checkpoint..."
mkdir -p checkpoints
cd checkpoints
if [ ! -f "sam2.1_hiera_large.pt" ]; then
    wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
fi
cd ..

echo "✓ SAM2 environment setup complete!"
echo "Test with: .venv/bin/python test_sam2.py"
