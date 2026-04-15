#!/bin/bash
set -e

echo "Setting up DINOv2 environment (Python 3.10+)..."

# Create virtual environment
uv venv --python 3.10 .venv
source .venv/bin/activate

echo "Installing dependencies..."
uv pip install -e .

echo "Testing DINOv2 model download..."
python -c "import torch; torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')"

echo "✓ DINOv2 environment setup complete!"
echo "Test with: .venv/bin/python test_dinov2.py"
