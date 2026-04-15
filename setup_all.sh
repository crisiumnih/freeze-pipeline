#!/bin/bash
set -e

echo "=========================================="
echo "FreeZe v2 - Complete Setup"
echo "=========================================="
echo

# Check for uv
if ! command -v uv &> /dev/null; then
    echo "Error: uv not found. Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "✓ uv found"
echo

# Check for GeDi source
if [ ! -d "GeDi" ]; then
    echo "Cloning GeDi repository..."
    git clone https://github.com/fabiopoiesi/gedi.git GeDi
else
    echo "✓ GeDi repository found"
fi
echo

# Setup each environment
echo "1/4 Setting up GeDi environment..."
cd environments/gedi
bash setup.sh
cd ../..
echo

echo "2/4 Setting up DINOv2 environment..."
cd environments/dinov2
bash setup.sh
cd ../..
echo

echo "3/4 Setting up SAM2 environment..."
cd environments/sam2
bash setup.sh
cd ../..
echo

echo "4/4 Setting up main freeze package..."
cd freeze
uv venv --python 3.10 .venv
source .venv/bin/activate
uv pip install -e .
cd ..
echo

echo "=========================================="
echo "✓ Setup complete!"
echo "=========================================="
echo
echo "Next steps:"
echo "1. Download HOPE dataset to: data/hope/"
echo "2. Test environments: bash test_all.sh"
echo "3. Run demo: cd freeze && source .venv/bin/activate && python ../scripts/demo.py"
echo
