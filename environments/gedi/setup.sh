#!/bin/bash
set -e

echo "Setting up GeDi environment (Python 3.8)..."

# Create virtual environment
uv venv --python 3.8 .venv
source .venv/bin/activate

echo "Installing PyTorch 1.8.1 wheel..."
# Download PyTorch 1.8.1 wheel if not present
TORCH_WHEEL="torch-1.8.1-cp38-cp38-linux_x86_64.whl"
if [ ! -f "$TORCH_WHEEL" ]; then
    echo "Downloading PyTorch 1.8.1..."
    wget https://github.com/isl-org/open3d_downloads/releases/download/torch1.8.1/$TORCH_WHEEL
fi
uv pip install ./$TORCH_WHEEL

echo "Installing Open3D 0.15.1 wheel..."
# Download Open3D wheel if not present
OPEN3D_WHEEL="open3d-0.15.1-cp38-cp38-manylinux_2_27_x86_64.whl"
if [ ! -f "$OPEN3D_WHEEL" ]; then
    echo "Downloading Open3D 0.15.1..."
    wget https://github.com/isl-org/open3d_downloads/releases/download/torch1.8.1/$OPEN3D_WHEEL
fi
uv pip install ./$OPEN3D_WHEEL

echo "Installing other dependencies..."
uv pip install -r requirements.txt

echo "Building PointNet2 ops..."
cd ../../GeDi/backbones/pointnet2_ops_lib || {
    echo "Error: GeDi source code not found. Please clone GeDi repository."
    exit 1
}

# Modify setup.py for RTX 4090
sed -i 's/os.environ\["TORCH_CUDA_ARCH_LIST"\] = .*/os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6"/' setup.py

rm -rf build
uv pip install --no-build-isolation .

cd ../../../environments/gedi

echo "Downloading GeDi pretrained model..."
cd ../..
if [ ! -d "GeDi/data/chkpts" ]; then
    cd GeDi
    python download_data.py
    cd ..
fi
cd environments/gedi

echo "✓ GeDi environment setup complete!"
echo "Test with: .venv/bin/python test_gedi.py"
