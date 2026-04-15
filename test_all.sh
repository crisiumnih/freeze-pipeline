#!/bin/bash
set -e

echo "=========================================="
echo "Testing all environments"
echo "=========================================="
echo

echo "Testing GeDi..."
cd environments/gedi
.venv/bin/python test_gedi.py
cd ../..
echo

echo "Testing DINOv2..."
cd environments/dinov2
.venv/bin/python test_dinov2.py
cd ../..
echo

echo "Testing SAM2..."
cd environments/sam2
.venv/bin/python test_sam2.py
cd ../..
echo

echo "=========================================="
echo "✓ All environment tests passed!"
echo "=========================================="
