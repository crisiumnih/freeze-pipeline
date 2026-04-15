# FreeZe-v2: Training-Free Zero-Shot 6D Pose Estimation

An enhanced implementation of the **FreeZe** pipeline ([Caraffa et al., 2024](https://arxiv.org/abs/2312.00947)) for training-free, zero-shot 6D object pose estimation using geometric and visual foundation models.

## What's New in v2

| Improvement | Description |
|-------------|-------------|
| **Mesh Normalization** | Critical fix: normalizes query meshes to a consistent scale before feature extraction — enables RANSAC to find correspondences across scale mismatches |
| **ICP Refinement** | Optional iterative closest point refinement after RANSAC for improved pose accuracy |
| **Enhanced Matching** | Lowe's ratio test + optional mutual nearest-neighbor check for cleaner correspondences |
| **SAM2 Integration** | Uses SAM2 automatic segmentation for scene proposals |

## Architecture

```
freeze_v2/
├── environments/          # Isolated model environments
│   ├── gedi/              # Python 3.8 + PyTorch 1.8 (GeDi geometric features)
│   ├── dinov2/            # Python 3.10 + PyTorch 2.0 (DINOv2 visual features)
│   └── sam2/              # Python 3.10 + PyTorch 2.0 (SAM2 segmentation)
│
├── freeze/                # Main package (Python 3.10)
│   └── freeze/
│       ├── models/        # Clean client wrappers (subprocess-based)
│       ├── query/         # Query object processing pipeline
│       ├── target/        # Target scene processing pipeline
│       ├── matching/      # Pose estimation (RANSAC + ICP)
│       └── visualization/ # Pipeline visualization + GIF generation
│
└── scripts/               # Demo and test scripts
```

## Pipeline

```
Query Object (.obj)
    ↓ MeshNormalizer      → Normalize to 0.2m scale
    ↓ GeometricProcessor  → GeDi features (32-dim)
    ↓ MultiViewRenderer   → Render 8+ views
    ↓ VisualProcessor     → DINOv2 features (1024-dim)
    ↓ FeatureBackProjector → Back-project to 3D
    ↓ FeatureFusion       → Fused features (1056-dim)

Target Scene (RGB-D)
    ↓ SAM2                → Segment proposals
    ↓ DepthLifter         → Lift to 3D point clouds
    ↓ GeometricProcessor  → GeDi features (32-dim)
    ↓ VisualProcessor     → DINOv2 crop features (1024-dim)
    ↓ FeatureFusion       → Fused features (1056-dim)

Matching
    ↓ CorrespondenceMatcher → FLANN + ratio test
    ↓ RANSACEstimator       → 6-DoF pose
    ↓ ICPRefiner (optional) → Refined pose
```

## Setup

See [SETUP_GUIDE.md](SETUP_GUIDE.md) for full installation instructions.

```bash
chmod +x setup_all.sh
./setup_all.sh
```

Requires 3 isolated environments due to conflicting dependencies (GeDi needs PyTorch 1.8, SAM2/DINOv2 need PyTorch 2.0+).

## Quick Start

```bash
cd freeze
source .venv/bin/activate

# Geometric-only (fast, ~5s)
python ../scripts/test_query_pipeline.py

# Full fused pipeline with visualization
python ../scripts/demo_visual.py \
    --query-mesh data/hope/meshes/eval/Mustard.obj \
    --target-rgb data/hope/hope_image/valid/scene_0000/0000_rgb.jpg \
    --target-depth data/hope/hope_image/valid/scene_0000/0000_depth.png
```

## Results

Evaluated on HOPE dataset (preliminary, full benchmark in progress):

| Mode | Inliers | RMSE |
|------|---------|------|
| Geometric (32-dim) | 25 | 6.8mm |
| Fused (1056-dim) | **48** | **6.5mm** |

Visual features improve inlier count by ~2x with slightly better accuracy.

## Dataset: HOPE

Download the [HOPE dataset](https://github.com/swtyree/hope-dataset):
```
data/hope/
├── meshes/eval/     # 3D CAD models (.obj)
└── hope_image/      # RGB-D scenes with GT poses
    ├── valid/
    └── test/
```

## Citation

If you use this code, please also cite the original FreeZe paper:

```bibtex
@inproceedings{caraffa2024freeze,
  title={FreeZe: Training-free zero-shot 6D pose estimation with geometric and vision foundation models},
  author={Caraffa, Andrea and Boscaini, Davide and Hamza, Amir and Poiesi, Fabio},
  booktitle={arXiv preprint arXiv:2312.00947},
  year={2024}
}
```
