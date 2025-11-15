# freeze-pipeline: Reproducing and Profiling the FreeZe Pipeline

**freeze-pipeline** is an independent reproduction and experimental implementation of the **FreeZe** pipeline — *Training-Free Zero-Shot 6D Object Pose Estimation using Foundation Models*.  
The project focuses on **understanding, modularizing, and benchmarking** each stage of FreeZe, beginning with the **GeDi geometric foundation model**.  
Unlike model-free approaches, FreeZe relies on **3D model–based zero-shot registration** using pretrained geometric and visual encoders (GeDi + DINOv2), without any task-specific training.  

The goal of this repo is to:
- Run **GeDi** inference in real-time on standard object datasets (e.g., LM-O, YCB-V),
- Integrate **DINOv2** visual features and perform 3D–3D registration,
- Assemble the **end-to-end FreeZe pipeline**, from feature extraction to pose refinement,
- Optimize and profile components (e.g., distilling GeDi → *dGeDi*).

---

## Overview

FreeZe estimates the 6D pose of a known 3D object in a scene **without training**:
1. **Feature Extraction**  
   - *Geometric:* GeDi encodes each 3D point into a distinct shape descriptor.  
   - *Visual:* DINOv2 extracts patch-level texture embeddings from RGB images.
2. **Feature Fusion**  
   - Normalized geometric and visual descriptors are concatenated to form fused 3D features.
3. **Pose Estimation**  
   - 3D–3D correspondences are found using nearest-neighbor search in fused feature space;  
     RANSAC estimates coarse pose.
4. **Pose Refinement**  
   - ICP and symmetry-aware refinement produce the final alignment.

---

## High-Level Checklist

| Stage | Description | Status |
|--------|--------------|--------|
| **GeDi Integration** | Build, test, and benchmark GeDi on RTX 4090 | ✅ Done |
| **Real-time Profiling** | Measure inference latency per object | ✅ Done |
| **Dataset Setup** | Integrate LM-O / YCB-V for controlled tests | ✅ Done |
| **Query Object Extraction** | extraxting deatures | ✅ Done |


Completed in QOP
# 1. GeDi processing (geometric features)
# 2. Multi-view rendering with masks
# 3. DINOv2 feature extraction (6 views)
# 4. Feature back-projection
# 5. Multi-view aggregation

---

## Structure

It uses three fully isolated environments:

`master_env` – runs the controller pipeline

`gedi_env` – GeDi inference

`dino_env` – DINOv2 inference

Each model is executed inside its own environment to avoid dependency conflicts.

```
freeze-pipeline/
│
├── integration/
│   └── run_pipeline.py         # main controller (calls both models)
│
├── gedi/
│   ├── .venv/              # GeDi environment
│   ├── gedi_process_object.py
│   └── run_gedi_inference.py
│
├── dino/
│   ├── .venv/              # DINO environment
│   └── demo_infer.py
│
├── data/
│   └── 20objects/
│       └── data/
│           └── Kinfu_*/        # object folders
│               ├── object.xyz
│               ├── rgb_noseg/
│               ├── depth_noseg/
│               ├── mask/
│               └── info/
│
└── processed/                  # all output is saved here
    ├── <object>_pc.ply
    ├── <object>_gedi.npy
    └── <object>_dino.npy

```

## References


```bibtex
@inproceedings{Poiesi2021,
  title   = {Learning general and distinctive 3D local deep descriptors for point cloud registration},
  author  = {Poiesi, Fabio and Boscaini, Davide},
  booktitle = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year    = {(early access) 2022}
}
```


