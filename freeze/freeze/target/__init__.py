"""
Target scene processing for FreeZe pipeline.
Processes RGB-D images to extract features from object proposals.
"""

from .segmentation import SegmentationProcessor
from .depth_lifting import DepthLifter
from .geometric import TargetGeometricProcessor
from .visual import VisualProcessor
from .fusion import TargetFeatureFusion
from .pipeline import TargetPipeline

__all__ = [
    'SegmentationProcessor',
    'DepthLifter',
    'TargetGeometricProcessor',
    'VisualProcessor',
    'TargetFeatureFusion',
    'TargetPipeline',
]
