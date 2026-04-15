"""
Complete target scene processing pipeline.
Orchestrates segmentation, depth lifting, and feature extraction.
"""
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import logging

from .segmentation import SegmentationProcessor
from .depth_lifting import DepthLifter
from .geometric import TargetGeometricProcessor
from .visual import VisualProcessor
from .fusion import TargetFeatureFusion

logger = logging.getLogger(__name__)


class TargetPipeline:
    """Complete pipeline for processing target scenes."""

    def __init__(self,
                 sam2_client,
                 gedi_client,
                 dinov2_client=None,
                 use_visual: bool = False,
                 camera_intrinsics: Optional[Tuple[float, float, float, float]] = None,
                 depth_scale: float = 1000.0,
                 min_proposal_area: int = 1000,
                 max_proposals: int = 20,
                 min_points: int = 100):
        """
        Initialize target processing pipeline.

        Args:
            sam2_client: SAM2Client for segmentation
            gedi_client: GeDiClient for geometric features
            dinov2_client: Optional DINOv2Client for visual features
            use_visual: Whether to extract and fuse visual features
            camera_intrinsics: Optional (fx, fy, cx, cy). Auto-estimated if None
            depth_scale: Depth units to meters conversion
            min_proposal_area: Minimum mask area for proposals
            max_proposals: Maximum number of proposals to process
            min_points: Minimum points per proposal for feature extraction
        """
        self.segmentation = SegmentationProcessor(
            sam2_client,
            min_area=min_proposal_area,
            max_masks=max_proposals
        )

        self.depth_lifter = DepthLifter(depth_scale=depth_scale)
        if camera_intrinsics is not None:
            fx, fy, cx, cy = camera_intrinsics
            self.depth_lifter.set_intrinsics(fx, fy, cx, cy)

        self.geometric = TargetGeometricProcessor(gedi_client, min_points=min_points)

        self.use_visual = use_visual and dinov2_client is not None
        if self.use_visual:
            self.visual = VisualProcessor(dinov2_client)
            self.fusion = TargetFeatureFusion(k_neighbors=5, distance_threshold=50.0)
        else:
            self.visual = None
            self.fusion = None

        logger.info(f"TargetPipeline initialized")
        logger.info(f"  Segmentation: min_area={min_proposal_area}, max={max_proposals}")
        logger.info(f"  Depth: scale={depth_scale}")
        logger.info(f"  Geometric: min_points={min_points}")
        logger.info(f"  Visual features: {self.use_visual}")

    def process_scene(self,
                     rgb_path: Union[str, Path],
                     depth_path: Union[str, Path],
                     filter_overlap: bool = True,
                     subsample_points: Optional[int] = None) -> Dict:
        """
        Process complete RGB-D scene.

        Args:
            rgb_path: Path to RGB image
            depth_path: Path to depth image
            filter_overlap: Whether to filter overlapping proposals
            subsample_points: Optional point subsampling factor

        Returns:
            Dictionary with:
                - proposals: List of proposal dicts with features
                - num_proposals: int
                - scene_metadata: dict with scene info
        """
        rgb_path = Path(rgb_path)
        depth_path = Path(depth_path)

        logger.info(f"Processing scene: {rgb_path.name}")

        # Step 1: Segmentation
        logger.info("[1/4] Segmenting scene...")
        proposals = self.segmentation.segment(rgb_path)

        if filter_overlap:
            proposals = self.segmentation.filter_by_overlap(proposals, iou_threshold=0.5)

        logger.info(f"  → {len(proposals)} proposals")

        # Step 2: Load depth and RGB
        logger.info("[2/4] Loading depth and RGB...")
        depth_image = DepthLifter.load_depth_image(depth_path)

        from PIL import Image
        rgb_image = np.array(Image.open(rgb_path))

        # Auto-estimate intrinsics if not set
        if self.depth_lifter.fx == 525.0:  # Default value, not explicitly set
            height, width = depth_image.shape
            fx, fy, cx, cy = DepthLifter.estimate_intrinsics_from_fov(width, height, fov_deg=60.0)
            self.depth_lifter.set_intrinsics(fx, fy, cx, cy)
            logger.info(f"  Auto-estimated intrinsics: fx={fx:.1f}, fy={fy:.1f}")

        # Step 3: Depth lifting
        logger.info("[3/4] Lifting proposals to 3D...")
        # Default subsampling if not specified (take every 3rd point)
        if subsample_points is None:
            subsample_points = 3

        point_clouds = self.depth_lifter.lift_multiple_proposals(
            depth_image, proposals, rgb_image, subsample=subsample_points
        )

        # Preserve original mask, bbox, and camera intrinsics for visual feature extraction
        for i, (pc, prop) in enumerate(zip(point_clouds, proposals)):
            pc['mask'] = prop['mask']
            pc['bbox'] = prop.get('bbox', [0, 0, 0, 0])
            pc['fx'] = self.depth_lifter.fx
            pc['fy'] = self.depth_lifter.fy
            pc['cx'] = self.depth_lifter.cx
            pc['cy'] = self.depth_lifter.cy

        total_points = sum(pc['num_points'] for pc in point_clouds)
        logger.info(f"  → {total_points} total points (subsampled 1/{subsample_points})")

        # Step 4: Feature extraction
        if self.use_visual:
            logger.info("[4/6] Extracting geometric features...")
            proposal_features = self.geometric.process_proposals(point_clouds)
            logger.info(f"  → {len(proposal_features)} proposals with geometric features")

            logger.info("[5/6] Extracting visual features...")
            proposal_features = self.visual.process_proposals(rgb_image, proposal_features)
            logger.info(f"  → {len(proposal_features)} proposals with visual features")

            logger.info("[6/6] Fusing geometric + visual features...")
            proposal_features = self.fusion.fuse_proposals(proposal_features)
            logger.info(f"  → {len(proposal_features)} proposals with fused features")

            feature_type = "fused (geometric + visual)"
        else:
            logger.info("[4/4] Extracting geometric features...")
            proposal_features = self.geometric.process_proposals(point_clouds)
            logger.info(f"  → {len(proposal_features)} proposals with geometric features")
            feature_type = "geometric only"

        # Compile results
        result = {
            'proposals': proposal_features,
            'num_proposals': len(proposal_features),
            'segmentation_masks': proposals,  # Raw SAM2 masks for visualization
            'feature_type': feature_type,
            'scene_metadata': {
                'rgb_path': str(rgb_path),
                'depth_path': str(depth_path),
                'image_size': rgb_image.shape[:2],
                'total_initial_proposals': len(proposals),
                'total_lifted_proposals': len(point_clouds),
                'total_feature_proposals': len(proposal_features),
                'use_visual': self.use_visual
            }
        }

        logger.info(f"Scene processing complete: {len(proposal_features)} proposals ({feature_type})")

        return result

    def process_with_known_object(self,
                                  rgb_path: Union[str, Path],
                                  depth_path: Union[str, Path],
                                  object_bbox: List[int],
                                  expand_bbox: float = 0.1) -> Optional[Dict]:
        """
        Process scene with known object bounding box.

        Useful for debugging or when approximate object location is known.

        Args:
            rgb_path: Path to RGB image
            depth_path: Path to depth image
            object_bbox: [x, y, w, h] bounding box
            expand_bbox: Fraction to expand bbox (e.g., 0.1 = 10%)

        Returns:
            Single proposal dict with features, or None if failed
        """
        rgb_path = Path(rgb_path)
        depth_path = Path(depth_path)

        logger.info(f"Processing scene with known bbox: {object_bbox}")

        # Expand bbox
        x, y, w, h = object_bbox
        expand_w = int(w * expand_bbox)
        expand_h = int(h * expand_bbox)

        x = max(0, x - expand_w // 2)
        y = max(0, y - expand_h // 2)
        w = w + expand_w
        h = h + expand_h

        # Create mask from bbox
        from PIL import Image
        rgb_image = np.array(Image.open(rgb_path))
        height, width = rgb_image.shape[:2]

        mask = np.zeros((height, width), dtype=bool)
        mask[y:y+h, x:x+w] = True

        # Create proposal from bbox
        proposal = self.segmentation.segment_with_bbox(rgb_path, [x, y, w, h])

        # Load depth
        depth_image = DepthLifter.load_depth_image(depth_path)

        # Lift to 3D
        pc = self.depth_lifter.lift_proposal(depth_image, proposal['mask'], rgb_image)

        # Extract features
        result = self.geometric.process_proposal(pc)

        if result is not None:
            logger.info(f"Processed known object: {result['num_points']} points")

        return result
