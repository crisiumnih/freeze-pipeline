"""
Visual feature extraction for target scene proposals.
Uses DINOv2 to extract features from RGB image regions.
"""
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
from PIL import Image
import tempfile
import logging

logger = logging.getLogger(__name__)


class VisualProcessor:
    """Extract DINOv2 visual features from target proposals."""

    def __init__(self, dinov2_client, patch_size: int = 14):
        """
        Initialize visual processor.

        Args:
            dinov2_client: DINOv2Client instance
            patch_size: DINOv2 patch size (14 for ViT-B/14)
        """
        self.dinov2 = dinov2_client
        self.patch_size = patch_size
        logger.info("VisualProcessor initialized")

    def process_proposal(self,
                        rgb_image: np.ndarray,
                        mask: np.ndarray,
                        bbox: Optional[list] = None,
                        output_dir: Optional[Path] = None) -> Dict:
        """
        Extract visual features for a single proposal.

        Args:
            rgb_image: (H, W, 3) RGB image
            mask: (H, W) boolean mask for this proposal
            bbox: Optional [x, y, w, h] bounding box
            output_dir: Optional directory to save debug outputs

        Returns:
            Dict with:
                - features: (N_patches, 1024) DINOv2 features
                - coords: (N_patches, 2) pixel coordinates (x, y) of patch centers
                - patch_size: Patch size in pixels
                - feature_dim: Feature dimension (1024)
        """
        # Get bounding box if not provided
        if bbox is None:
            ys, xs = np.where(mask)
            if len(xs) == 0:
                logger.warning("Empty mask, no features to extract")
                return {
                    'features': np.zeros((0, 1024), dtype=np.float32),
                    'coords': np.zeros((0, 2), dtype=np.float32),
                    'patch_size': self.patch_size,
                    'feature_dim': 1024
                }
            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()
            bbox = [x_min, y_min, x_max - x_min + 1, y_max - y_min + 1]

        x, y, w, h = bbox

        # Crop RGB image to bounding box
        crop = rgb_image[y:y+h, x:x+w]

        # Crop mask to bounding box
        mask_crop = mask[y:y+h, x:x+w]

        # Save crop to temp file for DINOv2 (it expects file paths)
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            crop_path = f.name
            Image.fromarray(crop).save(crop_path)

        try:
            # Extract DINOv2 features for the crop (with coordinates)
            features, coords_local, info = self.dinov2.extract_features(crop_path, return_coords=True)
        finally:
            # Clean up temp file
            Path(crop_path).unlink(missing_ok=True)

        # features: (num_patches, 1024)
        # coords_local: (num_patches, 2) in crop coordinates

        # Convert coordinates to global image coordinates
        coords_global = coords_local + np.array([x, y])

        # Filter features by mask
        # For each patch center, check if it's inside the mask
        valid_mask = np.zeros(len(coords_global), dtype=bool)

        for i, (px, py) in enumerate(coords_global):
            # Convert to integer pixel coordinates
            px_int = int(np.clip(px, 0, rgb_image.shape[1] - 1))
            py_int = int(np.clip(py, 0, rgb_image.shape[0] - 1))

            # Check if this pixel is in the mask
            valid_mask[i] = mask[py_int, px_int]

        # Filter features and coordinates
        features_filtered = features[valid_mask]
        coords_filtered = coords_global[valid_mask]

        logger.info(f"Extracted {len(features_filtered)}/{len(features)} visual features "
                   f"(filtered by mask)")

        result = {
            'features': features_filtered,
            'coords': coords_filtered,
            'patch_size': self.patch_size,
            'feature_dim': 1024
        }

        return result

    def process_proposals(self,
                         rgb_image: np.ndarray,
                         proposals: list,
                         output_dir: Optional[Path] = None) -> list:
        """
        Extract visual features for multiple proposals.

        Args:
            rgb_image: (H, W, 3) RGB image
            proposals: List of proposal dicts with 'mask' and optionally 'bbox'
            output_dir: Optional directory to save debug outputs

        Returns:
            List of proposals with added 'visual_features' dict
        """
        logger.info(f"Processing {len(proposals)} proposals for visual features")

        for i, proposal in enumerate(proposals):
            if output_dir is not None:
                prop_dir = output_dir / f"proposal_{i:02d}"
                prop_dir.mkdir(parents=True, exist_ok=True)
            else:
                prop_dir = None

            visual_result = self.process_proposal(
                rgb_image,
                proposal['mask'],
                bbox=proposal.get('bbox'),
                output_dir=prop_dir
            )

            proposal['visual_features'] = visual_result

        return proposals
