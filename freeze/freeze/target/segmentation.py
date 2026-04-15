"""
Scene segmentation using SAM2.
Generates object proposals from RGB images.
"""
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Union
import logging

logger = logging.getLogger(__name__)


class SegmentationProcessor:
    """Generate object proposals using SAM2."""

    def __init__(self, sam2_client, min_area: int = 1000, max_masks: int = 50):
        """
        Initialize segmentation processor.

        Args:
            sam2_client: SAM2Client instance
            min_area: Minimum mask area in pixels (filter small masks)
            max_masks: Maximum number of masks to return
        """
        self.sam2 = sam2_client
        self.min_area = min_area
        self.max_masks = max_masks
        logger.info(f"SegmentationProcessor initialized: min_area={min_area}, max_masks={max_masks}")

    def segment(self, image_path: Union[str, Path]) -> List[Dict]:
        """
        Segment image into object proposals.

        Args:
            image_path: Path to RGB image

        Returns:
            List of proposals, each containing:
                - mask: (H, W) boolean mask
                - bbox: [x, y, w, h] bounding box
                - area: int, number of pixels
                - score: float, confidence score
                - id: int, proposal index
        """
        import tempfile
        from PIL import Image as PILImage

        image_path = Path(image_path)
        logger.info(f"Segmenting image: {image_path}")

        # Run SAM2 automatic mask generation
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            output_dir = tmp / 'masks'

            metadata = self.sam2.generate_masks(image_path, output_dir, top_k=self.max_masks)

            masks = metadata['masks']

            logger.info(f"SAM2 generated {len(masks)} initial masks")

            # Load masks from files and filter
            proposals = []
            for i, mask_data in enumerate(masks):
                area = mask_data['area']

                # Filter by area
                if area < self.min_area:
                    continue

                # Load mask from file
                mask_path = Path(mask_data['mask_path'])
                mask_img = PILImage.open(mask_path)
                mask = np.array(mask_img) > 0  # Convert to boolean

                # Extract and convert bbox format
                bbox_dict = mask_data['bbox']
                bbox = [bbox_dict['x'], bbox_dict['y'], bbox_dict['width'], bbox_dict['height']]

                proposal = {
                    'mask': mask,
                    'bbox': bbox,
                    'area': area,
                    'score': 1.0,  # SAM2 doesn't provide IoU score in auto mode
                    'id': len(proposals)
                }

                proposals.append(proposal)

        # Sort by area (larger first) and limit
        proposals.sort(key=lambda x: x['area'], reverse=True)
        proposals = proposals[:self.max_masks]

        # Reassign IDs after sorting
        for i, proposal in enumerate(proposals):
            proposal['id'] = i

        logger.info(f"Filtered to {len(proposals)} proposals (min_area={self.min_area})")

        return proposals

    def segment_with_bbox(self,
                         image_path: Union[str, Path],
                         bbox: List[int]) -> Dict:
        """
        Create proposal from known bounding box.

        Args:
            image_path: Path to RGB image
            bbox: [x, y, w, h] bounding box

        Returns:
            Single proposal dict with mask, bbox, etc.
        """
        from PIL import Image as PILImage

        logger.info(f"Creating proposal from bbox: {bbox}")

        # Load image to get size
        img = PILImage.open(image_path)
        height, width = img.size[::-1]

        # Create mask from bbox
        mask = np.zeros((height, width), dtype=bool)
        x, y, w, h = bbox
        mask[y:y+h, x:x+w] = True

        area = mask.sum()

        proposal = {
            'mask': mask,
            'bbox': bbox,
            'area': int(area),
            'score': 1.0,
            'id': 0
        }

        logger.info(f"Generated proposal from bbox: area={area}")

        return proposal

    def filter_by_overlap(self, proposals: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
        """
        Filter overlapping proposals using NMS-like strategy.

        Args:
            proposals: List of proposals (assumes sorted by score/area)
            iou_threshold: IoU threshold for considering overlap

        Returns:
            Filtered list of proposals
        """
        if len(proposals) <= 1:
            return proposals

        kept = []
        masks = [p['mask'] for p in proposals]

        for i, proposal in enumerate(proposals):
            # Check overlap with all kept proposals
            keep = True
            mask_i = masks[i]

            for kept_idx in kept:
                mask_kept = masks[kept_idx]

                # Compute IoU
                intersection = (mask_i & mask_kept).sum()
                union = (mask_i | mask_kept).sum()
                iou = intersection / (union + 1e-8)

                if iou > iou_threshold:
                    keep = False
                    break

            if keep:
                kept.append(i)

        filtered_proposals = [proposals[i] for i in kept]

        logger.info(f"Filtered {len(proposals)} → {len(filtered_proposals)} proposals "
                   f"(IoU threshold={iou_threshold})")

        return filtered_proposals
