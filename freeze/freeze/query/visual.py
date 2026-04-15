"""
Visual feature extraction using DINOv2.
"""
import numpy as np
from PIL import Image
from typing import Dict, Optional, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class VisualProcessor:
    """Extract visual features from RGB images using DINOv2."""

    def __init__(self, dinov2_client):
        """
        Initialize visual processor.

        Args:
            dinov2_client: DINOv2Client instance for feature extraction
        """
        self.dinov2 = dinov2_client
        logger.info("VisualProcessor initialized")

    def process(self,
                image_path: Union[str, Path],
                mask_path: Optional[Union[str, Path]] = None
                ) -> Dict[str, np.ndarray]:
        """
        Extract DINOv2 visual features from an image.

        Args:
            image_path: Path to RGB image
            mask_path: Optional path to binary mask (extracts features only at masked pixels)

        Returns:
            {
                'features': [N, 1024],  # DINOv2 features
                'coords': [N, 2],  # Pixel coordinates (y, x)
                'image_size': (H, W),  # Processed image size (518x518)
                'original_size': (H, W),  # Original image size
                'num_pixels': int  # Number of pixels/features
            }
        """
        image_path = Path(image_path)
        logger.info(f"Processing image: {image_path}")

        if mask_path is not None:
            mask_path = Path(mask_path)
            logger.info(f"Using mask: {mask_path}")

        # Extract features using DINOv2 client
        logger.debug("Extracting DINOv2 features...")
        features, coords, info = self.dinov2.extract_features(
            image_path,
            mask_path=mask_path,
            return_coords=True
        )

        logger.info(f"Extracted features with shape {features.shape}")

        return {
            'features': features,
            'coords': coords,
            'image_size': info.get('image_size', (518, 518)),
            'original_size': info.get('original_size', (0, 0)),
            'num_pixels': len(features)
        }

    def process_with_image(self,
                          image: Union[np.ndarray, Image.Image],
                          mask: Optional[np.ndarray] = None,
                          temp_dir: Optional[Path] = None
                          ) -> Dict[str, np.ndarray]:
        """
        Process image array directly (saves to temp file).

        Args:
            image: RGB image as np.ndarray [H, W, 3] or PIL Image
            mask: Optional binary mask [H, W]
            temp_dir: Directory for temporary files

        Returns:
            Same as process()
        """
        import tempfile

        if temp_dir is None:
            temp_dir = Path(tempfile.gettempdir())
        else:
            temp_dir = Path(temp_dir)
            temp_dir.mkdir(parents=True, exist_ok=True)

        # Convert to PIL if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype(np.uint8))

        # Save to temp file
        temp_image = temp_dir / 'temp_image.png'
        image.save(temp_image)

        temp_mask = None
        if mask is not None:
            temp_mask = temp_dir / 'temp_mask.png'
            mask_img = Image.fromarray((mask * 255).astype(np.uint8))
            mask_img.save(temp_mask)

        # Process
        result = self.process(temp_image, temp_mask)

        # Cleanup
        temp_image.unlink()
        if temp_mask is not None:
            temp_mask.unlink()

        return result
