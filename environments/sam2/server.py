#!/usr/bin/env python3
"""
SAM2 Inference Server
Provides automatic mask generation via file-based API
"""
import sys
import json
import numpy as np
import cv2
from PIL import Image
from pathlib import Path

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


class SAM2Server:
    def __init__(self, checkpoint_path=None):
        """Initialize SAM2 model."""
        if checkpoint_path is None:
            checkpoint_path = Path(__file__).parent / "checkpoints/sam2.1_hiera_large.pt"

        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

        print(f"Loading SAM2 model from {checkpoint_path}...", file=sys.stderr)
        self.sam2 = build_sam2(model_cfg, str(checkpoint_path), device="cuda")
        self.mask_generator = SAM2AutomaticMaskGenerator(
            self.sam2,
            points_per_side=64,
            pred_iou_thresh=0.7,
            stability_score_thresh=0.85,
            min_mask_region_area=500,
        )
        print("SAM2 model loaded successfully!", file=sys.stderr)

    def generate_masks(self, image_path, top_k=None):
        """
        Generate segmentation masks for an image.

        Args:
            image_path: Path to RGB image
            top_k: Optional, keep only top K masks by area

        Returns:
            list of mask dictionaries
        """
        # Load image
        image = cv2.imread(str(image_path))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        print(f"Generating masks for {image_path}...", file=sys.stderr)
        masks = self.mask_generator.generate(image_rgb)
        print(f"Generated {len(masks)} masks", file=sys.stderr)

        # Sort by area
        masks_sorted = sorted(masks, key=lambda x: x['area'], reverse=True)

        # Keep top K if specified
        if top_k is not None:
            masks_sorted = masks_sorted[:top_k]

        return masks_sorted, image_rgb

    def process_file_request(self, request_path):
        """Process a file-based inference request."""
        with open(request_path) as f:
            request = json.load(f)

        image_path = request['image']
        output_dir = Path(request['output_dir'])
        top_k = request.get('top_k', None)

        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate masks
        masks, image_rgb = self.generate_masks(image_path, top_k)

        # Save masks and metadata
        metadata = []
        base_name = Path(image_path).stem

        for i, mask_data in enumerate(masks):
            mask = mask_data['segmentation']
            area = mask_data['area']
            bbox = mask_data['bbox']  # [x, y, w, h]

            # Save binary mask
            mask_filename = f"{base_name}_mask_{i:02d}.png"
            mask_path = output_dir / mask_filename
            mask_img = (mask * 255).astype(np.uint8)
            Image.fromarray(mask_img).save(mask_path)

            # Save crop
            x, y, w, h = [int(v) for v in bbox]
            crop = image_rgb[y:y+h, x:x+w]
            crop_filename = f"{base_name}_crop_{i:02d}.png"
            crop_path = output_dir / crop_filename
            Image.fromarray(crop).save(crop_path)

            metadata.append({
                'mask_id': i,
                'mask_path': str(mask_path),
                'crop_path': str(crop_path),
                'area': int(area),
                'bbox': {'x': x, 'y': y, 'width': w, 'height': h}
            })

        # Save metadata
        metadata_path = output_dir / f"{base_name}_metadata.npy"
        np.save(metadata_path, {
            'image_path': str(image_path),
            'num_masks': len(masks),
            'masks': metadata
        })

        return {
            'status': 'success',
            'num_masks': len(masks),
            'metadata_path': str(metadata_path),
            'output_dir': str(output_dir)
        }


def main():
    if len(sys.argv) < 2:
        print("Usage: python server.py <request_json_path>", file=sys.stderr)
        sys.exit(1)

    request_path = sys.argv[1]

    # Initialize server
    server = SAM2Server()

    # Process request
    result = server.process_file_request(request_path)

    # Print result as JSON to stdout
    print(json.dumps(result))


if __name__ == '__main__':
    main()
