#!/usr/bin/env python3
"""
DINOv2 Inference Server
Provides visual feature extraction via file-based API
"""
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from pathlib import Path


class DINOv2Server:
    def __init__(self, model_name='dinov2_vitl14'):
        """Initialize DINOv2 model."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"Loading DINOv2 model ({model_name})...", file=sys.stderr)
        self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        self.model = self.model.to(self.device)
        self.model.eval()
        print("DINOv2 model loaded successfully!", file=sys.stderr)

        # Configuration
        self.patch_size = 14
        self.img_size = 518  # 518 = 37 * 14
        self.feature_dim = 1024  # for ViT-L

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(self.img_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        # Mask preprocessing
        self.mask_transform = transforms.Compose([
            transforms.Resize(self.img_size, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.CenterCrop(self.img_size),
            transforms.ToTensor(),
        ])

    def extract_features(self, image_path, mask_path=None):
        """
        Extract DINOv2 features from an image.

        Args:
            image_path: Path to RGB image
            mask_path: Optional path to binary mask

        Returns:
            dict with 'features', 'coords', and metadata
        """
        # Load image
        img = Image.open(image_path).convert('RGB')
        orig_h, orig_w = img.size[1], img.size[0]

        # Transform
        x = self.transform(img).unsqueeze(0).to(self.device)

        # Extract patch tokens
        with torch.no_grad():
            output = self.model.get_intermediate_layers(x, n=1)[0]
            patch_tokens = output[:, 1:, :]  # Remove CLS token

        batch_size, num_patches, feature_dim = patch_tokens.shape

        print(f"DEBUG: patch_tokens shape = {patch_tokens.shape}", file=sys.stderr)
        print(f"DEBUG: batch_size={batch_size}, num_patches={num_patches}, feature_dim={feature_dim}", file=sys.stderr)

        # Compute actual grid size from model
        # DINOv2 returns patches, need to figure out grid dimensions
        # For 518x518 image with patch_size=14: should be 37x37 = 1369
        # But if we get 1368, it might be 36x38 or needs padding

        grid_h = grid_w = int(np.sqrt(num_patches))

        # If not perfect square, try rectangular
        if grid_h * grid_w != num_patches:
            # Try to find factors
            for h in range(grid_h, grid_h + 3):
                if num_patches % h == 0:
                    w = num_patches // h
                    if abs(h - w) <= 2:  # Close to square
                        grid_h, grid_w = h, w
                        break

        print(f"DEBUG: Attempting reshape to [{batch_size}, {grid_h}, {grid_w}, {feature_dim}]", file=sys.stderr)
        print(f"DEBUG: Target elements: {batch_size * grid_h * grid_w * feature_dim}", file=sys.stderr)
        print(f"DEBUG: Actual elements: {patch_tokens.numel()}", file=sys.stderr)

        # Verify the reshape will work
        if grid_h * grid_w != num_patches:
            # Pad to nearest perfect square
            grid_size = int(np.ceil(np.sqrt(num_patches)))
            print(f"Warning: Padding {num_patches} patches to {grid_size}x{grid_size} = {grid_size*grid_size}", file=sys.stderr)
            # Pad the patches
            pad_size = grid_size * grid_size - num_patches
            padding = torch.zeros(batch_size, pad_size, feature_dim, device=patch_tokens.device)
            patch_tokens = torch.cat([patch_tokens, padding], dim=1)
            grid_h = grid_w = grid_size

        features_spatial = patch_tokens.reshape(batch_size, grid_h, grid_w, feature_dim)
        features_spatial = features_spatial.permute(0, 3, 1, 2)  # [1, 1024, 37, 37]

        # Upsample to full resolution
        features_upsampled = F.interpolate(
            features_spatial,
            size=(self.img_size, self.img_size),
            mode='bilinear',
            align_corners=False
        )

        # Apply mask if provided
        if mask_path:
            mask = Image.open(mask_path).convert('L')
            mask_tensor = self.mask_transform(mask).to(self.device)
            mask_binary = (mask_tensor > 0.5).float()

            # Get masked coordinates
            mask_coords = torch.nonzero(mask_binary[0], as_tuple=False)

            if len(mask_coords) == 0:
                return {
                    'features': np.array([]),
                    'coords': np.array([]),
                    'feature_dim': feature_dim,
                    'num_pixels': 0
                }

            # Extract features at masked locations
            y_coords = mask_coords[:, 0]
            x_coords = mask_coords[:, 1]
            features_at_mask = features_upsampled[0, :, y_coords, x_coords]
            features_at_mask = features_at_mask.transpose(0, 1)  # [N, 1024]

            features_array = features_at_mask.cpu().numpy()
            coords_array = mask_coords.cpu().numpy()
        else:
            # Return all features
            features_array = features_upsampled[0].permute(1, 2, 0).cpu().numpy()  # [H, W, 1024]
            features_array = features_array.reshape(-1, feature_dim)  # [H*W, 1024]

            # Create coordinate grid
            h, w = self.img_size, self.img_size
            y_grid, x_grid = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
            coords_array = torch.stack([y_grid, x_grid], dim=-1).reshape(-1, 2).numpy()

        return {
            'features': features_array,
            'coords': coords_array,
            'feature_dim': int(feature_dim),
            'image_size': (self.img_size, self.img_size),
            'original_size': (orig_h, orig_w),
            'num_pixels': len(features_array)
        }

    def process_file_request(self, request_path):
        """Process a file-based inference request."""
        with open(request_path) as f:
            request = json.load(f)

        image_path = request['image']
        mask_path = request.get('mask', None)
        output_path = request['output']

        # Extract features
        result = self.extract_features(image_path, mask_path)

        # Save output
        np.save(output_path, result)

        # Return result info
        return {
            'status': 'success',
            'num_pixels': result['num_pixels'],
            'feature_dim': result['feature_dim'],
            'output_shape': result['features'].shape
        }


def main():
    if len(sys.argv) < 2:
        print("Usage: python server.py <request_json_path>", file=sys.stderr)
        sys.exit(1)

    request_path = sys.argv[1]

    # Initialize server
    server = DINOv2Server()

    # Process request
    result = server.process_file_request(request_path)

    # Print result as JSON to stdout
    print(json.dumps(result))


if __name__ == '__main__':
    main()
