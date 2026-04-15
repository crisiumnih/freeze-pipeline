#!/usr/bin/env python3
"""
GeDi Inference Server
Provides geometric feature extraction via file-based API
"""
import sys
import json
import numpy as np
import torch
import open3d as o3d
from pathlib import Path

# Add GeDi to path
GEDI_PATH = Path(__file__).parent.parent.parent / 'GeDi'
sys.path.insert(0, str(GEDI_PATH))

from gedi import GeDi


class GeDiServer:
    def __init__(self, checkpoint_path=None):
        """Initialize GeDi model."""
        if checkpoint_path is None:
            checkpoint_path = GEDI_PATH / 'data/chkpts/3dmatch/chkpt.tar'

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load model
        print(f"Loading GeDi model from {checkpoint_path}...", file=sys.stderr)
        config = {
            'dim': 32,
            'samples_per_batch': 500,
            'samples_per_patch_lrf': 4000,
            'samples_per_patch_out': 512,
            'r_lrf': .5,
            'fchkpt_gedi_net': str(checkpoint_path)
        }
        self.model = GeDi(config=config)
        print("GeDi model loaded successfully!", file=sys.stderr)

    def compute_features(self, point_cloud):
        """
        Compute GeDi features for a point cloud.

        Args:
            point_cloud: np.ndarray of shape [N, 3] or [N, 6] (xyz or xyzrgb)

        Returns:
            features: np.ndarray of shape [N, 32]
        """
        # Handle different input formats
        if point_cloud.shape[1] >= 3:
            points = point_cloud[:, :3]  # Extract xyz
        else:
            raise ValueError(f"Invalid point cloud shape: {point_cloud.shape}")

        # Convert to torch tensor on CPU (radius search requires CPU)
        pts_tensor = torch.tensor(points, dtype=torch.float32)

        # Compute features
        with torch.no_grad():
            features = self.model.compute(pts=pts_tensor, pcd=pts_tensor)

        # Already returns numpy array
        return features

    def process_file_request(self, request_path):
        """Process a file-based inference request."""
        with open(request_path) as f:
            request = json.load(f)

        input_path = request['input']
        output_path = request['output']

        # Load input
        if input_path.endswith('.npy'):
            point_cloud = np.load(input_path)
        elif input_path.endswith('.ply') or input_path.endswith('.pcd'):
            pcd = o3d.io.read_point_cloud(input_path)
            point_cloud = np.asarray(pcd.points)
        elif input_path.endswith('.xyz'):
            point_cloud = np.loadtxt(input_path)
        else:
            raise ValueError(f"Unsupported input format: {input_path}")

        # Compute features
        features = self.compute_features(point_cloud)

        # Save output
        np.save(output_path, features)

        # Return result info
        return {
            'status': 'success',
            'input_shape': point_cloud.shape,
            'output_shape': features.shape,
            'feature_dim': features.shape[1],
            'num_points': features.shape[0]
        }


def main():
    if len(sys.argv) < 2:
        print("Usage: python server.py <request_json_path>", file=sys.stderr)
        sys.exit(1)

    request_path = sys.argv[1]

    # Initialize server
    server = GeDiServer()

    # Process request
    result = server.process_file_request(request_path)

    # Print result as JSON to stdout
    print(json.dumps(result))


if __name__ == '__main__':
    main()
