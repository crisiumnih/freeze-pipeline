"""
Geometric feature extraction using GeDi.
"""
import numpy as np
from typing import Dict, Optional, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class GeometricProcessor:
    """Extract geometric features from 3D point clouds using GeDi."""

    def __init__(self, gedi_client):
        """
        Initialize geometric processor.

        Args:
            gedi_client: GeDiClient instance for feature extraction
        """
        self.gedi = gedi_client
        logger.info("GeometricProcessor initialized")

    def process(self,
                point_cloud: np.ndarray,
                downsample_voxel_size: Optional[float] = None
                ) -> Dict[str, np.ndarray]:
        """
        Extract geometric features from point cloud.

        Args:
            point_cloud: [N, 3] or [N, 6] array (xyz or xyzrgb)
            downsample_voxel_size: Optional voxel size for downsampling (meters)

        Returns:
            {
                'points': [M, 3],  # Processed point cloud
                'features': [M, 32],  # GeDi features
                'num_points': int  # Number of points
            }
        """
        logger.info(f"Processing point cloud with {len(point_cloud)} points")

        # Extract xyz coordinates
        if point_cloud.shape[1] >= 3:
            points = point_cloud[:, :3].astype(np.float32)
        else:
            raise ValueError(f"Invalid point cloud shape: {point_cloud.shape}")

        # Downsample if requested
        if downsample_voxel_size is not None:
            import open3d as o3d
            logger.debug(f"Downsampling with voxel size {downsample_voxel_size}")
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd = pcd.voxel_down_sample(voxel_size=downsample_voxel_size)
            points = np.asarray(pcd.points, dtype=np.float32)
            logger.info(f"Downsampled to {len(points)} points")

        # Extract GeDi features
        logger.debug("Extracting GeDi features...")
        features = self.gedi.compute_features(points)

        logger.info(f"Extracted features with shape {features.shape}")

        return {
            'points': points,
            'features': features,
            'num_points': len(points)
        }

    def process_from_file(self,
                          file_path: Union[str, Path],
                          downsample_voxel_size: Optional[float] = None
                          ) -> Dict[str, np.ndarray]:
        """
        Load and process point cloud from file.

        Args:
            file_path: Path to point cloud file (.ply, .pcd, .xyz, .npy)
            downsample_voxel_size: Optional voxel size for downsampling

        Returns:
            Same as process()
        """
        file_path = Path(file_path)
        logger.info(f"Loading point cloud from {file_path}")

        # Load based on file extension
        if file_path.suffix in ['.ply', '.pcd']:
            import open3d as o3d
            pcd = o3d.io.read_point_cloud(str(file_path))
            point_cloud = np.asarray(pcd.points, dtype=np.float32)
        elif file_path.suffix == '.xyz':
            point_cloud = np.loadtxt(file_path, dtype=np.float32)
        elif file_path.suffix == '.npy':
            point_cloud = np.load(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

        logger.info(f"Loaded {len(point_cloud)} points")

        return self.process(point_cloud, downsample_voxel_size)
