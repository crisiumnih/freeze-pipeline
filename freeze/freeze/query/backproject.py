"""
Feature back-projection for query objects.
Maps 2D visual features from rendered views to 3D point cloud coordinates.
"""
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class FeatureBackProjector:
    """Back-project 2D visual features to 3D point cloud."""

    def __init__(self, image_size=(640, 480), fov=60.0):
        """
        Initialize back-projector.

        Args:
            image_size: (width, height) of rendered images
            fov: Field of view in degrees (vertical FOV)
        """
        self.image_size = image_size
        self.width, self.height = image_size
        self.fov = fov

        # Compute camera intrinsics
        self.focal_length = self.height / (2.0 * np.tan(np.radians(fov / 2.0)))
        self.cx = self.width / 2.0
        self.cy = self.height / 2.0

        # Camera intrinsic matrix
        self.K = np.array([
            [self.focal_length, 0, self.cx],
            [0, self.focal_length, self.cy],
            [0, 0, 1]
        ])

        self.K_inv = np.linalg.inv(self.K)

        logger.info(f"FeatureBackProjector initialized: image_size={image_size}, fov={fov}°")
        logger.info(f"  Focal length: {self.focal_length:.2f}")
        logger.info(f"  Principal point: ({self.cx:.1f}, {self.cy:.1f})")

    def get_camera_pose(self, azimuth: float, elevation: float, distance: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute camera rotation and translation from spherical coordinates.

        Args:
            azimuth: Azimuth angle in degrees (rotation around Y-axis)
            elevation: Elevation angle in degrees (rotation up/down)
            distance: Distance from origin

        Returns:
            R: 3x3 rotation matrix (world to camera)
            t: 3x1 translation vector (world to camera)
        """
        # Convert to radians
        az_rad = np.radians(azimuth)
        el_rad = np.radians(elevation)

        # Camera position in world coordinates (spherical to cartesian)
        # Azimuth rotates around Y (vertical), elevation tilts up/down
        x = distance * np.cos(el_rad) * np.sin(az_rad)
        y = distance * np.sin(el_rad)
        z = distance * np.cos(el_rad) * np.cos(az_rad)

        camera_pos = np.array([x, y, z])

        # Look-at transformation
        # Camera looks at origin, up vector is Y-axis
        forward = -camera_pos / np.linalg.norm(camera_pos)  # Look at origin
        right = np.cross(np.array([0, 1, 0]), forward)
        right = right / np.linalg.norm(right)
        up = np.cross(forward, right)

        # Rotation matrix (world to camera)
        R = np.stack([right, up, forward], axis=0)

        # Translation vector (world to camera)
        t = -R @ camera_pos

        return R, t

    def unproject_pixels(self,
                        coords: np.ndarray,
                        depths: np.ndarray,
                        R: np.ndarray,
                        t: np.ndarray) -> np.ndarray:
        """
        Unproject 2D pixel coordinates to 3D world points.

        Args:
            coords: (N, 2) pixel coordinates [y, x] (image space)
            depths: (N,) depth values at each pixel
            R: 3x3 camera rotation (world to camera)
            t: 3x1 camera translation (world to camera)

        Returns:
            points_3d: (N, 3) 3D points in world coordinates
        """
        # Convert [y, x] to [x, y] for camera model
        pixels = coords[:, [1, 0]].astype(float)

        # Homogeneous pixel coordinates
        pixels_h = np.column_stack([pixels, np.ones(len(pixels))])

        # Back-project to camera space
        points_cam = (self.K_inv @ pixels_h.T).T * depths[:, None]

        # Transform to world space
        # x_world = R^-1 @ (x_cam - t)
        R_inv = R.T  # Rotation is orthogonal
        points_world = (R_inv @ (points_cam - t).T).T

        return points_world

    def backproject_view(self,
                        features: np.ndarray,
                        coords: np.ndarray,
                        depth_map: np.ndarray,
                        azimuth: float,
                        elevation: float,
                        distance: float) -> Dict:
        """
        Back-project features from a single view.

        Args:
            features: (N, D) visual features (e.g., 1024-dim DINOv2)
            coords: (N, 2) pixel coordinates [y, x]
            depth_map: (H, W) depth map from rendering
            azimuth: Camera azimuth angle (degrees)
            elevation: Camera elevation angle (degrees)
            distance: Camera distance

        Returns:
            Dictionary with:
                - points_3d: (N, 3) 3D points in world coordinates
                - features: (N, D) visual features
                - view_metadata: dict with camera parameters
        """
        # Get camera pose
        R, t = self.get_camera_pose(azimuth, elevation, distance)

        # Clip coordinates to valid depth map bounds
        height, width = depth_map.shape
        coords_clipped = coords.copy()
        coords_clipped[:, 0] = np.clip(coords_clipped[:, 0], 0, height - 1)
        coords_clipped[:, 1] = np.clip(coords_clipped[:, 1], 0, width - 1)

        # Sample depth at feature locations
        depths = depth_map[coords_clipped[:, 0], coords_clipped[:, 1]]

        # Filter out invalid depths
        valid_mask = (depths > 0) & np.isfinite(depths)

        if valid_mask.sum() == 0:
            logger.warning(f"No valid depths for view (az={azimuth}, el={elevation})")
            return {
                'points_3d': np.zeros((0, 3)),
                'features': np.zeros((0, features.shape[1])),
                'view_metadata': {
                    'azimuth': azimuth,
                    'elevation': elevation,
                    'distance': distance,
                    'num_valid': 0
                }
            }

        # Filter by valid depths
        coords_valid = coords_clipped[valid_mask]
        features_valid = features[valid_mask]
        depths_valid = depths[valid_mask]

        # Unproject to 3D
        points_3d = self.unproject_pixels(coords_valid, depths_valid, R, t)

        logger.debug(f"Back-projected {len(points_3d)} points from view (az={azimuth}, el={elevation})")

        return {
            'points_3d': points_3d,
            'features': features_valid,
            'view_metadata': {
                'azimuth': azimuth,
                'elevation': elevation,
                'distance': distance,
                'num_valid': len(points_3d),
                'num_total': len(coords)
            }
        }

    def backproject_multiview(self,
                             view_results: List[Dict],
                             depth_maps: List[np.ndarray]) -> Dict:
        """
        Back-project features from multiple views and merge.

        Args:
            view_results: List of dicts from VisualProcessor.process(), each with:
                - features: (N, D)
                - coords: (N, 2)
                - image_size: (H, W)
            depth_maps: List of (H, W) depth maps, one per view

        Returns:
            Dictionary with:
                - points_3d: (M, 3) merged 3D points from all views
                - features: (M, D) corresponding visual features
                - view_indices: (M,) which view each point came from
                - metadata: dict with aggregated statistics
        """
        all_points = []
        all_features = []
        all_view_indices = []

        for view_idx, (result, depth_map) in enumerate(zip(view_results, depth_maps)):
            # Extract view metadata
            metadata = result.get('metadata', {})
            azimuth = metadata.get('azimuth', 0.0)
            elevation = metadata.get('elevation', 0.0)
            distance = metadata.get('distance', 2.0)

            # Back-project this view
            bp_result = self.backproject_view(
                features=result['features'],
                coords=result['coords'],
                depth_map=depth_map,
                azimuth=azimuth,
                elevation=elevation,
                distance=distance
            )

            # Accumulate
            if len(bp_result['points_3d']) > 0:
                all_points.append(bp_result['points_3d'])
                all_features.append(bp_result['features'])
                all_view_indices.append(np.full(len(bp_result['points_3d']), view_idx))

        if len(all_points) == 0:
            logger.warning("No valid points across all views")
            return {
                'points_3d': np.zeros((0, 3)),
                'features': np.zeros((0, view_results[0]['features'].shape[1])),
                'view_indices': np.zeros(0, dtype=int),
                'metadata': {'num_views': len(view_results), 'num_points': 0}
            }

        # Concatenate all views
        points_3d = np.vstack(all_points)
        features = np.vstack(all_features)
        view_indices = np.concatenate(all_view_indices)

        logger.info(f"Back-projected {len(points_3d)} total points from {len(view_results)} views")

        return {
            'points_3d': points_3d,
            'features': features,
            'view_indices': view_indices,
            'metadata': {
                'num_views': len(view_results),
                'num_points': len(points_3d),
                'points_per_view': [len(p) for p in all_points]
            }
        }
