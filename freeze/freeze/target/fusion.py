"""
Feature fusion for target proposals.
Combines geometric (GeDi) and visual (DINOv2) features.
"""
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class TargetFeatureFusion:
    """Fuse geometric and visual features for target proposals."""

    def __init__(self, k_neighbors: int = 5, distance_threshold: float = 50.0):
        """
        Initialize feature fusion.

        Args:
            k_neighbors: Number of neighbors for k-NN aggregation
            distance_threshold: Max pixel distance for valid neighbors
        """
        self.k_neighbors = k_neighbors
        self.distance_threshold = distance_threshold
        logger.info(f"TargetFeatureFusion initialized (k={k_neighbors}, "
                   f"dist_threshold={distance_threshold}px)")

    def fuse_proposal(self,
                     proposal: Dict,
                     camera_matrix: Optional[np.ndarray] = None) -> Dict:
        """
        Fuse geometric and visual features for a single proposal.

        The fusion process:
        1. We have 3D points with geometric features (from GeDi)
        2. We have 2D pixel coordinates with visual features (from DINOv2)
        3. Project 3D points to 2D using camera matrix
        4. For each 3D point, find k nearest 2D visual features
        5. Aggregate visual features and concatenate with geometric

        Args:
            proposal: Dict with:
                - points: (N, 3) 3D points
                - features: (N, 32) geometric features
                - visual_features: Dict with 'features' (M, 1024) and 'coords' (M, 2)
            camera_matrix: (3, 3) camera intrinsics. If None, uses stored value

        Returns:
            Dict with:
                - points: (N, 3) 3D points (same as input)
                - features: (N, 1056) fused features (32 geo + 1024 vis)
                - geometric_features: (N, 32) original geometric
                - visual_features_aggregated: (N, 1024) aggregated visual
        """
        # Get data
        points_3d = proposal['points']  # (N, 3)
        geo_features = proposal['features']  # (N, 32)
        visual_data = proposal.get('visual_features', {})

        vis_features = visual_data.get('features')  # (M, 1024)
        vis_coords = visual_data.get('coords')  # (M, 2)

        # Check if we have visual features
        if vis_features is None or len(vis_features) == 0:
            logger.warning("No visual features available, using geometric only")
            # Return geometric features with zero padding for visual
            zero_vis = np.zeros((len(points_3d), 1024), dtype=np.float32)
            fused = np.concatenate([geo_features, zero_vis], axis=1)
            return {
                'points': points_3d,
                'features': fused,
                'geometric_features': geo_features,
                'visual_features_aggregated': zero_vis
            }

        # Get camera matrix
        if camera_matrix is None:
            camera_matrix = proposal.get('camera_matrix')
            if camera_matrix is None:
                # Use stored intrinsics from proposal
                fx = proposal.get('fx', 600.0)
                fy = proposal.get('fy', 600.0)
                cx = proposal.get('cx', 320.0)
                cy = proposal.get('cy', 240.0)
                camera_matrix = np.array([
                    [fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1]
                ], dtype=np.float32)

        # Project 3D points to 2D
        points_2d = self._project_points(points_3d, camera_matrix)

        # For each 3D point, find k nearest visual features in 2D
        vis_aggregated = self._aggregate_visual_features(
            points_2d,
            vis_coords,
            vis_features
        )

        # Concatenate geometric and visual features
        fused = np.concatenate([geo_features, vis_aggregated], axis=1)

        logger.info(f"Fused features: {len(points_3d)} points, "
                   f"{fused.shape[1]}-dim (32 geo + 1024 vis)")

        return {
            'points': points_3d,
            'features': fused,
            'geometric_features': geo_features,
            'visual_features_aggregated': vis_aggregated
        }

    def _project_points(self,
                       points_3d: np.ndarray,
                       camera_matrix: np.ndarray) -> np.ndarray:
        """
        Project 3D points to 2D image coordinates.

        Args:
            points_3d: (N, 3) 3D points
            camera_matrix: (3, 3) camera intrinsics

        Returns:
            points_2d: (N, 2) pixel coordinates
        """
        # Project: p_2d = K @ p_3d
        # p_2d_homogeneous = camera_matrix @ points_3d.T  # (3, N)
        # But we need to handle perspective divide by Z

        fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
        cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]

        x = points_3d[:, 0]
        y = points_3d[:, 1]
        z = points_3d[:, 2]

        # Avoid division by zero
        z = np.where(np.abs(z) < 1e-6, 1e-6, z)

        u = fx * x / z + cx
        v = fy * y / z + cy

        points_2d = np.stack([u, v], axis=1)

        return points_2d

    def _aggregate_visual_features(self,
                                   points_2d: np.ndarray,
                                   vis_coords: np.ndarray,
                                   vis_features: np.ndarray) -> np.ndarray:
        """
        Aggregate visual features using k-NN in 2D pixel space.

        Args:
            points_2d: (N, 2) 2D coordinates of 3D points
            vis_coords: (M, 2) 2D coordinates of visual features
            vis_features: (M, 1024) visual features

        Returns:
            aggregated: (N, 1024) aggregated visual features
        """
        N = len(points_2d)
        M = len(vis_coords)

        logger.debug(f"Aggregating {M} visual features to {N} 3D points")

        if M == 0:
            return np.zeros((N, 1024), dtype=np.float32)

        # For each point in points_2d, find k nearest in vis_coords
        aggregated = np.zeros((N, 1024), dtype=np.float32)

        # Compute pairwise distances (N, M)
        # Use chunking to avoid memory issues for large N
        chunk_size = 1000

        for start_idx in range(0, N, chunk_size):
            end_idx = min(start_idx + chunk_size, N)
            chunk = points_2d[start_idx:end_idx]  # (chunk_size, 2)

            # Pairwise distances: (chunk_size, M)
            diffs = chunk[:, np.newaxis, :] - vis_coords[np.newaxis, :, :]  # (chunk, M, 2)
            distances = np.linalg.norm(diffs, axis=2)  # (chunk, M)

            # For each point in chunk, find k nearest
            for i in range(len(chunk)):
                dists = distances[i]  # (M,)

                # Find k nearest (or all if M < k)
                k = min(self.k_neighbors, M)
                nearest_indices = np.argpartition(dists, k-1)[:k]

                # Filter by distance threshold
                valid_mask = dists[nearest_indices] < self.distance_threshold
                valid_indices = nearest_indices[valid_mask]

                if len(valid_indices) > 0:
                    # Average the features
                    aggregated[start_idx + i] = vis_features[valid_indices].mean(axis=0)
                else:
                    # No valid neighbors, use zeros
                    aggregated[start_idx + i] = 0.0

        return aggregated

    def fuse_proposals(self, proposals: list) -> list:
        """
        Fuse features for multiple proposals.

        Args:
            proposals: List of proposal dicts

        Returns:
            List of proposals with fused features
        """
        logger.info(f"Fusing features for {len(proposals)} proposals")

        for i, proposal in enumerate(proposals):
            try:
                fused = self.fuse_proposal(proposal)
                proposal.update(fused)
            except Exception as e:
                logger.error(f"Fusion failed for proposal {i}: {e}")
                # Keep geometric features only
                proposal['features'] = proposal.get('features', np.zeros((0, 32)))

        return proposals
