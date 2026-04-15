"""
Feature fusion for query objects.
Merges geometric features (GeDi) with visual features (DINOv2) into unified representation.
"""
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Literal
import logging

logger = logging.getLogger(__name__)


class FeatureFusion:
    """Fuse geometric and visual features for query objects."""

    def __init__(self,
                 fusion_method: Literal['concatenate', 'weighted', 'attention'] = 'concatenate',
                 k_neighbors: int = 5,
                 distance_threshold: float = 0.1):
        """
        Initialize feature fusion.

        Args:
            fusion_method: How to combine features
                - 'concatenate': Simple concatenation [geo; visual]
                - 'weighted': Weighted average based on distance
                - 'attention': Attention-based fusion (simple dot-product)
            k_neighbors: Number of nearest visual features to consider per geometric point
            distance_threshold: Max distance for visual feature association (in normalized units)
        """
        self.fusion_method = fusion_method
        self.k_neighbors = k_neighbors
        self.distance_threshold = distance_threshold

        logger.info(f"FeatureFusion initialized: method={fusion_method}, k={k_neighbors}")

    def _find_nearest_neighbors(self,
                               query_points: np.ndarray,
                               reference_points: np.ndarray,
                               k: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Find k nearest neighbors for each query point.

        Uses chunked processing to avoid memory issues with large point clouds.

        Args:
            query_points: (N, 3) query point positions
            reference_points: (M, 3) reference point positions
            k: Number of neighbors

        Returns:
            indices: (N, k) indices of nearest neighbors
            distances: (N, k) distances to nearest neighbors
        """
        n_query = len(query_points)
        n_ref = len(reference_points)

        indices = np.zeros((n_query, k), dtype=np.int32)
        distances = np.zeros((n_query, k), dtype=np.float32)

        # Process in chunks to avoid memory issues
        chunk_size = min(1000, n_query)

        logger.debug(f"Finding {k}-NN for {n_query} points in {n_ref} references")

        for start_idx in range(0, n_query, chunk_size):
            end_idx = min(start_idx + chunk_size, n_query)
            chunk = query_points[start_idx:end_idx]

            # Compute pairwise distances for this chunk
            # Broadcasting: (chunk_size, 1, 3) - (1, n_ref, 3) -> (chunk_size, n_ref)
            diffs = chunk[:, np.newaxis, :] - reference_points[np.newaxis, :, :]
            chunk_distances = np.sqrt((diffs ** 2).sum(axis=2))

            # Find k smallest distances for each query point
            # argsort is faster than argpartition for small k
            if k < 20:
                sorted_indices = np.argsort(chunk_distances, axis=1)
                chunk_indices = sorted_indices[:, :k]
            else:
                # Use argpartition for larger k
                chunk_indices = np.argpartition(chunk_distances, k, axis=1)[:, :k]
                # Sort the k neighbors
                for i in range(len(chunk)):
                    order = np.argsort(chunk_distances[i, chunk_indices[i]])
                    chunk_indices[i] = chunk_indices[i][order]

            # Extract distances for the k neighbors
            chunk_k_distances = np.zeros((len(chunk), k))
            for i in range(len(chunk)):
                chunk_k_distances[i] = chunk_distances[i, chunk_indices[i]]

            indices[start_idx:end_idx] = chunk_indices
            distances[start_idx:end_idx] = chunk_k_distances

        return indices, distances

    def _aggregate_visual_features(self,
                                   geometric_points: np.ndarray,
                                   visual_points: np.ndarray,
                                   visual_features: np.ndarray) -> np.ndarray:
        """
        Aggregate visual features to geometric points using k-NN.

        Args:
            geometric_points: (N, 3) geometric point positions
            visual_points: (M, 3) visual point positions (back-projected)
            visual_features: (M, D) visual features at each visual point

        Returns:
            aggregated_features: (N, D) aggregated visual features for each geometric point
        """
        # Find nearest visual features for each geometric point
        indices, distances = self._find_nearest_neighbors(
            geometric_points, visual_points, self.k_neighbors
        )

        # Filter by distance threshold
        # Normalize distances by point cloud extent
        extent = geometric_points.max(axis=0) - geometric_points.min(axis=0)
        scale = np.linalg.norm(extent)
        normalized_distances = distances / (scale + 1e-8)

        # Mask out far neighbors
        valid_mask = normalized_distances < self.distance_threshold

        # Compute weights (inverse distance)
        weights = np.zeros_like(distances)
        weights[valid_mask] = 1.0 / (distances[valid_mask] + 1e-6)

        # Normalize weights per point
        weight_sum = weights.sum(axis=1, keepdims=True)
        weight_sum[weight_sum == 0] = 1.0  # Avoid division by zero
        weights = weights / weight_sum

        # Aggregate features
        aggregated = np.zeros((len(geometric_points), visual_features.shape[1]))

        for i in range(len(geometric_points)):
            # Weighted average of k nearest visual features
            neighbor_features = visual_features[indices[i]]
            aggregated[i] = (neighbor_features * weights[i, :, None]).sum(axis=0)

        num_valid = (weight_sum[:, 0] > 1e-6).sum()
        logger.debug(f"Aggregated visual features: {num_valid}/{len(geometric_points)} points have neighbors")

        return aggregated

    def fuse(self,
             geometric_result: Dict,
             visual_result: Dict) -> Dict:
        """
        Fuse geometric and visual features.

        Args:
            geometric_result: Output from GeometricProcessor.process()
                - points: (N, 3) point positions
                - features: (N, D_geo) geometric features
            visual_result: Output from FeatureBackProjector.backproject_multiview()
                - points_3d: (M, 3) back-projected visual points
                - features: (M, D_vis) visual features

        Returns:
            Dictionary with:
                - points: (N, 3) point positions (from geometric)
                - features: (N, D_fused) fused features
                - geometric_features: (N, D_geo) original geometric features
                - visual_features: (N, D_vis) aggregated visual features
                - fusion_metadata: dict with fusion statistics
        """
        geometric_points = geometric_result['points']
        geometric_features = geometric_result['features']

        visual_points = visual_result['points_3d']
        visual_features = visual_result['features']

        logger.info(f"Fusing features: {len(geometric_points)} geometric points, "
                   f"{len(visual_points)} visual points")

        # Aggregate visual features to geometric points
        aggregated_visual = self._aggregate_visual_features(
            geometric_points, visual_points, visual_features
        )

        # Fuse features based on method
        if self.fusion_method == 'concatenate':
            # Simple concatenation
            fused_features = np.concatenate([geometric_features, aggregated_visual], axis=1)

        elif self.fusion_method == 'weighted':
            # Weighted average (requires same dimensions)
            # Pad shorter features with zeros if needed
            geo_dim = geometric_features.shape[1]
            vis_dim = aggregated_visual.shape[1]

            if geo_dim == vis_dim:
                # Equal weight average
                fused_features = 0.5 * geometric_features + 0.5 * aggregated_visual
            else:
                # Concatenate if dimensions don't match
                logger.warning(f"Dimension mismatch ({geo_dim} vs {vis_dim}), using concatenation")
                fused_features = np.concatenate([geometric_features, aggregated_visual], axis=1)

        elif self.fusion_method == 'attention':
            # Simple attention: use geometric features as query, visual as key/value
            # Attention weights = softmax(geo · vis^T)
            attention = geometric_features @ aggregated_visual.T
            attention_weights = np.exp(attention) / np.exp(attention).sum(axis=1, keepdims=True)

            # Weighted visual features
            attended_visual = attention_weights @ aggregated_visual

            # Concatenate with geometric
            fused_features = np.concatenate([geometric_features, attended_visual], axis=1)

        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")

        logger.info(f"Fused features: {fused_features.shape} "
                   f"(geo: {geometric_features.shape[1]}, vis: {aggregated_visual.shape[1]})")

        return {
            'points': geometric_points,
            'features': fused_features,
            'geometric_features': geometric_features,
            'visual_features': aggregated_visual,
            'fusion_metadata': {
                'num_points': len(geometric_points),
                'geometric_dim': geometric_features.shape[1],
                'visual_dim': aggregated_visual.shape[1],
                'fused_dim': fused_features.shape[1],
                'fusion_method': self.fusion_method,
                'k_neighbors': self.k_neighbors,
                'num_visual_points': len(visual_points)
            }
        }

    def fuse_multiresolution(self,
                            geometric_results: list[Dict],
                            visual_result: Dict) -> list[Dict]:
        """
        Fuse visual features with multiple geometric resolutions.

        Args:
            geometric_results: List of geometric results at different resolutions
            visual_result: Single visual result to fuse with all resolutions

        Returns:
            List of fused results, one per geometric resolution
        """
        fused_results = []

        for i, geo_result in enumerate(geometric_results):
            logger.info(f"Fusing resolution {i+1}/{len(geometric_results)}")
            fused = self.fuse(geo_result, visual_result)
            fused['resolution_index'] = i
            fused_results.append(fused)

        return fused_results
