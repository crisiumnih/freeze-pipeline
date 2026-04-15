"""
Geometric feature extraction for target proposals.
Extracts GeDi features from lifted point clouds.
"""
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class TargetGeometricProcessor:
    """Extract geometric features from target object proposals."""

    def __init__(self, gedi_client, min_points: int = 100):
        """
        Initialize target geometric processor.

        Args:
            gedi_client: GeDiClient instance
            min_points: Minimum points required for feature extraction
        """
        self.gedi = gedi_client
        self.min_points = min_points
        logger.info(f"TargetGeometricProcessor initialized: min_points={min_points}")

    def process_proposal(self, point_cloud: Dict) -> Optional[Dict]:
        """
        Extract geometric features from a single proposal point cloud.

        Args:
            point_cloud: Dict with 'points' (N, 3) and optional metadata

        Returns:
            Dict with:
                - points: (N, 3) point positions
                - features: (N, 32) GeDi features
                - num_points: int
                - proposal_id: int (from input)
            Or None if too few points
        """
        points = point_cloud['points']

        if len(points) < self.min_points:
            logger.warning(f"Skipping proposal with only {len(points)} points "
                          f"(min={self.min_points})")
            return None

        logger.debug(f"Processing proposal with {len(points)} points")

        # Extract GeDi features
        features = self.gedi.compute_features(points)

        result = {
            'points': points,
            'features': features,
            'num_points': len(points),
            'proposal_id': point_cloud.get('proposal_id', -1)
        }

        # Copy over any additional metadata
        for key in ['colors', 'mask', 'bbox', 'fx', 'fy', 'cx', 'cy']:
            if key in point_cloud:
                result[key] = point_cloud[key]

        logger.debug(f"Extracted features: {features.shape}")

        return result

    def process_proposals(self,
                          point_clouds: List[Dict],
                          max_points_per_proposal: int = 10000) -> List[Dict]:
        """
        Extract geometric features from multiple proposals.

        Args:
            point_clouds: List of point cloud dicts
            max_points_per_proposal: Downsample if proposal has more points

        Returns:
            List of feature dicts (only valid proposals)
        """
        results = []

        for i, pc in enumerate(point_clouds):
            # Use downsampling if point cloud is large
            if len(pc['points']) > max_points_per_proposal:
                result = self.process_with_downsampling(pc, target_points=max_points_per_proposal)
            else:
                result = self.process_proposal(pc)

            if result is not None:
                results.append(result)

        logger.info(f"Processed {len(results)}/{len(point_clouds)} proposals "
                   f"(filtered {len(point_clouds) - len(results)} with too few points)")

        return results

    def process_with_downsampling(self,
                                  point_cloud: Dict,
                                  target_points: int = 10000) -> Optional[Dict]:
        """
        Process proposal with adaptive downsampling.

        Args:
            point_cloud: Point cloud dict
            target_points: Target number of points (downsample if more)

        Returns:
            Feature dict or None
        """
        points = point_cloud['points']

        if len(points) < self.min_points:
            return None

        # Downsample if too many points
        if len(points) > target_points:
            logger.debug(f"Downsampling from {len(points)} to ~{target_points} points")

            # Random sampling
            indices = np.random.choice(len(points), target_points, replace=False)
            points = points[indices]

            # Update point cloud
            point_cloud = point_cloud.copy()
            point_cloud['points'] = points

            if 'colors' in point_cloud and point_cloud['colors'] is not None:
                point_cloud['colors'] = point_cloud['colors'][indices]

        return self.process_proposal(point_cloud)
