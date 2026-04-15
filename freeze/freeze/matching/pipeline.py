"""
Complete matching pipeline for 6D pose estimation.
Orchestrates correspondence matching and RANSAC pose estimation.
"""
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import logging

from .correspondence import CorrespondenceMatcher
from .ransac import RANSACPoseEstimator
from .icp import ICPRefiner

logger = logging.getLogger(__name__)


class MatchingPipeline:
    """Complete pipeline for matching query to scene and estimating pose."""

    def __init__(self,
                 feature_distance_threshold: float = 0.15,
                 ransac_inlier_threshold: float = 0.01,
                 ransac_iterations: int = 1000,
                 min_correspondences: int = 10,
                 min_inliers: int = 10,
                 use_icp: bool = False,
                 icp_max_iterations: int = 50,
                 icp_tolerance: float = 1e-6):
        """
        Initialize matching pipeline.

        Args:
            feature_distance_threshold: Max feature distance for correspondence
            ransac_inlier_threshold: Spatial distance threshold for RANSAC inliers (meters)
            ransac_iterations: Max RANSAC iterations
            min_correspondences: Min correspondences for matching
            min_inliers: Min inliers for valid pose
            use_icp: Whether to refine poses with ICP
            icp_max_iterations: Max ICP iterations
            icp_tolerance: ICP convergence tolerance
        """
        self.matcher = CorrespondenceMatcher(
            distance_threshold=feature_distance_threshold,
            min_correspondences=min_correspondences
        )

        self.ransac = RANSACPoseEstimator(
            max_iterations=ransac_iterations,
            inlier_threshold=ransac_inlier_threshold,
            min_inliers=min_inliers
        )

        self.use_icp = use_icp
        if use_icp:
            self.icp = ICPRefiner(
                max_iterations=icp_max_iterations,
                tolerance=icp_tolerance
            )
        else:
            self.icp = None

        logger.info(f"MatchingPipeline initialized (ICP={'enabled' if use_icp else 'disabled'})")

    def estimate_pose_for_proposal(self,
                                   query_data: Dict,
                                   proposal_data: Dict,
                                   feature_type: str = 'geometric') -> Optional[Dict]:
        """
        Estimate pose for query in a single proposal.

        Args:
            query_data: Query feature dict
            proposal_data: Proposal feature dict
            feature_type: 'geometric', 'visual', or 'fused'

        Returns:
            Pose dict with R, t, inliers, score or None if failed
        """
        # Step 1: Match features
        match = self.matcher.match_to_proposal(query_data, proposal_data, feature_type)

        if match is None:
            return None

        # Step 2: RANSAC pose estimation
        pose = self.ransac.estimate_pose_ransac(
            match['query_points'],
            match['target_points']
        )

        if pose is None:
            return None

        # Step 3: ICP refinement (optional)
        if self.use_icp:
            pose = self.icp.refine_pose_result(
                pose,
                query_data['points'],
                proposal_data['points']
            )

        # Add match info to pose result
        pose['proposal_id'] = proposal_data.get('proposal_id', -1)
        pose['num_correspondences'] = match['num_correspondences']
        pose['feature_type'] = feature_type
        pose['success'] = True  # Mark as successful

        return pose

    def estimate_pose_in_scene(self,
                               query_data: Dict,
                               scene_proposals: List[Dict],
                               feature_type: str = 'geometric',
                               return_all: bool = False) -> Optional[Dict]:
        """
        Estimate pose for query in scene (try all proposals).

        Args:
            query_data: Query feature dict
            scene_proposals: List of proposal feature dicts
            feature_type: Feature type to use
            return_all: If True, return all successful poses; else return best

        Returns:
            Best pose dict or list of all poses (if return_all=True)
        """
        logger.info(f"Estimating pose in scene with {len(scene_proposals)} proposals")

        poses = []

        for i, proposal in enumerate(scene_proposals):
            pose = self.estimate_pose_for_proposal(query_data, proposal, feature_type)

            if pose is not None:
                poses.append(pose)
                logger.info(f"  Proposal {proposal.get('proposal_id', i)}: "
                           f"{pose['num_inliers']}/{pose['num_correspondences']} inliers "
                           f"(score={pose['score']})")

        if len(poses) == 0:
            logger.warning("No valid poses found in any proposal")
            return None if not return_all else []

        # Sort by score (number of inliers)
        poses.sort(key=lambda x: x['score'], reverse=True)

        if return_all:
            logger.info(f"Found {len(poses)} valid poses")
            return poses
        else:
            best_pose = poses[0]
            logger.info(f"Best pose: proposal {best_pose['proposal_id']}, "
                       f"score={best_pose['score']}, "
                       f"inlier_ratio={best_pose['inlier_ratio']:.1%}")
            return best_pose

    def transform_points(self,
                        points: np.ndarray,
                        R: np.ndarray,
                        t: np.ndarray) -> np.ndarray:
        """
        Transform points using pose.

        Args:
            points: (N, 3) points
            R: (3, 3) rotation
            t: (3,) translation

        Returns:
            transformed_points: (N, 3)
        """
        return (R @ points.T).T + t

    def evaluate_pose(self,
                     query_points: np.ndarray,
                     target_points: np.ndarray,
                     R: np.ndarray,
                     t: np.ndarray,
                     inlier_threshold: Optional[float] = None) -> Dict:
        """
        Evaluate pose quality.

        Args:
            query_points: (N, 3) query points
            target_points: (N, 3) target points
            R, t: Estimated pose
            inlier_threshold: Override default inlier threshold

        Returns:
            Evaluation dict with metrics
        """
        if inlier_threshold is None:
            inlier_threshold = self.ransac.inlier_threshold

        # Transform query points
        transformed = self.transform_points(query_points, R, t)

        # Compute distances
        distances = np.linalg.norm(transformed - target_points, axis=1)

        # Inliers
        inliers = distances < inlier_threshold

        # Statistics
        eval_result = {
            'num_inliers': inliers.sum(),
            'num_points': len(query_points),
            'inlier_ratio': inliers.sum() / len(query_points),
            'mean_distance': distances.mean(),
            'median_distance': np.median(distances),
            'max_distance': distances.max(),
            'distances': distances,
            'inliers': inliers
        }

        return eval_result
