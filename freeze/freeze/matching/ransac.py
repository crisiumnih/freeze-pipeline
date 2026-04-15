"""
RANSAC-based 6D pose estimation from 3D-3D correspondences.
"""
import numpy as np
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class RANSACPoseEstimator:
    """Estimate 6D pose using RANSAC."""

    def __init__(self,
                 max_iterations: int = 1000,
                 inlier_threshold: float = 0.01,  # 1cm
                 min_inliers: int = 10,
                 confidence: float = 0.99):
        """
        Initialize RANSAC pose estimator.

        Args:
            max_iterations: Maximum RANSAC iterations
            inlier_threshold: Distance threshold for inliers (meters)
            min_inliers: Minimum inliers for valid pose
            confidence: Desired confidence level
        """
        self.max_iterations = max_iterations
        self.inlier_threshold = inlier_threshold
        self.min_inliers = min_inliers
        self.confidence = confidence

        logger.info(f"RANSACPoseEstimator initialized: "
                   f"max_iter={max_iterations}, thresh={inlier_threshold}m")

    def estimate_pose_svd(self,
                         src_points: np.ndarray,
                         dst_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate rigid transformation using SVD (closed-form solution).

        Args:
            src_points: (N, 3) source points
            dst_points: (N, 3) destination points

        Returns:
            R: (3, 3) rotation matrix
            t: (3,) translation vector
        """
        # Center the point clouds
        src_centroid = src_points.mean(axis=0)
        dst_centroid = dst_points.mean(axis=0)

        src_centered = src_points - src_centroid
        dst_centered = dst_points - dst_centroid

        # Compute covariance matrix
        H = src_centered.T @ dst_centered

        # SVD
        U, S, Vt = np.linalg.svd(H)

        # Rotation matrix
        R = Vt.T @ U.T

        # Handle reflection case
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        # Translation
        t = dst_centroid - R @ src_centroid

        return R, t

    def compute_inliers(self,
                       src_points: np.ndarray,
                       dst_points: np.ndarray,
                       R: np.ndarray,
                       t: np.ndarray) -> np.ndarray:
        """
        Compute inlier mask.

        Args:
            src_points: (N, 3) source points
            dst_points: (N, 3) destination points
            R: (3, 3) rotation
            t: (3,) translation

        Returns:
            inlier_mask: (N,) boolean mask
        """
        # Transform source points
        transformed = (R @ src_points.T).T + t

        # Compute distances
        distances = np.linalg.norm(transformed - dst_points, axis=1)

        # Inlier mask
        inlier_mask = distances < self.inlier_threshold

        return inlier_mask

    def estimate_pose_ransac(self,
                            query_points: np.ndarray,
                            target_points: np.ndarray) -> Optional[Dict]:
        """
        Estimate pose using RANSAC.

        Args:
            query_points: (N, 3) query object points
            target_points: (N, 3) corresponding target scene points

        Returns:
            Pose dict with R, t, inliers, score or None if failed
        """
        if len(query_points) < 3:
            logger.warning("Need at least 3 correspondences")
            return None

        n_correspondences = len(query_points)
        best_inliers = None
        best_R = None
        best_t = None
        best_score = 0

        logger.info(f"RANSAC with {n_correspondences} correspondences")

        for iteration in range(self.max_iterations):
            # Sample 3 random correspondences
            sample_indices = np.random.choice(n_correspondences, 3, replace=False)
            src_sample = query_points[sample_indices]
            dst_sample = target_points[sample_indices]

            # Estimate pose from sample
            try:
                R, t = self.estimate_pose_svd(src_sample, dst_sample)
            except np.linalg.LinAlgError:
                continue

            # Compute inliers
            inlier_mask = self.compute_inliers(query_points, target_points, R, t)
            num_inliers = inlier_mask.sum()

            # Update best if better
            if num_inliers > best_score:
                best_score = num_inliers
                best_inliers = inlier_mask
                best_R = R
                best_t = t

                # Early termination if enough inliers
                inlier_ratio = num_inliers / n_correspondences
                if inlier_ratio > 0.8 and num_inliers >= self.min_inliers:
                    logger.info(f"Early termination at iter {iteration}: "
                               f"{num_inliers} inliers ({inlier_ratio:.1%})")
                    break

        # Check if we have enough inliers
        if best_inliers is None or best_score < self.min_inliers:
            logger.warning(f"RANSAC failed: only {best_score} inliers (min={self.min_inliers})")
            return None

        # Refine pose using all inliers
        R_refined, t_refined = self.estimate_pose_svd(
            query_points[best_inliers],
            target_points[best_inliers]
        )

        # Recompute inliers with refined pose
        inlier_mask_refined = self.compute_inliers(
            query_points, target_points, R_refined, t_refined
        )
        num_inliers_refined = inlier_mask_refined.sum()

        logger.info(f"RANSAC success: {num_inliers_refined}/{n_correspondences} inliers "
                   f"({num_inliers_refined/n_correspondences:.1%})")

        # Compute RMSE on inliers
        inlier_query_points = query_points[inlier_mask_refined]
        inlier_target_points = target_points[inlier_mask_refined]
        transformed_points = (R_refined @ inlier_query_points.T).T + t_refined
        errors = np.linalg.norm(transformed_points - inlier_target_points, axis=1)
        rmse = np.sqrt(np.mean(errors ** 2))

        result = {
            'R': R_refined,
            't': t_refined,
            'inliers': inlier_mask_refined,
            'num_inliers': num_inliers_refined,
            'num_correspondences': n_correspondences,
            'inlier_ratio': num_inliers_refined / n_correspondences,
            'score': num_inliers_refined,
            'rmse': rmse
        }

        return result

    def pose_to_matrix(self, R: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Convert R, t to 4x4 transformation matrix.

        Args:
            R: (3, 3) rotation
            t: (3,) translation

        Returns:
            T: (4, 4) homogeneous transformation matrix
        """
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    def matrix_to_pose(self, T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract R, t from 4x4 transformation matrix.

        Args:
            T: (4, 4) transformation matrix

        Returns:
            R: (3, 3) rotation
            t: (3,) translation
        """
        R = T[:3, :3]
        t = T[:3, 3]
        return R, t

    def compute_pose_error(self,
                          R_est: np.ndarray,
                          t_est: np.ndarray,
                          R_gt: np.ndarray,
                          t_gt: np.ndarray) -> Dict:
        """
        Compute pose error metrics.

        Args:
            R_est, t_est: Estimated pose
            R_gt, t_gt: Ground truth pose

        Returns:
            Dict with rotation_error (degrees), translation_error (meters)
        """
        # Rotation error (angle of rotation difference)
        R_diff = R_gt.T @ R_est
        trace = np.trace(R_diff)
        # Clamp trace to [-1, 3] to avoid numerical issues
        trace = np.clip(trace, -1, 3)
        angle = np.arccos((trace - 1) / 2)
        rotation_error = np.degrees(angle)

        # Translation error (Euclidean distance)
        translation_error = np.linalg.norm(t_est - t_gt)

        return {
            'rotation_error_deg': rotation_error,
            'translation_error_m': translation_error
        }
