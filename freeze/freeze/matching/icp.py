"""
Iterative Closest Point (ICP) pose refinement.
Refines initial RANSAC pose estimates for better accuracy.
"""
import numpy as np
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ICPRefiner:
    """ICP-based pose refinement."""

    def __init__(self,
                 max_iterations: int = 50,
                 tolerance: float = 1e-6,
                 distance_threshold: Optional[float] = None,
                 subsample_source: Optional[int] = None):
        """
        Initialize ICP refiner.

        Args:
            max_iterations: Maximum ICP iterations
            tolerance: Convergence tolerance (change in error)
            distance_threshold: Max distance for valid correspondences (meters)
                              If None, uses adaptive threshold
            subsample_source: Subsample source points for speed (e.g., 1000)
        """
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.distance_threshold = distance_threshold
        self.subsample_source = subsample_source
        logger.info(f"ICPRefiner initialized (max_iter={max_iterations}, tol={tolerance})")

    def refine(self,
               source_points: np.ndarray,
               target_points: np.ndarray,
               initial_R: np.ndarray,
               initial_t: np.ndarray,
               return_history: bool = False) -> Dict:
        """
        Refine pose using ICP.

        Args:
            source_points: (N, 3) source point cloud (query)
            target_points: (M, 3) target point cloud (scene proposal)
            initial_R: (3, 3) initial rotation matrix
            initial_t: (3,) initial translation vector
            return_history: If True, return iteration history

        Returns:
            Dict with:
                - R: (3, 3) refined rotation
                - t: (3,) refined translation
                - num_iterations: Number of iterations run
                - final_error: Final mean distance
                - converged: Whether ICP converged
                - num_correspondences: Final number of correspondences
                - history: List of (error, R, t) per iteration (if return_history)
        """
        # Subsample source points if requested
        if self.subsample_source is not None and len(source_points) > self.subsample_source:
            indices = np.random.choice(len(source_points), self.subsample_source, replace=False)
            source_points = source_points[indices]
            logger.debug(f"Subsampled source to {len(source_points)} points")

        # Initialize transformation
        R = initial_R.copy()
        t = initial_t.copy()

        # Adaptive distance threshold
        if self.distance_threshold is None:
            # Use 3x the median distance as threshold
            transformed = (R @ source_points.T).T + t
            distances = np.linalg.norm(transformed[:, np.newaxis, :] - target_points[np.newaxis, :, :], axis=2)
            min_distances = distances.min(axis=1)
            threshold = 3.0 * np.median(min_distances)
            logger.debug(f"Adaptive distance threshold: {threshold:.4f}m")
        else:
            threshold = self.distance_threshold

        prev_error = float('inf')
        history = []

        for iteration in range(self.max_iterations):
            # Step 1: Transform source points with current pose
            transformed = (R @ source_points.T).T + t

            # Step 2: Find nearest neighbors (correspondences)
            # For each source point, find nearest target point
            correspondences, distances = self._find_correspondences(
                transformed, target_points, threshold
            )

            if len(correspondences) == 0:
                logger.warning(f"ICP iteration {iteration}: No valid correspondences")
                break

            # Step 3: Compute alignment error
            src_matched = source_points[correspondences[:, 0]]
            tgt_matched = target_points[correspondences[:, 1]]
            error = distances.mean()

            if return_history:
                history.append({
                    'iteration': iteration,
                    'error': error,
                    'num_correspondences': len(correspondences),
                    'R': R.copy(),
                    't': t.copy()
                })

            logger.debug(f"ICP iteration {iteration}: error={error:.6f}, "
                        f"correspondences={len(correspondences)}")

            # Step 4: Check convergence
            if abs(prev_error - error) < self.tolerance:
                logger.info(f"ICP converged at iteration {iteration} (error={error:.6f})")
                converged = True
                break

            prev_error = error

            # Step 5: Estimate new transformation
            R_new, t_new = self._estimate_transformation(src_matched, tgt_matched)

            # Update transformation
            R = R_new
            t = t_new

        else:
            logger.info(f"ICP reached max iterations ({self.max_iterations})")
            converged = False

        # Final correspondences
        transformed = (R @ source_points.T).T + t
        correspondences, distances = self._find_correspondences(
            transformed, target_points, threshold
        )

        result = {
            'R': R,
            't': t,
            'num_iterations': iteration + 1 if converged else self.max_iterations,
            'final_error': prev_error,
            'converged': converged,
            'num_correspondences': len(correspondences),
            'correspondence_indices': correspondences,
            'distances': distances
        }

        if return_history:
            result['history'] = history

        logger.info(f"ICP refinement complete: {result['num_iterations']} iterations, "
                   f"error={result['final_error']:.6f}, "
                   f"correspondences={result['num_correspondences']}")

        return result

    def _find_correspondences(self,
                             source: np.ndarray,
                             target: np.ndarray,
                             threshold: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find nearest neighbor correspondences.

        Args:
            source: (N, 3) source points
            target: (M, 3) target points
            threshold: Max distance for valid correspondence

        Returns:
            correspondences: (K, 2) array of [source_idx, target_idx]
            distances: (K,) distances for each correspondence
        """
        # Compute pairwise distances (N, M)
        # Use chunking to avoid memory issues
        chunk_size = 1000
        all_correspondences = []
        all_distances = []

        for start_idx in range(0, len(source), chunk_size):
            end_idx = min(start_idx + chunk_size, len(source))
            chunk = source[start_idx:end_idx]

            # Pairwise distances for this chunk
            diffs = chunk[:, np.newaxis, :] - target[np.newaxis, :, :]
            dists = np.linalg.norm(diffs, axis=2)

            # Find nearest neighbor for each point in chunk
            nearest_idx = dists.argmin(axis=1)
            nearest_dist = dists.min(axis=1)

            # Filter by threshold
            valid = nearest_dist < threshold
            valid_src_idx = np.arange(start_idx, end_idx)[valid]
            valid_tgt_idx = nearest_idx[valid]
            valid_dists = nearest_dist[valid]

            all_correspondences.append(np.stack([valid_src_idx, valid_tgt_idx], axis=1))
            all_distances.append(valid_dists)

        if len(all_correspondences) > 0:
            correspondences = np.concatenate(all_correspondences, axis=0)
            distances = np.concatenate(all_distances, axis=0)
        else:
            correspondences = np.zeros((0, 2), dtype=int)
            distances = np.zeros(0)

        return correspondences, distances

    def _estimate_transformation(self,
                                source: np.ndarray,
                                target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate rigid transformation using SVD (same as RANSAC).

        Args:
            source: (N, 3) source points
            target: (N, 3) target points

        Returns:
            R: (3, 3) rotation matrix
            t: (3,) translation vector
        """
        # Center point clouds
        source_centroid = source.mean(axis=0)
        target_centroid = target.mean(axis=0)

        source_centered = source - source_centroid
        target_centered = target - target_centroid

        # Compute covariance matrix
        H = source_centered.T @ target_centered

        # SVD
        U, S, Vt = np.linalg.svd(H)

        # Rotation matrix
        R = Vt.T @ U.T

        # Handle reflection case
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        # Translation
        t = target_centroid - R @ source_centroid

        return R, t

    def refine_pose_result(self,
                          pose_result: Dict,
                          query_points: np.ndarray,
                          target_points: np.ndarray) -> Dict:
        """
        Convenience method to refine a pose result dict.

        Args:
            pose_result: Dict from RANSAC with R, t
            query_points: Query point cloud
            target_points: Target point cloud

        Returns:
            Refined pose result dict with ICP refinement info
        """
        if pose_result is None:
            return None

        # Refine the pose
        icp_result = self.refine(
            query_points,
            target_points,
            pose_result['R'],
            pose_result['t']
        )

        # Create refined result
        refined = pose_result.copy()
        refined['R'] = icp_result['R']
        refined['t'] = icp_result['t']
        refined['icp_iterations'] = icp_result['num_iterations']
        refined['icp_error'] = icp_result['final_error']
        refined['icp_converged'] = icp_result['converged']
        refined['icp_correspondences'] = icp_result['num_correspondences']

        # Mark as refined
        refined['refined'] = True

        return refined
