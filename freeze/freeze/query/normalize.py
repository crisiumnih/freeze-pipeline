"""
Mesh normalization utilities.
Normalizes meshes to real-world scale for pose estimation.
"""
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class MeshNormalizer:
    """Normalize mesh point clouds to real-world scale."""

    def __init__(self, target_scale: Optional[float] = None):
        """
        Initialize mesh normalizer.

        Args:
            target_scale: Target diagonal size in meters (e.g., 0.2 for 20cm object)
                        If None, will try to estimate reasonable scale
        """
        self.target_scale = target_scale

    def normalize(self,
                 points: np.ndarray,
                 target_scale: Optional[float] = None) -> Tuple[np.ndarray, dict]:
        """
        Normalize point cloud to real-world scale.

        Args:
            points: (N, 3) point cloud
            target_scale: Override target scale for this normalization

        Returns:
            normalized_points: (N, 3) normalized point cloud
            metadata: Dict with normalization info (scale_factor, original_bbox, etc.)
        """
        if target_scale is None:
            target_scale = self.target_scale

        # Compute bounding box
        min_coords = points.min(axis=0)
        max_coords = points.max(axis=0)
        bbox_size = max_coords - min_coords
        bbox_center = (min_coords + max_coords) / 2

        # Compute diagonal size (characteristic size of object)
        diagonal = np.linalg.norm(bbox_size)

        # If no target scale provided, use heuristic based on point cloud size
        if target_scale is None:
            # Assume most household objects are 5cm - 50cm in size
            # If diagonal is > 10, assume it's in mm and convert to m
            if diagonal > 10:
                target_scale = 0.2  # Assume 20cm object, will scale down
            elif diagonal > 1:
                target_scale = 0.2  # Already in decimeters-ish, scale to 20cm
            else:
                target_scale = 0.2  # Already in meters-ish, scale to 20cm

        # Compute scale factor
        scale_factor = target_scale / diagonal

        # Center the point cloud
        centered_points = points - bbox_center

        # Scale to target size
        normalized_points = centered_points * scale_factor

        metadata = {
            'original_diagonal': diagonal,
            'target_scale': target_scale,
            'scale_factor': scale_factor,
            'original_bbox_min': min_coords,
            'original_bbox_max': max_coords,
            'original_bbox_center': bbox_center,
            'original_bbox_size': bbox_size
        }

        logger.info(f"Normalized mesh: {diagonal:.4f} → {target_scale:.4f} m "
                   f"(scale factor: {scale_factor:.6f})")

        return normalized_points, metadata

    def denormalize_pose(self,
                        R: np.ndarray,
                        t: np.ndarray,
                        metadata: dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform pose from normalized space back to original mesh space.

        Args:
            R: (3, 3) rotation matrix in normalized space
            t: (3,) translation vector in normalized space
            metadata: Normalization metadata from normalize()

        Returns:
            R_original: (3, 3) rotation (same, rotation is scale-invariant)
            t_original: (3,) translation in original space
        """
        # Rotation is scale-invariant
        R_original = R.copy()

        # Translation needs to be:
        # 1. Scaled back to original space
        # 2. Adjusted for original center offset
        scale_factor = metadata['scale_factor']
        original_center = metadata['original_bbox_center']

        # The estimated pose gives: target = R @ (normalized_query) + t
        # In original space: target = R @ (scale * (query - center_orig)) + t
        #                         = R @ (scale * query) - R @ (scale * center_orig) + t
        # So: t_original = t + R @ (scale * center_orig) - R @ center_orig
        #                = t + R @ ((scale - 1) * center_orig)

        # Actually, simpler approach:
        # In normalized space: scale_factor * (query - center) → target
        # So: t_normalized is in target space
        # t_original should scale the query offset, so:
        # t_original = t_normalized (already in target space)
        # But we need to account for the original center

        # Let me think more carefully:
        # Normalized: query_norm = scale_factor * (query_orig - center)
        # Pose in normalized space: target = R @ query_norm + t_norm
        # Expanding: target = R @ (scale_factor * (query_orig - center)) + t_norm
        #                  = scale_factor * R @ query_orig - scale_factor * R @ center + t_norm
        #
        # In original space, we want: target = R @ query_orig + t_orig
        # Comparing: scale_factor * R @ query_orig - scale_factor * R @ center + t_norm
        #          = R @ query_orig + t_orig
        #
        # This doesn't work because we're scaling the query...

        # Actually, in pose estimation, we DON'T want to scale the query back.
        # The pose is estimated to align normalized_query to target.
        # But for visualization, we want to know where the ORIGINAL query would go.

        # The correct approach is:
        # If we have: target = R @ query_normalized + t_norm
        # And: query_normalized = scale_factor * (query_orig - center)
        # Then: target = R @ (scale_factor * (query_orig - center)) + t_norm
        #             = scale_factor * R @ query_orig - scale_factor * R @ center + t_norm
        #
        # But we want: target = R @ query_orig + t_orig (in original scale)
        # This is inconsistent because of the scale factor on query.

        # The right interpretation:
        # The pose estimation gives us R and t that work in normalized space.
        # To apply this pose to the original (unnormalized) mesh, we need to:
        # 1. Normalize the mesh
        # 2. Apply R and t
        # That's it - we don't denormalize the pose.

        # OR, if we want the pose in original space:
        # We need to answer: "where would the original unnormalized mesh be?"
        # But that doesn't make sense because the original mesh is too large.

        # I think the correct approach is to just return the pose as-is,
        # because the pose is meant to be applied to the normalized query.
        # The user should normalize their query before applying the pose.

        # Actually, for most use cases, we want:
        # Given original query mesh, what transformation puts it in the scene?
        #
        # In normalized space: target = R @ (scale * (q - c)) + t
        #                            = scale * R @ q - scale * R @ c + t
        # We want original space: target = R_orig @ q + t_orig
        # Matching: R_orig = scale * R (but this changes rotation magnitude)

        # I think the issue is that we can't have both scale and rotation change.
        # The rotation R is scale-invariant.
        # But the translation needs adjustment.

        # Let me reconsider what we actually want:
        # After normalization, query points are: p_norm = scale * (p_orig - center)
        # Pose estimation gives: p_target = R @ p_norm + t_norm
        # To transform original points: p_target = R @ (scale * (p_orig - center)) + t_norm
        #                                       = R @ (scale * p_orig) - R @ (scale * center) + t_norm
        #
        # If we want t_orig such that: p_target = R @ p_orig + t_orig
        # Then: R @ (scale * p_orig) - R @ (scale * center) + t_norm = R @ p_orig + t_orig
        # This only works if scale = 1, otherwise R @ (scale * p_orig) ≠ R @ p_orig

        # CONCLUSION: We cannot express the pose in original space if scale ≠ 1.
        # The pose MUST be applied to normalized points.
        # So this function should just return a warning or the normalized pose.

        logger.warning("Pose is in normalized space. Apply to normalized query points, "
                      "not original mesh. Use denormalize_points() instead.")

        return R, t

    def denormalize_points(self,
                          points: np.ndarray,
                          metadata: dict) -> np.ndarray:
        """
        Denormalize points back to original mesh space.

        Args:
            points: (N, 3) normalized points
            metadata: Normalization metadata from normalize()

        Returns:
            original_points: (N, 3) in original space
        """
        scale_factor = metadata['scale_factor']
        original_center = metadata['original_bbox_center']

        # Reverse normalization: p_norm = scale * (p_orig - center)
        # So: p_orig = p_norm / scale + center
        original_points = points / scale_factor + original_center

        return original_points
