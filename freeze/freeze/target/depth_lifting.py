"""
Depth lifting: RGB-D images to 3D point clouds.
Converts masked depth images to 3D point clouds for feature extraction.
"""
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


class DepthLifter:
    """Lift 2D masks with depth to 3D point clouds."""

    def __init__(self,
                 fx: float = 525.0,
                 fy: float = 525.0,
                 cx: float = 319.5,
                 cy: float = 239.5,
                 depth_scale: float = 1000.0):
        """
        Initialize depth lifter with camera intrinsics.

        Args:
            fx, fy: Focal lengths in pixels
            cx, cy: Principal point in pixels
            depth_scale: Depth units to meters (e.g., 1000 for mm → m)
        """
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.depth_scale = depth_scale

        # Camera intrinsic matrix
        self.K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])

        logger.info(f"DepthLifter initialized: fx={fx}, fy={fy}, cx={cx}, cy={cy}, scale={depth_scale}")

    def lift_proposal(self,
                     depth_image: np.ndarray,
                     mask: np.ndarray,
                     rgb_image: Optional[np.ndarray] = None,
                     subsample: Optional[int] = None) -> Dict:
        """
        Lift a masked region to 3D point cloud.

        Args:
            depth_image: (H, W) depth map in depth units
            mask: (H, W) boolean mask
            rgb_image: Optional (H, W, 3) RGB image for colors
            subsample: Optional subsampling factor (take every Nth point)

        Returns:
            Dictionary with:
                - points: (N, 3) 3D point cloud [x, y, z] in meters
                - colors: (N, 3) RGB colors [0-1] if rgb_image provided
                - valid_mask: (N,) boolean mask of valid points (depth > 0)
                - num_points: int
        """
        height, width = depth_image.shape
        assert mask.shape == (height, width), "Mask and depth must have same shape"

        # Get masked pixels
        ys, xs = np.where(mask)

        if len(xs) == 0:
            logger.warning("Empty mask, no points to lift")
            return {
                'points': np.zeros((0, 3)),
                'colors': np.zeros((0, 3)),
                'valid_mask': np.zeros(0, dtype=bool),
                'num_points': 0
            }

        # Subsample if requested
        if subsample is not None and subsample > 1:
            indices = np.arange(len(xs))[::subsample]
            xs = xs[indices]
            ys = ys[indices]

        # Get depth values
        depths = depth_image[ys, xs].astype(np.float32) / self.depth_scale  # Convert to meters

        # Filter invalid depths
        valid = depths > 0
        xs_valid = xs[valid]
        ys_valid = ys[valid]
        depths_valid = depths[valid]

        if len(depths_valid) == 0:
            logger.warning("No valid depths in mask")
            return {
                'points': np.zeros((0, 3)),
                'colors': np.zeros((0, 3)),
                'valid_mask': np.zeros(0, dtype=bool),
                'num_points': 0
            }

        # Back-project to 3D
        # x = (u - cx) * z / fx
        # y = (v - cy) * z / fy
        # z = depth
        x = (xs_valid - self.cx) * depths_valid / self.fx
        y = (ys_valid - self.cy) * depths_valid / self.fy
        z = depths_valid

        points = np.stack([x, y, z], axis=1)

        # Extract colors if RGB provided
        colors = None
        if rgb_image is not None:
            colors = rgb_image[ys_valid, xs_valid].astype(np.float32) / 255.0

        logger.debug(f"Lifted {len(points)} points from mask (area={mask.sum()})")

        return {
            'points': points,
            'colors': colors if colors is not None else np.zeros((len(points), 3)),
            'valid_mask': valid,
            'num_points': len(points)
        }

    def lift_multiple_proposals(self,
                                depth_image: np.ndarray,
                                proposals: list,
                                rgb_image: Optional[np.ndarray] = None,
                                subsample: Optional[int] = None) -> list:
        """
        Lift multiple proposals to point clouds.

        Args:
            depth_image: (H, W) depth map
            proposals: List of proposal dicts with 'mask' key
            rgb_image: Optional RGB image
            subsample: Optional subsampling factor

        Returns:
            List of point cloud dicts (same order as proposals)
        """
        point_clouds = []

        for i, proposal in enumerate(proposals):
            mask = proposal['mask']
            pc = self.lift_proposal(depth_image, mask, rgb_image, subsample)
            pc['proposal_id'] = proposal.get('id', i)
            point_clouds.append(pc)

        logger.info(f"Lifted {len(point_clouds)} proposals to point clouds")

        return point_clouds

    def set_intrinsics(self, fx: float, fy: float, cx: float, cy: float):
        """Update camera intrinsics."""
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
        logger.info(f"Updated intrinsics: fx={fx}, fy={fy}, cx={cx}, cy={cy}")

    @staticmethod
    def load_depth_image(path: Union[str, Path], depth_scale: Optional[float] = None) -> np.ndarray:
        """
        Load depth image from file.

        Args:
            path: Path to depth image (.png, .npy)
            depth_scale: Optional scale override

        Returns:
            (H, W) depth array
        """
        from PIL import Image

        path = Path(path)

        if path.suffix == '.npy':
            depth = np.load(path)
        elif path.suffix in ['.png', '.jpg', '.jpeg']:
            depth = np.array(Image.open(path))
        else:
            raise ValueError(f"Unsupported depth format: {path.suffix}")

        logger.debug(f"Loaded depth image: {depth.shape}, range=[{depth.min()}, {depth.max()}]")

        return depth

    @staticmethod
    def estimate_intrinsics_from_fov(width: int, height: int, fov_deg: float = 60.0) -> Tuple[float, float, float, float]:
        """
        Estimate camera intrinsics from image size and FOV.

        Args:
            width, height: Image dimensions
            fov_deg: Vertical field of view in degrees

        Returns:
            (fx, fy, cx, cy)
        """
        fov_rad = np.radians(fov_deg)
        fy = height / (2.0 * np.tan(fov_rad / 2.0))
        fx = fy  # Assume square pixels

        cx = width / 2.0
        cy = height / 2.0

        return fx, fy, cx, cy
