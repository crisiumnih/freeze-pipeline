"""
Complete query object processing pipeline.
Orchestrates mesh sampling, normalization, and feature extraction (geometric + visual).
"""
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Union
import subprocess
import json
import tempfile
import logging

from .normalize import MeshNormalizer
from .geometric import GeometricProcessor
from .renderer import MultiViewRenderer
from .visual import VisualProcessor
from .backproject import FeatureBackProjector
from .fusion import FeatureFusion

logger = logging.getLogger(__name__)


class QueryPipeline:
    """Complete pipeline for processing query objects."""

    def __init__(self,
                 gedi_client,
                 dinov2_client=None,
                 use_visual: bool = True,
                 target_scale: float = 0.2,
                 num_points: int = 10000,
                 num_views: int = 8,
                 k_neighbors: int = 5):
        """
        Initialize query processing pipeline.

        Args:
            gedi_client: GeDiClient for geometric features
            dinov2_client: Optional DINOv2Client for visual features
            use_visual: Whether to extract and fuse visual features
            target_scale: Target mesh size in meters (default 20cm)
            num_points: Number of points to sample from mesh
            num_views: Number of rendered views for visual features
            k_neighbors: k for k-NN aggregation in fusion
        """
        self.gedi = gedi_client
        self.dinov2 = dinov2_client
        self.use_visual = use_visual and dinov2_client is not None
        self.num_points = num_points
        self.num_views = num_views

        # Initialize sub-modules
        self.normalizer = MeshNormalizer(target_scale=target_scale)
        self.geo_processor = GeometricProcessor(gedi_client)

        if self.use_visual:
            self.renderer = MultiViewRenderer(
                image_size=(640, 480)
            )
            self.vis_processor = VisualProcessor(dinov2_client)
            self.backprojector = FeatureBackProjector()
            self.fusion = FeatureFusion(
                fusion_method='concatenate',
                k_neighbors=k_neighbors,
                distance_threshold=0.1  # 10cm threshold
            )
        else:
            self.renderer = None
            self.vis_processor = None
            self.backprojector = None
            self.fusion = None

        logger.info(f"QueryPipeline initialized")
        logger.info(f"  Visual features: {self.use_visual}")
        logger.info(f"  Target scale: {target_scale}m")
        logger.info(f"  Sample points: {num_points}")
        if self.use_visual:
            logger.info(f"  Render views: {num_views}")

    def process(self,
                mesh_path: Union[str, Path],
                num_points: Optional[int] = None,
                output_dir: Optional[Path] = None) -> Dict:
        """
        Process mesh through complete pipeline.

        Args:
            mesh_path: Path to mesh file (.obj, .ply, etc.)
            num_points: Override default point count
            output_dir: Optional directory to save debug outputs

        Returns:
            Dict with:
                - points: (N, 3) normalized 3D points
                - features: (N, D) features (32 or 1056-dim)
                - geometric_features: (N, 32) geometric only
                - visual_features: (N, 1024) visual only (if use_visual)
                - normalization: Normalization metadata
                - mesh_path: Input mesh path
                - num_points: Number of points
                - feature_type: 'geometric' or 'fused'
        """
        mesh_path = Path(mesh_path)
        if not mesh_path.exists():
            raise FileNotFoundError(f"Mesh not found: {mesh_path}")

        if num_points is None:
            num_points = self.num_points

        logger.info(f"Processing query: {mesh_path.name}")

        # Stage 1: Sample mesh
        logger.info("[1/3] Sampling mesh...")
        points_raw = self._sample_mesh(mesh_path, num_points)
        logger.info(f"  → Sampled {len(points_raw)} points")

        # Stage 2: Normalize
        logger.info("[2/3] Normalizing mesh...")
        points_norm, norm_metadata = self.normalizer.normalize(points_raw)
        logger.info(f"  → Normalized: {norm_metadata['original_diagonal']:.3f}m "
                   f"→ {norm_metadata['target_scale']:.3f}m "
                   f"(factor: {norm_metadata['scale_factor']:.6f})")

        # Stage 3: Geometric features
        logger.info("[3/3] Extracting geometric features...")
        geo_result = self.geo_processor.process(points_norm)
        logger.info(f"  → Features: {geo_result['features'].shape}")

        # Build result (start with geometric only)
        result = {
            'points': geo_result['points'],
            'features': geo_result['features'],
            'geometric_features': geo_result['features'],
            'normalization': norm_metadata,
            'mesh_path': str(mesh_path),
            'num_points': len(geo_result['points']),
            'feature_type': 'geometric'
        }

        # Stages 4-7: Visual features (optional)
        if self.use_visual:
            logger.info("\n[4/7] Processing visual features...")

            visual_result = self._process_visual(
                mesh_path,
                points_norm,
                norm_metadata,
                output_dir
            )

            logger.info("[5/7] Aggregating visual features...")
            vis_aggregated = self._aggregate_visual(
                geo_result['points'],
                visual_result['points_3d'],
                visual_result['features']
            )

            logger.info("[6/7] Fusing geometric + visual features...")
            fused = np.concatenate([
                geo_result['features'],
                vis_aggregated
            ], axis=1)

            logger.info(f"  → Fused features: {fused.shape}")

            # Update result with fused features
            result['features'] = fused
            result['visual_features'] = vis_aggregated
            result['feature_type'] = 'fused'

        logger.info(f"\n✓ Query processing complete: {result['feature_type']} features")

        return result

    def _sample_mesh(self, mesh_path: Path, num_points: int) -> np.ndarray:
        """
        Sample points from mesh using GeDi environment.

        Args:
            mesh_path: Path to mesh file
            num_points: Number of points to sample

        Returns:
            points: (N, 3) sampled points
        """
        # Use GeDi environment's mesh sampler (subprocess)
        env_path = Path(__file__).parent.parent.parent.parent / 'environments/gedi'
        python = env_path / '.venv/bin/python'
        sampler = env_path / 'mesh_sampler.py'

        if not python.exists():
            raise RuntimeError(f"GeDi Python not found: {python}")
        if not sampler.exists():
            raise RuntimeError(f"Mesh sampler not found: {sampler}")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            output_path = tmp / 'sampled_points.npy'

            request = {
                'mesh_path': str(mesh_path),
                'num_points': num_points,
                'output_path': str(output_path)
            }

            request_path = tmp / 'request.json'
            with open(request_path, 'w') as f:
                json.dump(request, f)

            # Call sampler
            result = subprocess.run(
                [str(python), str(sampler), str(request_path)],
                capture_output=True,
                text=True,
                check=True,
                timeout=60
            )

            # Load sampled points
            points = np.load(output_path)

        return points

    def _process_visual(self,
                       mesh_path: Path,
                       points_norm: np.ndarray,
                       norm_metadata: Dict,
                       output_dir: Optional[Path] = None) -> Dict:
        """
        Process visual features through render → DINOv2 → backproject.

        Args:
            mesh_path: Path to mesh file
            points_norm: Normalized point cloud
            norm_metadata: Normalization metadata
            output_dir: Optional debug output directory

        Returns:
            Dict with:
                - points: (M, 3) 3D points with visual features
                - features: (M, 1024) visual features
        """
        # Stage 4: Render multi-view
        logger.info("  [4a] Rendering multi-view...")
        views = self.renderer.render(
            mesh_path,
            num_views=self.num_views,
            elevation_angles=[0, 30]
        )

        logger.info(f"  → Rendered {len(views)} views")

        # Stage 5: Extract visual features from each view
        logger.info("  [4b] Extracting DINOv2 features...")
        view_results = []
        depth_maps = []

        # Need temp directory for saving images (DINOv2 expects file paths)
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_dir = Path(tmpdir)

            for i, (image, depth, metadata) in enumerate(views):
                # Save temp image
                temp_image = temp_dir / f'temp_view_{i}.png'
                from PIL import Image
                Image.fromarray(image).save(temp_image)

                # Extract DINOv2 features
                result = self.vis_processor.process(temp_image)
                result['metadata'] = metadata

                view_results.append(result)
                depth_maps.append(depth)

        logger.info(f"  → Extracted features from {len(view_results)} views")

        # Stage 6: Back-project to 3D
        logger.info("  [4c] Back-projecting to 3D...")
        visual_result = self.backprojector.backproject_multiview(
            view_results,
            depth_maps
        )

        logger.info(f"  → Back-projected {visual_result['metadata']['num_points']} visual points")

        return visual_result

    def _aggregate_visual(self,
                         geo_points: np.ndarray,
                         vis_points: np.ndarray,
                         vis_features: np.ndarray) -> np.ndarray:
        """
        Aggregate visual features to geometric points using k-NN.

        Args:
            geo_points: (N, 3) geometric points
            vis_points: (M, 3) visual feature points
            vis_features: (M, 1024) visual features

        Returns:
            aggregated: (N, 1024) visual features for each geometric point
        """
        # Reconstruct dict format expected by FeatureFusion
        geo_dict = {
            'points': geo_points,
            'features': np.zeros((len(geo_points), 1), dtype=np.float32)  # Dummy geometric
        }

        vis_dict = {
            'points_3d': vis_points,
            'features': vis_features
        }

        # Fuse (will do k-NN aggregation internally)
        result = self.fusion.fuse(geo_dict, vis_dict)

        # Extract just the aggregated visual features
        vis_aggregated = result['visual_features']

        logger.info(f"  → Aggregated {vis_aggregated.shape[0]} visual features "
                   f"from {len(vis_points)} points")

        return vis_aggregated

    def process_batch(self,
                     mesh_paths: list,
                     output_dir: Optional[Path] = None) -> list:
        """
        Process multiple meshes in batch.

        Args:
            mesh_paths: List of mesh file paths
            output_dir: Optional directory for debug outputs

        Returns:
            List of result dicts
        """
        results = []

        for i, mesh_path in enumerate(mesh_paths):
            logger.info(f"\n{'='*70}")
            logger.info(f"Processing {i+1}/{len(mesh_paths)}: {Path(mesh_path).name}")
            logger.info(f"{'='*70}")

            try:
                if output_dir:
                    mesh_output = output_dir / Path(mesh_path).stem
                    mesh_output.mkdir(parents=True, exist_ok=True)
                else:
                    mesh_output = None

                result = self.process(mesh_path, output_dir=mesh_output)
                results.append(result)

            except Exception as e:
                logger.error(f"Failed to process {mesh_path}: {e}")
                results.append(None)

        logger.info(f"\nBatch complete: {sum(r is not None for r in results)}/{len(mesh_paths)} succeeded")

        return results
