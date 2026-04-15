"""
Multi-view renderer for query objects.
Renders 3D meshes from multiple viewpoints for visual feature extraction.
Uses GeDi environment (Python 3.8 + Open3D 0.15.1) via subprocess for stable rendering.
"""
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Optional, Union
import logging
import subprocess
import json
import tempfile

logger = logging.getLogger(__name__)


class MultiViewRenderer:
    """Render 3D mesh from multiple viewpoints using GeDi environment."""

    def __init__(self, image_size=(640, 480), env_path=None):
        """
        Initialize renderer.

        Args:
            image_size: (width, height) of rendered images
            env_path: Path to GeDi environment (auto-detected if None)
        """
        self.image_size = image_size

        if env_path is None:
            env_path = Path(__file__).parent.parent.parent.parent / 'environments/gedi'

        self.env_path = Path(env_path)
        self.python = self.env_path / '.venv/bin/python'
        self.server = self.env_path / 'render_server.py'

        if not self.python.exists():
            raise RuntimeError(
                f"GeDi environment not found at {self.env_path}. "
                f"Run: cd {self.env_path} && bash setup.sh"
            )

        logger.info(f"MultiViewRenderer initialized with image size {image_size}")

    def render(self,
               mesh_path: Union[str, Path],
               num_views: int = 8,
               distance: float = 2.0,
               elevation_angles: Optional[List[float]] = None
               ) -> List[Tuple[np.ndarray, dict]]:
        """
        Render mesh from multiple viewpoints.

        Args:
            mesh_path: Path to .obj mesh file
            num_views: Number of azimuth views (evenly spaced around object)
            distance: Camera distance multiplier
            elevation_angles: List of elevation angles in degrees (default: [0, 30])

        Returns:
            List of (image, depth, metadata) tuples
        """
        if elevation_angles is None:
            elevation_angles = [0, 30]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            output_dir = tmp / 'renders'
            output_dir.mkdir()

            # Create request
            request = {
                'mesh_path': str(mesh_path),
                'num_views': num_views,
                'elevation_angles': elevation_angles,
                'output_dir': str(output_dir)
            }

            request_path = tmp / 'request.json'
            with open(request_path, 'w') as f:
                json.dump(request, f)

            # Call render server
            try:
                result = subprocess.run(
                    [str(self.python), str(self.server), str(request_path)],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=300  # 5 minute timeout
                )
            except subprocess.CalledProcessError as e:
                raise RuntimeError(
                    f"Rendering failed:\n"
                    f"STDOUT: {e.stdout}\n"
                    f"STDERR: {e.stderr}"
                )
            except subprocess.TimeoutExpired:
                raise RuntimeError("Rendering timed out (>5 minutes)")

            # Parse result
            info = json.loads(result.stdout)

            # Load rendered images and depth maps
            rendered_views = []
            for view_info in info['views']:
                view_path = Path(view_info['path'])
                image = np.array(Image.open(view_path))

                # Load depth map
                depth_path = Path(view_info['depth_path'])
                depth = np.load(depth_path)

                metadata = {
                    'azimuth': view_info['azimuth'],
                    'elevation': view_info['elevation'],
                    'distance': view_info['distance']
                }

                rendered_views.append((image, depth, metadata))

        logger.info(f"Rendered {len(rendered_views)} views")
        return rendered_views

    def render_and_save(self,
                        mesh_path: Union[str, Path],
                        output_dir: Union[str, Path],
                        num_views: int = 8,
                        **kwargs) -> List[Path]:
        """
        Render views and save to disk.

        Args:
            mesh_path: Path to .obj mesh file
            output_dir: Directory to save rendered images
            num_views: Number of azimuth views
            **kwargs: Additional arguments for render()

        Returns:
            List of paths to saved images
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        views = self.render(mesh_path, num_views=num_views, **kwargs)

        saved_paths = []
        for i, (image, depth, metadata) in enumerate(views):
            filename = f"view_{i:03d}_az{metadata['azimuth']:.0f}_el{metadata['elevation']:.0f}.png"
            image_path = output_dir / filename

            Image.fromarray(image).save(image_path)

            # Also save depth
            depth_filename = f"depth_{i:03d}_az{metadata['azimuth']:.0f}_el{metadata['elevation']:.0f}.npy"
            depth_path = output_dir / depth_filename
            np.save(depth_path, depth)

            saved_paths.append(image_path)

        logger.info(f"Saved {len(saved_paths)} views to {output_dir}")
        return saved_paths
