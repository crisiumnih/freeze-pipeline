"""
DINOv2 Client - Clean Python API for DINOv2 visual feature extraction
"""
import subprocess
import json
import tempfile
import numpy as np
from pathlib import Path


class DINOv2Client:
    """Client for DINOv2 visual feature extraction."""

    def __init__(self, env_path=None):
        """
        Initialize DINOv2 client.

        Args:
            env_path: Path to DINOv2 environment (auto-detected if None)
        """
        if env_path is None:
            env_path = Path(__file__).parent.parent.parent.parent / 'environments/dinov2'

        self.env_path = Path(env_path)
        self.python = self.env_path / '.venv/bin/python'
        self.server = self.env_path / 'server.py'

        if not self.python.exists():
            raise RuntimeError(
                f"DINOv2 environment not found at {self.env_path}. "
                f"Run: cd {self.env_path} && bash setup.sh"
            )

    def extract_features(self, image_path, mask_path=None, return_coords=False):
        """
        Extract DINOv2 features from an image.

        Args:
            image_path: Path to RGB image
            mask_path: Optional path to binary mask
            return_coords: If True, return (features, coords, info)

        Returns:
            features: np.ndarray of shape [N, 1024]
            coords: np.ndarray of shape [N, 2] (if return_coords=True)
            info: dict with metadata (if return_coords=True)
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            output_path = tmp / 'output.npy'

            # Create request
            request = {
                'image': str(image_path),
                'output': str(output_path)
            }
            if mask_path is not None:
                request['mask'] = str(mask_path)

            request_path = tmp / 'request.json'
            with open(request_path, 'w') as f:
                json.dump(request, f)

            # Call DINOv2 subprocess
            try:
                result = subprocess.run(
                    [str(self.python), str(self.server), str(request_path)],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=120  # 2 minute timeout
                )
            except subprocess.CalledProcessError as e:
                raise RuntimeError(
                    f"DINOv2 inference failed:\n"
                    f"STDOUT: {e.stdout}\n"
                    f"STDERR: {e.stderr}"
                )
            except subprocess.TimeoutExpired:
                raise RuntimeError("DINOv2 inference timed out (>2 minutes)")

            # Parse result
            info = json.loads(result.stdout)

            # Load output
            data = np.load(output_path, allow_pickle=True).item()

        features = data['features']

        if return_coords:
            return features, data['coords'], info
        return features

    def test_connection(self):
        """Test if DINOv2 environment is working."""
        try:
            # Create a small test image
            from PIL import Image
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                img = Image.new('RGB', (224, 224), color=(128, 128, 128))
                img.save(f.name)
                test_img = f.name

            features = self.extract_features(test_img)

            Path(test_img).unlink()  # Clean up

            return features.shape[1] == 1024
        except Exception as e:
            print(f"DINOv2 connection test failed: {e}")
            return False
