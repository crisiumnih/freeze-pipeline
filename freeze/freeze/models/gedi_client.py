"""
GeDi Client - Clean Python API for GeDi geometric feature extraction
"""
import subprocess
import json
import tempfile
import numpy as np
from pathlib import Path


class GeDiClient:
    """Client for GeDi geometric feature extraction."""

    def __init__(self, env_path=None):
        """
        Initialize GeDi client.

        Args:
            env_path: Path to GeDi environment (auto-detected if None)
        """
        if env_path is None:
            # Auto-detect environment path
            env_path = Path(__file__).parent.parent.parent.parent / 'environments/gedi'

        self.env_path = Path(env_path)
        self.python = self.env_path / '.venv/bin/python'
        self.server = self.env_path / 'server.py'

        # Verify environment exists
        if not self.python.exists():
            raise RuntimeError(
                f"GeDi environment not found at {self.env_path}. "
                f"Run: cd {self.env_path} && bash setup.sh"
            )

    def compute_features(self, point_cloud, return_info=False):
        """
        Compute GeDi geometric features for a point cloud.

        Args:
            point_cloud: np.ndarray of shape [N, 3] or [N, 6] or str (file path)
            return_info: If True, return (features, info_dict)

        Returns:
            features: np.ndarray of shape [N, 32]
            info: dict with metadata (if return_info=True)
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            # Handle input
            if isinstance(point_cloud, (str, Path)):
                input_path = Path(point_cloud)
            else:
                input_path = tmp / 'input.npy'
                np.save(input_path, point_cloud)

            output_path = tmp / 'output.npy'

            # Create request
            request = {
                'input': str(input_path),
                'output': str(output_path)
            }
            request_path = tmp / 'request.json'
            with open(request_path, 'w') as f:
                json.dump(request, f)

            # Call GeDi subprocess
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
                    f"GeDi inference failed:\n"
                    f"STDOUT: {e.stdout}\n"
                    f"STDERR: {e.stderr}"
                )
            except subprocess.TimeoutExpired:
                raise RuntimeError("GeDi inference timed out (>5 minutes)")

            # Parse result
            info = json.loads(result.stdout)

            # Load output
            features = np.load(output_path)

        if return_info:
            return features, info
        return features

    def test_connection(self):
        """Test if GeDi environment is working."""
        try:
            # Test with a small point cloud
            test_pc = np.random.randn(10, 3).astype(np.float32)
            features = self.compute_features(test_pc)
            return features.shape == (10, 32)
        except Exception as e:
            print(f"GeDi connection test failed: {e}")
            return False
