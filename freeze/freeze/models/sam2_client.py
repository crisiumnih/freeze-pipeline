"""
SAM2 Client - Clean Python API for SAM2 segmentation
"""
import subprocess
import json
import tempfile
import numpy as np
from pathlib import Path


class SAM2Client:
    """Client for SAM2 automatic mask generation."""

    def __init__(self, env_path=None):
        """
        Initialize SAM2 client.

        Args:
            env_path: Path to SAM2 environment (auto-detected if None)
        """
        if env_path is None:
            env_path = Path(__file__).parent.parent.parent.parent / 'environments/sam2'

        self.env_path = Path(env_path)
        self.python = self.env_path / '.venv/bin/python'
        self.server = self.env_path / 'server.py'

        if not self.python.exists():
            raise RuntimeError(
                f"SAM2 environment not found at {self.env_path}. "
                f"Run: cd {self.env_path} && bash setup.sh"
            )

    def generate_masks(self, image_path, output_dir, top_k=None):
        """
        Generate segmentation masks for an image.

        Args:
            image_path: Path to RGB image
            output_dir: Directory to save masks and crops
            top_k: Optional, keep only top K masks by area

        Returns:
            metadata: dict with mask information
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            # Create request
            request = {
                'image': str(image_path),
                'output_dir': str(output_dir)
            }
            if top_k is not None:
                request['top_k'] = int(top_k)

            request_path = tmp / 'request.json'
            with open(request_path, 'w') as f:
                json.dump(request, f)

            # Call SAM2 subprocess
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
                    f"SAM2 inference failed:\n"
                    f"STDOUT: {e.stdout}\n"
                    f"STDERR: {e.stderr}"
                )
            except subprocess.TimeoutExpired:
                raise RuntimeError("SAM2 inference timed out (>5 minutes)")

            # Parse result
            info = json.loads(result.stdout)

        # Load metadata
        metadata_path = info['metadata_path']
        metadata = np.load(metadata_path, allow_pickle=True).item()

        return metadata

    def test_connection(self):
        """Test if SAM2 environment is working."""
        try:
            # Create a small test image
            from PIL import Image
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp = Path(tmpdir)
                test_img = tmp / 'test.png'
                img = Image.new('RGB', (480, 640), color=(128, 128, 128))
                img.save(test_img)

                output_dir = tmp / 'output'
                metadata = self.generate_masks(test_img, output_dir)

                return metadata['num_masks'] >= 0
        except Exception as e:
            print(f"SAM2 connection test failed: {e}")
            return False
