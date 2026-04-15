"""
FreeZe: Training-free zero-shot 6D pose estimation
"""

__version__ = "0.1.0"

from .models import GeDiClient, DINOv2Client, SAM2Client

__all__ = ['GeDiClient', 'DINOv2Client', 'SAM2Client']
