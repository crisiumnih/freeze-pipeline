#!/usr/bin/env python3
"""
Test model client wrappers
"""
import sys
from pathlib import Path

# Add freeze package to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'freeze'))

from freeze.models import GeDiClient, DINOv2Client, SAM2Client

def test_gedi():
    """Test GeDi client."""
    print("\n" + "="*60)
    print("Testing GeDi Client")
    print("="*60)

    try:
        client = GeDiClient()
        print("✓ GeDi client initialized")

        # Test connection
        if client.test_connection():
            print("✓ GeDi connection test passed")
            return True
        else:
            print("✗ GeDi connection test failed")
            return False

    except Exception as e:
        print(f"✗ GeDi client error: {e}")
        return False


def test_dinov2():
    """Test DINOv2 client."""
    print("\n" + "="*60)
    print("Testing DINOv2 Client")
    print("="*60)

    try:
        client = DINOv2Client()
        print("✓ DINOv2 client initialized")

        # Test connection
        if client.test_connection():
            print("✓ DINOv2 connection test passed")
            return True
        else:
            print("✗ DINOv2 connection test failed")
            return False

    except Exception as e:
        print(f"✗ DINOv2 client error: {e}")
        return False


def test_sam2():
    """Test SAM2 client."""
    print("\n" + "="*60)
    print("Testing SAM2 Client")
    print("="*60)

    try:
        client = SAM2Client()
        print("✓ SAM2 client initialized")

        # Test connection
        if client.test_connection():
            print("✓ SAM2 connection test passed")
            return True
        else:
            print("✗ SAM2 connection test failed")
            return False

    except Exception as e:
        print(f"✗ SAM2 client error: {e}")
        return False


def main():
    """Run all client tests."""
    print("="*60)
    print("FreeZe Model Client Tests")
    print("="*60)

    results = {
        'GeDi': test_gedi(),
        'DINOv2': test_dinov2(),
        'SAM2': test_sam2()
    }

    print("\n" + "="*60)
    print("Test Results Summary")
    print("="*60)

    for model, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{model:12s} {status}")

    all_passed = all(results.values())

    print("="*60)
    if all_passed:
        print("✓ All client tests passed!")
        return 0
    else:
        print("✗ Some client tests failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
