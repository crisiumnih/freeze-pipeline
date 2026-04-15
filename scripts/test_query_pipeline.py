#!/usr/bin/env python3
"""
Test QueryPipeline - Complete query processing pipeline.
Tests both geometric-only and fused feature modes.
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / 'freeze'))

from freeze.models import GeDiClient, DINOv2Client
from freeze.query import QueryPipeline


def main():
    print("="*70)
    print("QUERY PIPELINE TEST")
    print("="*70)

    # Test query
    query_mesh = Path('/home/sra/Prajwal/freeze_v2/data/hope/meshes/eval/Ketchup.obj')

    if not query_mesh.exists():
        print(f"✗ Mesh not found: {query_mesh}")
        return 1

    # Initialize clients
    print("\n[1/4] Initializing clients...")
    gedi = GeDiClient()
    dinov2 = DINOv2Client()
    print("  ✓ GeDi + DINOv2 initialized")

    # ===== TEST 1: Geometric Only =====
    print("\n[2/4] Testing GEOMETRIC-only pipeline...")
    pipeline_geo = QueryPipeline(
        gedi,
        use_visual=False,
        target_scale=0.2,
        num_points=5000
    )

    result_geo = pipeline_geo.process(query_mesh)

    print(f"\n  Results:")
    print(f"    Points: {result_geo['points'].shape}")
    print(f"    Features: {result_geo['features'].shape}")
    print(f"    Feature type: {result_geo['feature_type']}")
    print(f"    Normalization scale: {result_geo['normalization']['target_scale']:.3f}m")

    # Validate geometric
    assert result_geo['features'].shape == (5000, 32), "Wrong geometric feature shape"
    assert result_geo['feature_type'] == 'geometric', "Wrong feature type"
    assert 'geometric_features' in result_geo, "Missing geometric_features"
    assert 'visual_features' not in result_geo, "Should not have visual features"

    print(f"\n  ✓ Geometric-only pipeline PASSED")

    # ===== TEST 2: Fused Features =====
    print("\n[3/4] Testing FUSED (geometric + visual) pipeline...")
    print("  (Using 5000 points, 4 views for faster testing)")
    pipeline_fused = QueryPipeline(
        gedi,
        dinov2,
        use_visual=True,
        target_scale=0.2,
        num_points=5000,
        num_views=4
    )

    result_fused = pipeline_fused.process(query_mesh)

    print(f"\n  Results:")
    print(f"    Points: {result_fused['points'].shape}")
    print(f"    Features: {result_fused['features'].shape}")
    print(f"    - Geometric: {result_fused['geometric_features'].shape}")
    print(f"    - Visual: {result_fused['visual_features'].shape}")
    print(f"    Feature type: {result_fused['feature_type']}")

    # Validate fused
    assert result_fused['features'].shape == (5000, 1056), "Wrong fused feature shape"
    assert result_fused['feature_type'] == 'fused', "Wrong feature type"
    assert 'geometric_features' in result_fused, "Missing geometric_features"
    assert 'visual_features' in result_fused, "Missing visual_features"
    assert result_fused['geometric_features'].shape == (5000, 32), "Wrong geo shape"
    assert result_fused['visual_features'].shape == (5000, 1024), "Wrong vis shape"

    # Check no NaN
    assert not np.any(np.isnan(result_fused['features'])), "Features contain NaN"

    print(f"\n  ✓ Fused pipeline PASSED")

    # ===== TEST 3: Different Scales =====
    print("\n[4/4] Testing different scales...")
    pipeline_small = QueryPipeline(gedi, use_visual=False, target_scale=0.1)
    result_small = pipeline_small.process(query_mesh)

    print(f"    Scale 10cm: {result_small['normalization']['target_scale']:.3f}m")
    assert result_small['normalization']['target_scale'] == 0.1

    pipeline_large = QueryPipeline(gedi, use_visual=False, target_scale=0.5)
    result_large = pipeline_large.process(query_mesh)

    print(f"    Scale 50cm: {result_large['normalization']['target_scale']:.3f}m")
    assert result_large['normalization']['target_scale'] == 0.5

    print(f"\n  ✓ Scale variation PASSED")

    # ===== SUMMARY =====
    print("\n" + "="*70)
    print("ALL TESTS PASSED! ✓✓✓")
    print("="*70)

    print("\nQuery Pipeline Ready:")
    print("  ✓ Geometric-only mode (32-dim)")
    print("  ✓ Fused mode (1056-dim)")
    print("  ✓ Mesh normalization working")
    print("  ✓ Multiple scales supported")

    print("\n" + "="*70)
    print("QUERY PIPELINE COMPLETE! 🎉")
    print("="*70)
    print("\nNow ready for full fused matching:")
    print("  Query: 1056-dim (32 geo + 1024 vis)")
    print("  Target: 1056-dim (32 geo + 1024 vis)")
    print("  → Full geometric + visual matching!")

    return 0


if __name__ == '__main__':
    sys.exit(main())
