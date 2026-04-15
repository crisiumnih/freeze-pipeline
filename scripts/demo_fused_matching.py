#!/usr/bin/env python3
"""
Demo: End-to-End Fused Matching

Tests complete pipeline with geometric + visual features:
  Query (mesh) → 1056-dim fused features
  Target (scene) → 1056-dim fused features
  Matching → 6-DoF pose

Compares:
  1. Geometric-only (32-dim)
  2. Fully-fused (1056-dim)
"""

import sys
from pathlib import Path
import argparse
import numpy as np
import time

# Add freeze to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'freeze'))

from freeze.models import GeDiClient, DINOv2Client, SAM2Client
from freeze.query import QueryPipeline
from freeze.target import TargetPipeline
from freeze.matching import MatchingPipeline


def print_section(title):
    """Print a section header."""
    print("\n" + "="*70)
    print(title)
    print("="*70)


def print_pose_result(result, label="Pose"):
    """Print pose estimation results."""
    print(f"\n{label}:")
    print(f"  Success: {result.get('success', False)}")
    if result.get('success'):
        print(f"  Inliers: {result.get('num_inliers', 0)}")
        print(f"  RMSE: {result.get('rmse', 0):.4f}m")
        print(f"  Rotation:\n{result['R']}")
        print(f"  Translation: {result['t'].flatten()}")
        if 'icp_iterations' in result:
            print(f"  ICP iterations: {result['icp_iterations']}")
    else:
        print(f"  Reason: {result.get('reason', 'Unknown')}")


def main():
    parser = argparse.ArgumentParser(description='Demo: End-to-End Fused Matching')
    parser.add_argument('--query-mesh', type=str, default=None,
                        help='Path to query mesh (uses HOPE Ketchup if not provided)')
    parser.add_argument('--target-rgb', type=str, default=None,
                        help='Path to target RGB image')
    parser.add_argument('--target-depth', type=str, default=None,
                        help='Path to target depth image')
    parser.add_argument('--use-icp', action='store_true',
                        help='Use ICP refinement')
    parser.add_argument('--num-query-points', type=int, default=5000,
                        help='Number of points to sample from query mesh')
    parser.add_argument('--num-views', type=int, default=4,
                        help='Number of views for rendering (actual views = 2x with elevations)')
    parser.add_argument('--max-proposals', type=int, default=5,
                        help='Max number of target proposals to process')
    args = parser.parse_args()

    print_section("🎯 FreeZe: End-to-End Fused Matching Demo")

    # Default paths
    if args.query_mesh is None:
        args.query_mesh = '/home/sra/Prajwal/freeze_v2/data/hope/meshes/eval/Ketchup.obj'
    if args.target_rgb is None:
        args.target_rgb = '/home/sra/Prajwal/freeze_v2/data/hope/scenes/ketchup_scene_rgb.png'
    if args.target_depth is None:
        args.target_depth = '/home/sra/Prajwal/freeze_v2/data/hope/scenes/ketchup_scene_depth.png'

    query_mesh = Path(args.query_mesh)
    target_rgb = Path(args.target_rgb)
    target_depth = Path(args.target_depth)

    # Verify files exist
    for path, name in [(query_mesh, 'Query mesh'), (target_rgb, 'Target RGB'), (target_depth, 'Target depth')]:
        if not path.exists():
            print(f"❌ {name} not found: {path}")
            print(f"   Please provide valid paths or use default HOPE dataset")
            return 1

    print(f"\nConfiguration:")
    print(f"  Query: {query_mesh.name}")
    print(f"  Target RGB: {target_rgb.name}")
    print(f"  Target Depth: {target_depth.name}")
    print(f"  Query points: {args.num_query_points}")
    print(f"  Render views: {args.num_views} (→ {args.num_views * 2} with elevations)")
    print(f"  Max proposals: {args.max_proposals}")
    print(f"  ICP refinement: {args.use_icp}")

    # ===== STEP 1: Initialize Clients =====
    print_section("[1/5] Initializing Model Clients")

    try:
        print("  Initializing GeDi...", end=' ', flush=True)
        gedi = GeDiClient()
        print("✓")
    except Exception as e:
        print(f"✗\n  Failed to initialize GeDi: {e}")
        return 1

    try:
        print("  Initializing DINOv2...", end=' ', flush=True)
        dinov2 = DINOv2Client()
        print("✓")
    except Exception as e:
        print(f"✗\n  Failed to initialize DINOv2: {e}")
        return 1

    try:
        print("  Initializing SAM2...", end=' ', flush=True)
        sam2 = SAM2Client()
        print("✓")
    except Exception as e:
        print(f"✗\n  Failed to initialize SAM2: {e}")
        return 1

    print("\n  ✅ All model clients initialized")

    # ===== STEP 2: Process Query Object =====
    print_section("[2/5] Processing Query Object")

    # Test 2a: Geometric-only
    print("\n[2a] Geometric-only mode (32-dim)...")
    t0 = time.time()
    query_pipeline_geo = QueryPipeline(
        gedi,
        use_visual=False,
        target_scale=0.2,
        num_points=args.num_query_points
    )
    query_geo = query_pipeline_geo.process(query_mesh)
    t_geo = time.time() - t0
    print(f"  ✓ Query features: {query_geo['features'].shape}")
    print(f"  ✓ Time: {t_geo:.1f}s")

    # Test 2b: Fused
    print("\n[2b] Fused mode (1056-dim = 32 geo + 1024 vis)...")
    t0 = time.time()
    query_pipeline_fused = QueryPipeline(
        gedi,
        dinov2,
        use_visual=True,
        target_scale=0.2,
        num_points=args.num_query_points,
        num_views=args.num_views
    )
    query_fused = query_pipeline_fused.process(query_mesh)
    t_fused = time.time() - t0
    print(f"  ✓ Query features: {query_fused['features'].shape}")
    print(f"  ✓ Geometric: {query_fused['geometric_features'].shape}")
    print(f"  ✓ Visual: {query_fused['visual_features'].shape}")
    print(f"  ✓ Time: {t_fused:.1f}s")

    # ===== STEP 3: Process Target Scene =====
    print_section("[3/5] Processing Target Scene")

    # Test 3a: Geometric-only
    print("\n[3a] Geometric-only mode (32-dim)...")
    t0 = time.time()
    target_pipeline_geo = TargetPipeline(
        sam2,
        gedi,
        use_visual=False,
        max_proposals=args.max_proposals
    )
    target_geo = target_pipeline_geo.process_scene(target_rgb, target_depth)
    t_geo = time.time() - t0
    print(f"  ✓ Proposals: {len(target_geo)}")
    if len(target_geo) > 0:
        print(f"  ✓ Example proposal features: {target_geo[0]['features'].shape}")
    print(f"  ✓ Time: {t_geo:.1f}s")

    # Test 3b: Fused
    print("\n[3b] Fused mode (1056-dim = 32 geo + 1024 vis)...")
    print("  (This will take 2-3 minutes with visual features...)")
    t0 = time.time()
    target_pipeline_fused = TargetPipeline(
        sam2,
        gedi,
        dinov2,
        use_visual=True,
        max_proposals=args.max_proposals
    )
    target_fused = target_pipeline_fused.process_scene(target_rgb, target_depth)
    t_fused = time.time() - t0
    print(f"  ✓ Proposals: {len(target_fused)}")
    if len(target_fused) > 0:
        print(f"  ✓ Example proposal features: {target_fused[0]['features'].shape}")
        print(f"  ✓ Example geometric: {target_fused[0]['geometric_features'].shape}")
        print(f"  ✓ Example visual: {target_fused[0]['visual_features'].shape}")
    print(f"  ✓ Time: {t_fused:.1f}s ({t_fused/60:.1f} min)")

    if len(target_geo) == 0 or len(target_fused) == 0:
        print("\n❌ No proposals generated! Cannot continue with matching.")
        print("   Check segmentation quality or try different scene.")
        return 1

    # ===== STEP 4: Matching - Geometric Only =====
    print_section("[4/5] Pose Estimation - GEOMETRIC-ONLY (32-dim)")

    matcher_geo = MatchingPipeline(
        feature_dim=32,
        use_icp=args.use_icp
    )

    print(f"\nMatching query against {len(target_geo)} proposals...")
    t0 = time.time()
    results_geo = []
    for i, proposal in enumerate(target_geo):
        result = matcher_geo.estimate_pose(query_geo, proposal)
        results_geo.append(result)
        if result['success']:
            print(f"  Proposal {i+1}: ✓ {result['num_inliers']} inliers, RMSE={result['rmse']:.4f}m")
        else:
            print(f"  Proposal {i+1}: ✗ {result.get('reason', 'Failed')}")

    t_match_geo = time.time() - t0

    # Find best match
    best_geo = max(results_geo, key=lambda x: x.get('num_inliers', 0))
    best_geo_idx = results_geo.index(best_geo)

    print(f"\n✅ Best match: Proposal {best_geo_idx + 1}")
    print_pose_result(best_geo, "Geometric-only Result")
    print(f"  Matching time: {t_match_geo:.2f}s")

    # ===== STEP 5: Matching - Fused =====
    print_section("[5/5] Pose Estimation - FUSED (1056-dim)")

    matcher_fused = MatchingPipeline(
        feature_dim=1056,
        use_icp=args.use_icp
    )

    print(f"\nMatching query against {len(target_fused)} proposals...")
    t0 = time.time()
    results_fused = []
    for i, proposal in enumerate(target_fused):
        result = matcher_fused.estimate_pose(query_fused, proposal)
        results_fused.append(result)
        if result['success']:
            print(f"  Proposal {i+1}: ✓ {result['num_inliers']} inliers, RMSE={result['rmse']:.4f}m")
        else:
            print(f"  Proposal {i+1}: ✗ {result.get('reason', 'Failed')}")

    t_match_fused = time.time() - t0

    # Find best match
    best_fused = max(results_fused, key=lambda x: x.get('num_inliers', 0))
    best_fused_idx = results_fused.index(best_fused)

    print(f"\n✅ Best match: Proposal {best_fused_idx + 1}")
    print_pose_result(best_fused, "Fused Result")
    print(f"  Matching time: {t_match_fused:.2f}s")

    # ===== FINAL COMPARISON =====
    print_section("📊 GEOMETRIC vs FUSED COMPARISON")

    print(f"\nGeometric-only (32-dim):")
    print(f"  Inliers: {best_geo.get('num_inliers', 0)}")
    print(f"  RMSE: {best_geo.get('rmse', float('inf')):.4f}m")
    print(f"  Success: {best_geo.get('success', False)}")

    print(f"\nFully-fused (1056-dim):")
    print(f"  Inliers: {best_fused.get('num_inliers', 0)}")
    print(f"  RMSE: {best_fused.get('rmse', float('inf')):.4f}m")
    print(f"  Success: {best_fused.get('success', False)}")

    if best_geo.get('success') and best_fused.get('success'):
        inlier_improvement = best_fused['num_inliers'] - best_geo['num_inliers']
        inlier_pct = (inlier_improvement / best_geo['num_inliers']) * 100 if best_geo['num_inliers'] > 0 else 0

        rmse_improvement = best_geo['rmse'] - best_fused['rmse']
        rmse_pct = (rmse_improvement / best_geo['rmse']) * 100 if best_geo['rmse'] > 0 else 0

        print(f"\n💡 Impact of Visual Features:")
        print(f"  Inliers: {inlier_improvement:+d} ({inlier_pct:+.1f}%)")
        print(f"  RMSE: {rmse_improvement:+.4f}m ({rmse_pct:+.1f}%)")

        if inlier_improvement > 0 or rmse_improvement > 0:
            print(f"\n  🎉 Visual features IMPROVED pose estimation!")
        elif inlier_improvement == 0 and rmse_improvement == 0:
            print(f"\n  ➡️  Visual features had NO CHANGE (geometric was sufficient)")
        else:
            print(f"\n  ⚠️  Visual features DEGRADED pose estimation (geometric was better)")

    print_section("✅ Demo Complete!")

    return 0


if __name__ == '__main__':
    sys.exit(main())
