#!/usr/bin/env python3
"""
Visual Demo: End-to-End FreeZe Pipeline with GIF Output

Creates a visual story of the complete pipeline:
1. Query object (mesh)
2. Query processing (renders, features)
3. Target scene (RGB + depth)
4. Segmentation (SAM2 proposals)
5. Pose estimation results
6. Geometric vs Fused comparison

Outputs:
- Individual frame images at each stage
- Combined GIF showing the full pipeline
"""

import sys
from pathlib import Path
import argparse
import time
import json
import numpy as np

# Add freeze to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'freeze'))

from freeze.models import GeDiClient, DINOv2Client, SAM2Client
from freeze.query import QueryPipeline, MultiViewRenderer
from freeze.target import TargetPipeline
from freeze.matching import MatchingPipeline
from freeze.visualization import PipelineVisualizer


def main():
    parser = argparse.ArgumentParser(description='Visual Demo: FreeZe Pipeline with GIF')
    parser.add_argument('--query-mesh', type=str, default=None,
                        help='Path to query mesh (uses HOPE Ketchup if not provided)')
    parser.add_argument('--target-rgb', type=str, default=None,
                        help='Path to target RGB image')
    parser.add_argument('--target-depth', type=str, default=None,
                        help='Path to target depth image')
    parser.add_argument('--output-dir', type=str,
                        default='/home/sra/Prajwal/freeze_v2/output/demo_visual',
                        help='Directory for output images and GIF')
    parser.add_argument('--use-icp', action='store_true',
                        help='Use ICP refinement')
    parser.add_argument('--num-query-points', type=int, default=5000,
                        help='Number of points to sample from query mesh')
    parser.add_argument('--num-views', type=int, default=4,
                        help='Number of views for rendering')
    parser.add_argument('--max-proposals', type=int, default=20,
                        help='Max number of target proposals')
    args = parser.parse_args()

    print("="*70)
    print("🎬 FreeZe Visual Pipeline Demo")
    print("="*70)

    # Default paths
    if args.query_mesh is None:
        args.query_mesh = '/home/sra/Prajwal/freeze_v2/data/hope/meshes/eval/Mustard.obj'
    if args.target_rgb is None:
        args.target_rgb = '/home/sra/Prajwal/freeze_v2/data/hope/hope_image/valid/scene_0000/0000_rgb.jpg'
    if args.target_depth is None:
        args.target_depth = '/home/sra/Prajwal/freeze_v2/data/hope/hope_image/valid/scene_0000/0000_depth.png'

    query_mesh = Path(args.query_mesh)
    target_rgb = Path(args.target_rgb)
    target_depth = Path(args.target_depth)
    output_dir = Path(args.output_dir)

    # Verify files
    for path, name in [(query_mesh, 'Query'), (target_rgb, 'RGB'), (target_depth, 'Depth')]:
        if not path.exists():
            print(f"❌ {name} not found: {path}")
            return 1

    print(f"\n📁 Inputs:")
    print(f"  Query: {query_mesh.name}")
    print(f"  Scene: {target_rgb.name}")
    print(f"  Output: {output_dir}")

    # Initialize visualizer
    viz = PipelineVisualizer(output_dir)

    # ===== FRAME 1: Title =====
    print("\n[Frame 1] Creating title...")
    viz.add_title_frame(
        "FreeZe: Pose Estimation Pipeline",
        "Geometric + Visual Feature Matching",
        duration=3.5
    )

    # ===== FRAME 2: Query Object =====
    print("[Frame 2] Visualizing query object...")
    viz.visualize_query_mesh(query_mesh, duration=3.5)

    # ===== Initialize Models =====
    print("\n🔧 Initializing models...")
    try:
        gedi = GeDiClient()
        dinov2 = DINOv2Client()
        sam2 = SAM2Client()
        print("  ✓ All models initialized")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return 1

    # ===== FRAME 3: Query Rendering =====
    print("\n[Frame 3] Rendering query views...")
    renderer = MultiViewRenderer(image_size=(640, 480))
    renders = renderer.render(query_mesh, num_views=args.num_views, elevation_angles=[0, 30])
    viz.visualize_query_renders(renders, duration=4.0)

    # ===== Process Query (both modes) =====
    print("\n🔍 Processing query object...")

    # Geometric-only
    print("  [Geometric] 32-dim features...")
    t0 = time.time()
    query_pipeline_geo = QueryPipeline(
        gedi,
        use_visual=False,
        target_scale=0.2,
        num_points=args.num_query_points
    )
    query_geo = query_pipeline_geo.process(query_mesh)
    print(f"    ✓ {query_geo['features'].shape} in {time.time()-t0:.1f}s")

    # Fused
    print("  [Fused] 1056-dim features...")
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
    print(f"    ✓ {query_fused['features'].shape} in {time.time()-t0:.1f}s")

    # Load camera intrinsics from scene metadata (before TargetPipeline init)
    scene_json = target_rgb.parent / f"{target_rgb.stem.replace('_rgb', '')}.json"
    camera_intrinsics = None
    cam_intr_tuple = None
    if scene_json.exists():
        with open(scene_json, 'r') as f:
            scene_data = json.load(f)
        camera_intrinsics = np.array(scene_data['camera']['intrinsics'])
        cam_intr_tuple = (camera_intrinsics[0,0], camera_intrinsics[1,1],
                          camera_intrinsics[0,2], camera_intrinsics[1,2])
        print(f"\n📷 Camera intrinsics: fx={cam_intr_tuple[0]:.1f}, fy={cam_intr_tuple[1]:.1f}")

    # ===== FRAME 4: Target Scene =====
    print("\n[Frame 4] Visualizing target scene...")
    viz.visualize_target_scene(target_rgb, target_depth, duration=3.5)

    # ===== Process Target (for segmentation viz) =====
    print("\n🎯 Processing target scene...")

    # First, get raw segmentation for visualization
    print("  [Segmentation] SAM2...")
    t0 = time.time()
    from PIL import Image as PILImage
    rgb_image = np.array(PILImage.open(target_rgb))
    depth_image = np.array(PILImage.open(target_depth))

    # Get raw segmentation masks
    sam2_temp_dir = Path(args.output_dir) / "sam2_temp"
    metadata = sam2.generate_masks(target_rgb, sam2_temp_dir, top_k=args.max_proposals)
    
    # Load masks for visualization
    raw_masks = []
    for mask_info in metadata['masks']:
        mask_img = PILImage.open(mask_info['mask_path'])
        mask = np.array(mask_img) > 0  # Convert to boolean
        bbox_dict = mask_info['bbox']
        bbox = [bbox_dict['x'], bbox_dict['y'], bbox_dict['width'], bbox_dict['height']]
        raw_masks.append({'segmentation': mask, 'bbox': bbox})
    
    print(f"    ✓ {len(raw_masks)} raw masks in {time.time()-t0:.1f}s")

    # ===== FRAME 5: Segmentation =====
    print("[Frame 5] Visualizing segmentation...")
    viz.visualize_segmentation(target_rgb, raw_masks[:args.max_proposals], duration=4.0)

    # Now process with pipeline
    print("  [Processing] Geometric features...")
    t0 = time.time()
    target_pipeline_geo = TargetPipeline(
        sam2,
        gedi,
        use_visual=False,
        camera_intrinsics=cam_intr_tuple,
        max_proposals=args.max_proposals
    )
    target_geo = target_pipeline_geo.process_scene(target_rgb, target_depth)
    target_geo_proposals = target_geo['proposals']
    print(f"    ✓ {len(target_geo_proposals)} proposals in {time.time()-t0:.1f}s")

    # Now get fused features
    print("  [Fused] With visual features...")
    print("    (This takes 2-3 minutes...)")
    t0 = time.time()
    target_pipeline_fused = TargetPipeline(
        sam2,
        gedi,
        dinov2,
        use_visual=True,
        camera_intrinsics=cam_intr_tuple,
        max_proposals=args.max_proposals
    )
    target_fused = target_pipeline_fused.process_scene(target_rgb, target_depth)
    target_fused_proposals = target_fused['proposals']
    print(f"    ✓ {len(target_fused_proposals)} proposals in {time.time()-t0:.1f}s")

    if len(target_geo_proposals) == 0:
        print("\n❌ No proposals! Cannot continue.")
        return 1

    # ===== Matching: Geometric =====
    print("\n🔗 Matching: Geometric-only...")
    matcher_geo = MatchingPipeline(
        use_icp=args.use_icp,
        ransac_inlier_threshold=0.005,  # 5mm - tighter to reduce false positives
        min_inliers=10,
        min_correspondences=10
    )

    results_geo = []
    for i, proposal in enumerate(target_geo_proposals):
        result = matcher_geo.estimate_pose_for_proposal(query_geo, proposal, feature_type='geometric')
        results_geo.append((i, result))
        if result and result.get('success'):
            print(f"  Proposal {i+1}: ✓ {result['num_inliers']} inliers")

    # Filter: only keep poses with z in 0.3–1.5m (reject background/noise)
    valid_results_geo = [(i, r) for i, r in results_geo
                         if r is not None and 0.3 < r['t'].flatten()[2] < 1.5]
    if not valid_results_geo:
        print("  ⚠️  No valid geometric matches")
        best_geo = None
        best_geo_idx = -1
    else:
        best_geo_idx, best_geo = max(valid_results_geo, key=lambda x: x[1].get('num_inliers', 0))
        t_geo = best_geo['t'].flatten()
        print(f"  → Best: Proposal {best_geo_idx+1}, {best_geo.get('num_inliers', 0)} inliers, t={t_geo}")

    # ===== Matching: Fused =====
    print("\n🔗 Matching: Fused...")
    matcher_fused = MatchingPipeline(
        use_icp=args.use_icp,
        ransac_inlier_threshold=0.005,
        min_inliers=10,
        min_correspondences=10
    )

    results_fused = []
    for i, proposal in enumerate(target_fused_proposals):
        result = matcher_fused.estimate_pose_for_proposal(query_fused, proposal, feature_type='fused')
        results_fused.append((i, result))
        if result and result.get('success'):
            print(f"  Proposal {i+1}: ✓ {result['num_inliers']} inliers")

    valid_results_fused = [(i, r) for i, r in results_fused
                           if r is not None and 0.3 < r['t'].flatten()[2] < 1.5]
    if not valid_results_fused:
        print("  ⚠️  No valid fused matches")
        best_fused = None
        best_fused_idx = -1
    else:
        best_fused_idx, best_fused = max(valid_results_fused, key=lambda x: x[1].get('num_inliers', 0))
        t_fused = best_fused['t'].flatten()
        print(f"  → Best: Proposal {best_fused_idx+1}, {best_fused.get('num_inliers', 0)} inliers, t={t_fused}")

    # ===== FRAME 6: Geometric Result =====
    if best_geo is not None:
        print("\n[Frame 6] Visualizing geometric result...")
        viz.visualize_pose_result(
            target_rgb,
            query_mesh,
            best_geo,
            best_geo_idx,
            camera_intrinsics=camera_intrinsics,
            duration=4.5
        )
    else:
        print("\n[Frame 6] Skipping geometric result (no matches)...")

    # ===== FRAME 7: Fused Result =====
    if best_fused is not None:
        print("[Frame 7] Visualizing fused result...")
        viz.visualize_pose_result(
            target_rgb,
            query_mesh,
            best_fused,
            best_fused_idx,
            camera_intrinsics=camera_intrinsics,
            duration=4.5
        )
    else:
        print("[Frame 7] Skipping fused result (no matches)...")

    # ===== FRAME 8: Comparison =====
    if best_geo is not None or best_fused is not None:
        print("[Frame 8] Creating comparison...")
        viz.visualize_comparison(best_geo, best_fused, duration=5.0)
    else:
        print("[Frame 8] Skipping comparison (no matches)...")

    # ===== FRAME 9: Final Summary =====
    print("[Frame 9] Creating summary...")
    if best_geo and best_fused and best_geo.get('success') and best_fused.get('success'):
        inlier_diff = best_fused['num_inliers'] - best_geo['num_inliers']
        summary = f"Visual Features: {inlier_diff:+d} inliers"
    else:
        summary = "See comparison for details"

    viz.add_title_frame(
        "Pipeline Complete! ✓",
        summary,
        duration=3.5
    )

    # ===== Create GIF =====
    print("\n🎬 Creating GIF...")
    gif_path = viz.create_gif(Path(args.output_dir) / 'freeze_pipeline.gif')

    # ===== Summary =====
    print("\n" + "="*70)
    print("✅ DEMO COMPLETE!")
    print("="*70)
    print(f"\n📊 Results:")
    if best_geo:
        print(f"  Geometric: {best_geo.get('num_inliers', 0)} inliers, "
              f"RMSE={best_geo.get('rmse', float('inf')):.4f}m")
    else:
        print(f"  Geometric: No valid matches")
    
    if best_fused:
        print(f"  Fused: {best_fused.get('num_inliers', 0)} inliers, "
              f"RMSE={best_fused.get('rmse', float('inf')):.4f}m")
    else:
        print(f"  Fused: No valid matches")

    print(f"\n📁 Outputs:")
    print(f"  Images: {args.output_dir}/")
    print(f"  GIF: {gif_path}")

    print("\n🎉 Open the GIF to see the complete pipeline!")

    return 0


if __name__ == '__main__':
    sys.exit(main())
