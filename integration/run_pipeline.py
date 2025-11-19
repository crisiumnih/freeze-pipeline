import subprocess
import os
import numpy as np
import argparse

BASE = "/home/sra/freeze"

# Environment activation commands
GEDI_ENV   = f"{BASE}/gedi_test/gedi/.venv/bin/activate"
SAM2_ENV   = f"{BASE}/sam2/.venv/bin/activate"
DINO_ENV   = f"{BASE}/dino_test/dinov2/.venv/bin/activate"

# Script paths
GEDI_SCRIPT = f"{BASE}/gedi_test/gedi/1_gedi_process_object.py"
DINO_SCRIPT = f"{BASE}/dino_test/dinov2/demo_infer.py"
BACKPROJECT_SCRIPT = f"{BASE}/gedi_test/gedi/2_backproject_features.py"
SAM2_SCRIPT = f"{BASE}/sam2/3_sam2_localization.py"
MATCH_SCRIPT = f"{BASE}/sam2/4_match_proposals.py"

# -------------------------------------------------------------
def run_gedi(object_folder, output_folder):
    """
    Runs GeDi on an object folder to extract geometric features.
    """
    print(f"Running GeDi on: {object_folder}")
    cmd = (
        f"source {GEDI_ENV} && "
        f"python {GEDI_SCRIPT} {object_folder} {output_folder}"
    )
    subprocess.run(["bash", "-c", cmd], check=True)

# -------------------------------------------------------------
def run_dino(image_path, mask_path, output_path):
    """
    Runs DINOv2 on an image with mask to extract dense features.
    """
    cmd = (
        f"source {DINO_ENV} && "
        f"python {DINO_SCRIPT} {image_path} {mask_path} {output_path}"
    )
    subprocess.run(["bash", "-c", cmd], check=True)

# -------------------------------------------------------------
def run_dino_multiview(out_folder, object_name):
    """
    Run DINOv2 on all rendered views with masks.
    Returns list of feature file paths.
    """
    view_names = ["front", "right", "left", "back", "top", "bottom"]
    
    feature_files = []
    
    for i, view_name in enumerate(view_names):
        img_path = os.path.join(out_folder, f"view_{i:02d}_{view_name}.png")
        mask_path = os.path.join(out_folder, f"mask_{i:02d}_{view_name}.png")
        feat_path = os.path.join(out_folder, f"dino_view_{i:02d}_{view_name}.npy")
        
        if not os.path.exists(img_path):
            print(f"Warning: {img_path} not found, skipping...")
            continue
            
        if not os.path.exists(mask_path):
            print(f"Warning: {mask_path} not found, skipping...")
            continue
        
        print(f"\n[DINOv2] Processing view {i+1}/{len(view_names)}: {view_name}")
        run_dino(img_path, mask_path, feat_path)
        feature_files.append(feat_path)
    
    return feature_files

# -------------------------------------------------------------
def run_backprojection(out_folder, object_name):
    """
    Back-project 2D features to 3D points.
    """
    print(f"Back-projecting features for: {object_name}")
    cmd = (
        f"source {GEDI_ENV} && "
        f"python {BACKPROJECT_SCRIPT} {out_folder} {object_name}"
    )
    subprocess.run(["bash", "-c", cmd], check=True)

# -------------------------------------------------------------
def run_sam2_localization(target_image_path, output_folder, top_k=5):
    """
    Run SAM2 for object localization in target image.
    """
    print(f"Running SAM2 localization on: {target_image_path}")
    cmd = (
        f"source {SAM2_ENV} && "
        f"python {SAM2_SCRIPT} {target_image_path} {output_folder} {top_k}"
    )
    subprocess.run(["bash", "-c", cmd], check=True)

# -------------------------------------------------------------
def run_proposal_matching(target_image_path, target_output_folder, query_features_path, top_k=5):
    """
    Match SAM2 proposals against query object features and rank by similarity.
    """
    print(f"Matching proposals to query object...")
    cmd = (
        f"source {SAM2_ENV} && "
        f"python {MATCH_SCRIPT} {target_image_path} {target_output_folder} "
        f"{query_features_path} {DINO_SCRIPT} {DINO_ENV} {top_k}"
    )
    subprocess.run(["bash", "-c", cmd], check=True)

# -------------------------------------------------------------
def process_query_object(query_obj_folder, query_out_folder):
    """
    Stage 1: Process query object (steps 1-3)
    - GeDi preprocessing & descriptor extraction
    - DINOv2 dense feature extraction
    - Back-projection of 2D features to 3D
    """
    os.makedirs(query_out_folder, exist_ok=True)
    query_object_name = os.path.basename(query_obj_folder.rstrip("/"))

    print("\n" + "="*60)
    print("STAGE 1: QUERY OBJECT PROCESSING")
    print("="*60)

    print("\n[STEP 1] Running GeDi preprocessing & descriptor extraction...")
    print("-"*60)
    run_gedi(query_obj_folder, query_out_folder)

    print("\n[STEP 2] Running DINOv2 dense feature extraction on all views...")
    print("-"*60)
    query_feature_files = run_dino_multiview(query_out_folder, query_object_name)

    print("\n[STEP 3] Back-projecting 2D features to 3D points...")
    print("-"*60)
    run_backprojection(query_out_folder, query_object_name)

    return query_object_name, query_feature_files

# -------------------------------------------------------------
def process_target_object(target_image_path, target_out_folder, query_features_path=None, top_k=5):
    """
    Stage 2: Process target object (step 4+)
    - SAM2 segmentation for object localization
    - Match proposals against query object (if query_features_path provided)
    """
    os.makedirs(target_out_folder, exist_ok=True)

    print("\n" + "="*60)
    print("STAGE 2: TARGET OBJECT PROCESSING")
    print("="*60)

    if not os.path.exists(target_image_path):
        print(f"\nError: Target image not found: {target_image_path}")
        print("Please provide a valid target image path.")
        return None

    print(f"\n[STEP 4] Running SAM2 localization on target image...")
    print(f"  Input: {target_image_path}")
    print("-"*60)
    run_sam2_localization(target_image_path, target_out_folder, top_k=top_k)

    # Check SAM2 results
    metadata_path = os.path.join(target_out_folder,
                                 os.path.splitext(os.path.basename(target_image_path))[0] +
                                 "_segmentation_metadata.npy")
    if os.path.exists(metadata_path):
        metadata = np.load(metadata_path, allow_pickle=True).item()
        print(f"\nGenerated {metadata['num_masks']} candidate masks")
        print("Top candidates by area:")
        for mask_info in metadata['masks'][:3]:
            print(f"  - Mask {mask_info['mask_id']}: area={mask_info['area']}, "
                  f"bbox={mask_info['bbox']}")

    # Match proposals against query object
    if query_features_path and os.path.exists(query_features_path):
        print(f"\n[STEP 5] Matching proposals against query object...")
        print("-"*60)
        run_proposal_matching(target_image_path, target_out_folder, query_features_path, top_k=top_k)

        # Load and display ranked results
        base_name = os.path.splitext(os.path.basename(target_image_path))[0]
        ranked_results_path = os.path.join(target_out_folder, f"{base_name}_ranked_matches.npy")

        if os.path.exists(ranked_results_path):
            ranked_data = np.load(ranked_results_path, allow_pickle=True).item()
            ranked_matches = ranked_data['ranked_matches']

            print(f"\nTop {min(3, len(ranked_matches))} matches by similarity to query:")
            for i, match in enumerate(ranked_matches[:3], 1):
                print(f"  {i}. Mask {match['mask_id']}: "
                      f"similarity={match['similarity_score']:.4f}, "
                      f"area={match['area']}, "
                      f"bbox={match['bbox']}")

            return ranked_data
    else:
        if query_features_path:
            print(f"\nWarning: Query features not found: {query_features_path}")
            print("Skipping proposal matching step.")

        return metadata if os.path.exists(metadata_path) else None

    return None

# -------------------------------------------------------------
def print_summary(query_out_folder=None, query_object_name=None, query_feature_files=None,
                  target_out_folder=None, target_image_path=None):
    """Print pipeline execution summary"""
    print("\n" + "="*60)
    print("PIPELINE SUMMARY")
    print("="*60)

    if query_out_folder and query_object_name:
        print("\nQuery Object:")
        if query_feature_files:
            print(f"  - Processed {len(query_feature_files)} views successfully")

        visual_feat_path = os.path.join(query_out_folder, f"{query_object_name}_visual_features.npy")
        if os.path.exists(visual_feat_path):
            visual_data = np.load(visual_feat_path, allow_pickle=True).item()
            print(f"  - Visual features: {visual_data['features'].shape}")
            print(f"  - Points with features: {(visual_data['num_views_per_point'] > 0).sum()}")

        gedi_feat_path = os.path.join(query_out_folder, f"{query_object_name}_gedi.npy")
        if os.path.exists(gedi_feat_path):
            gedi_data = np.load(gedi_feat_path)
            print(f"  - GeDi features: {gedi_data.shape}")

        print(f"  - Output location: {query_out_folder}")

    if target_out_folder and target_image_path and os.path.exists(target_image_path):
        print("\nTarget Object:")
        metadata_path = os.path.join(target_out_folder,
                                     os.path.splitext(os.path.basename(target_image_path))[0] +
                                     "_segmentation_metadata.npy")
        if os.path.exists(metadata_path):
            metadata = np.load(metadata_path, allow_pickle=True).item()
            print(f"  - Generated {metadata['num_masks']} candidate masks")
            print(f"  - Output location: {target_out_folder}")

    print("\n" + "="*60)
    print("Pipeline complete!")
    print("="*60)

# -------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the object localization pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run both stages (query + target)
  python run_pipeline.py --stage both

  # Run only query object processing (stage 1)
  python run_pipeline.py --stage query

  # Run only target object processing (stage 2)
  python run_pipeline.py --stage target --target-image /path/to/image.jpg

  # Specify custom paths
  python run_pipeline.py --stage both --query-folder /path/to/query --target-image /path/to/target.jpg
        """
    )

    parser.add_argument(
        "--stage",
        choices=["query", "target", "both"],
        default="both",
        help="Which stage(s) to run: 'query' (steps 1-3), 'target' (step 4+), or 'both' (default: both)"
    )

    parser.add_argument(
        "--query-folder",
        #default=f"{BASE}/data/20objects/data/Kinfu_Samurai1_light",
        default=f"{BASE}/data/hope/objects/Pineapple",
        help="Path to query object folder (default: %(default)s)"
    )

    parser.add_argument(
        "--query-output",
        default=f"{BASE}/processed/query2",
        help="Output folder for query processing results (default: %(default)s)"
    )

    parser.add_argument(
        "--target-image",
        default=f"{BASE}/data/target_scene_2.jpg",
        help="Path to target image (default: %(default)s)"
    )

    parser.add_argument(
        "--target-output",
        default=f"{BASE}/processed/target_2",
        help="Output folder for target processing results (default: %(default)s)"
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top SAM2 masks to keep (default: 5)"
    )

    args = parser.parse_args()

    # Track results for summary
    query_object_name = None
    query_feature_files = None

    # Stage 1: Query object processing
    if args.stage in ["query", "both"]:
        query_object_name, query_feature_files = process_query_object(
            args.query_folder,
            args.query_output
        )

    # Stage 2: Target object processing
    if args.stage in ["target", "both"]:
        # Determine query object name and features path
        if query_object_name is None:
            query_object_name = os.path.basename(args.query_folder.rstrip("/"))

        query_features_path = os.path.join(args.query_output, f"{query_object_name}_visual_features.npy")

        process_target_object(
            args.target_image,
            args.target_output,
            query_features_path=query_features_path,
            top_k=args.top_k
        )

    # Print summary
    print_summary(
        query_out_folder=args.query_output if args.stage in ["query", "both"] else None,
        query_object_name=query_object_name,
        query_feature_files=query_feature_files,
        target_out_folder=args.target_output if args.stage in ["target", "both"] else None,
        target_image_path=args.target_image if args.stage in ["target", "both"] else None
    )