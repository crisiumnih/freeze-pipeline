import os
import sys
import numpy as np
import torch
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import subprocess

def extract_dino_features(image_path, mask_path, output_path, dino_script, dino_env):
    """
    Extract DINOv2 features for a masked region using existing script.
    """
    cmd = (
        f"source {dino_env} && "
        f"python {dino_script} {image_path} {mask_path} {output_path}"
    )
    subprocess.run(["bash", "-c", cmd], check=True, capture_output=True)


def compute_feature_similarity(query_features, proposal_features):
    """
    Compute similarity between query object features and proposal features.

    Args:
        query_features: [N, D] array of query object features
        proposal_features: [M, D] array of proposal features

    Returns:
        similarity_score: scalar similarity score
    """
    # Compute mean feature vectors
    query_mean = query_features.mean(axis=0, keepdims=True)  # [1, D]
    proposal_mean = proposal_features.mean(axis=0, keepdims=True)  # [1, D]

    # Cosine similarity
    similarity = cosine_similarity(query_mean, proposal_mean)[0, 0]

    return float(similarity)


def match_proposals_to_query(target_image_path, target_output_folder, query_features_path,
                              dino_script, dino_env, top_k=5):
    """
    Match SAM2 proposals against query object and rank by similarity.

    Args:
        target_image_path: Path to target RGB image
        target_output_folder: Folder with SAM2 segmentation results
        query_features_path: Path to query object visual features (.npy)
        dino_script: Path to DINOv2 inference script
        dino_env: Path to DINOv2 virtual environment
        top_k: Number of top matches to return

    Returns:
        ranked_matches: List of dicts with mask info and similarity scores
    """
    print(f"\n{'='*60}")
    print("MATCHING SAM2 PROPOSALS TO QUERY OBJECT")
    print(f"{'='*60}\n")

    # Load query features
    print(f"Loading query features from: {query_features_path}")
    query_data = np.load(query_features_path, allow_pickle=True).item()
    query_features = query_data['features']  # [N, D]

    # Filter out points with no features
    valid_mask = (query_data['num_views_per_point'] > 0)
    query_features_valid = query_features[valid_mask]

    print(f"Query features shape: {query_features_valid.shape}")
    print(f"Feature dimension: {query_features_valid.shape[1]}")

    # Load SAM2 metadata
    base_name = os.path.splitext(os.path.basename(target_image_path))[0]
    metadata_path = os.path.join(target_output_folder, f"{base_name}_segmentation_metadata.npy")

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"SAM2 metadata not found: {metadata_path}")

    metadata = np.load(metadata_path, allow_pickle=True).item()
    masks_info = metadata['masks']

    print(f"\nFound {len(masks_info)} SAM2 proposals")
    print(f"Target image: {target_image_path}\n")

    # Extract features for each proposal and compute similarity
    results = []

    for i, mask_info in enumerate(masks_info):
        mask_id = mask_info['mask_id']
        mask_path = mask_info['mask_path']
        area = mask_info['area']
        bbox = mask_info['bbox']

        print(f"[Proposal {i+1}/{len(masks_info)}] Processing mask {mask_id}...")
        print(f"  Area: {area}, BBox: {bbox}")

        # Extract DINOv2 features for this proposal
        proposal_features_path = os.path.join(
            target_output_folder,
            f"{base_name}_mask_{mask_id:02d}_features.npy"
        )

        try:
            # Extract features if not already cached
            if not os.path.exists(proposal_features_path):
                print(f"  Extracting DINOv2 features...")
                extract_dino_features(
                    target_image_path,
                    mask_path,
                    proposal_features_path,
                    dino_script,
                    dino_env
                )

            # Load proposal features
            proposal_data = np.load(proposal_features_path, allow_pickle=True).item()
            proposal_features = proposal_data['features']  # [M, D]

            if len(proposal_features) == 0:
                print(f"  Warning: No features extracted (empty mask)")
                similarity = 0.0
            else:
                # Compute similarity
                similarity = compute_feature_similarity(query_features_valid, proposal_features)
                print(f"  Similarity score: {similarity:.4f}")

            results.append({
                'mask_id': mask_id,
                'mask_path': mask_path,
                'crop_path': mask_info['crop_path'],
                'area': area,
                'bbox': bbox,
                'similarity_score': similarity,
                'num_features': len(proposal_features)
            })

        except Exception as e:
            print(f"  Error processing mask {mask_id}: {e}")
            results.append({
                'mask_id': mask_id,
                'mask_path': mask_path,
                'crop_path': mask_info['crop_path'],
                'area': area,
                'bbox': bbox,
                'similarity_score': 0.0,
                'num_features': 0,
                'error': str(e)
            })

    # Sort by similarity score (descending)
    results_sorted = sorted(results, key=lambda x: x['similarity_score'], reverse=True)

    print(f"\n{'='*60}")
    print("RANKING RESULTS")
    print(f"{'='*60}\n")

    print("Top matches by similarity to query object:\n")
    for rank, result in enumerate(results_sorted[:top_k], 1):
        print(f"Rank {rank}:")
        print(f"  Mask ID: {result['mask_id']}")
        print(f"  Similarity: {result['similarity_score']:.4f}")
        print(f"  Area: {result['area']}")
        print(f"  BBox: {result['bbox']}")
        print(f"  Features: {result['num_features']}")
        print()

    # Save ranked results
    ranked_results_path = os.path.join(
        target_output_folder,
        f"{base_name}_ranked_matches.npy"
    )
    np.save(ranked_results_path, {
        'target_image': target_image_path,
        'query_features_path': query_features_path,
        'num_proposals': len(results_sorted),
        'ranked_matches': results_sorted,
        'top_k': top_k
    })

    print(f"Saved ranked results to: {ranked_results_path}")

    # Save top-k matches to 'top' folder
    top_folder = os.path.join(target_output_folder, "top")
    os.makedirs(top_folder, exist_ok=True)

    print(f"\nSaving top {top_k} matches to: {top_folder}")

    import shutil
    for i, match in enumerate(results_sorted[:top_k], 1):
        mask_id = match['mask_id']

        # Copy mask
        if os.path.exists(match['mask_path']):
            dst_mask = os.path.join(top_folder, f"rank_{i:02d}_mask_{mask_id:02d}_sim_{match['similarity_score']:.4f}.png")
            shutil.copy2(match['mask_path'], dst_mask)
            print(f"  Rank {i}: Copied mask (similarity={match['similarity_score']:.4f})")

        # Copy crop
        if os.path.exists(match['crop_path']):
            dst_crop = os.path.join(top_folder, f"rank_{i:02d}_crop_{mask_id:02d}_sim_{match['similarity_score']:.4f}.png")
            shutil.copy2(match['crop_path'], dst_crop)

    # Save top matches metadata
    top_metadata_path = os.path.join(top_folder, "top_matches.npy")
    np.save(top_metadata_path, {
        'top_k': top_k,
        'matches': results_sorted[:top_k]
    })
    print(f"\nSaved top {top_k} matches metadata to: {top_metadata_path}")

    return results_sorted


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python 4_match_proposals.py <target_image> <target_output_folder> <query_features_path> <dino_script> <dino_env> [top_k]")
        print()
        print("Arguments:")
        print("  target_image: Path to target RGB image")
        print("  target_output_folder: Folder containing SAM2 segmentation results")
        print("  query_features_path: Path to query visual features (.npy)")
        print("  dino_script: Path to DINOv2 inference script")
        print("  dino_env: Path to DINOv2 virtual environment activation")
        print("  top_k: (optional) Number of top matches to return, default=5")
        sys.exit(1)

    target_image_path = sys.argv[1]
    target_output_folder = sys.argv[2]
    query_features_path = sys.argv[3]
    dino_script = sys.argv[4]
    dino_env = sys.argv[5]
    top_k = int(sys.argv[6]) if len(sys.argv) > 6 else 5

    if not os.path.exists(target_image_path):
        print(f"Error: Target image not found: {target_image_path}")
        sys.exit(1)

    if not os.path.exists(query_features_path):
        print(f"Error: Query features not found: {query_features_path}")
        sys.exit(1)

    ranked_matches = match_proposals_to_query(
        target_image_path,
        target_output_folder,
        query_features_path,
        dino_script,
        dino_env,
        top_k=top_k
    )

    print(f"\n{'='*60}")
    print("MATCHING COMPLETE")
    print(f"{'='*60}")
    print(f"Top match: Mask {ranked_matches[0]['mask_id']} with similarity {ranked_matches[0]['similarity_score']:.4f}")
