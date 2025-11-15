import subprocess
import os
import numpy as np

BASE = "/home/sra/freeze"

GEDI_ENV   = f"{BASE}/gedi_test/gedi/.venv/bin/activate"
GEDI_SCRIPT = f"{BASE}/gedi_test/gedi/1_gedi_process_object.py"

DINO_ENV   = f"{BASE}/dino_test/dinov2/.venv/bin/activate"
DINO_SCRIPT = f"{BASE}/dino_test/dinov2/demo_infer.py"

BACKPROJECT_SCRIPT = f"{BASE}/gedi_test/gedi/2_backproject_features.py"  

# -------------------------------------------------------------
def run_gedi(object_folder, output_folder):
    """
    Runs GeDi on an object folder:
      object_folder = "data/20objects/data/Kinfu_Samurai1_light"
    """
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
    cmd = (
        f"source {GEDI_ENV} && "
        f"python {BACKPROJECT_SCRIPT} {out_folder} {object_name}"
    )
    subprocess.run(["bash", "-c", cmd], check=True)

# -------------------------------------------------------------
if __name__ == "__main__":

    # geometric data folder for one object
    obj_folder = f"{BASE}/data/20objects/data/Kinfu_Samurai1_light"

    # where to store output
    out_folder = f"{BASE}/processed"
    os.makedirs(out_folder, exist_ok=True)

    object_name = os.path.basename(obj_folder.rstrip("/"))

    print("="*60)
    print("STEP 1: Running GeDi preprocessing & descriptor extraction...")
    print("="*60)
    run_gedi(obj_folder, out_folder)

    print("\n" + "="*60)
    print("STEP 2: Running DINOv2 dense feature extraction on all views...")
    print("="*60)
    feature_files = run_dino_multiview(out_folder, object_name)

    print("\n" + "="*60)
    print("STEP 3: Back-projecting 2D features to 3D points...")
    print("="*60)
    run_backprojection(out_folder, object_name)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Processed {len(feature_files)} views successfully:")
    for f in feature_files:
        print(f"  - {f}")
    
    # Check if visual features were created
    visual_feat_path = os.path.join(out_folder, f"{object_name}_visual_features.npy")
    if os.path.exists(visual_feat_path):
        visual_data = np.load(visual_feat_path, allow_pickle=True).item()
        print(f"\nVisual features: {visual_feat_path}")
        print(f"  - Feature shape: {visual_data['features'].shape}")
        print(f"  - Points with features: {(visual_data['num_views_per_point'] > 0).sum()}")
    
    print("\n" + "="*60)
    print("Pipeline complete!")
    print("="*60)
    print(f"\nOutputs in: {out_folder}")
    print(f"  - Point cloud: {object_name}_pc.ply")
    print(f"  - GeDi features: {object_name}_gedi.npy")
    print(f"  - Visual features: {object_name}_visual_features.npy")