import os
import sys
import numpy as np
import torch
from PIL import Image
import cv2

def run_sam2_segmentation(image_path, output_dir, top_k=5):
    """
    Run SAM2 automatic mask generation on target image.
    
    Args:
        image_path: Path to input RGB image
        output_dir: Directory to save segmentation results
        top_k: Number of top masks to keep (by area)
    
    Returns:
        List of mask file paths
    """
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    
    print(f"Loading SAM2 model...")
    
    # Use SAM2.1 checkpoint
    sam2_dir = "/home/sra/freeze/sam2"
    checkpoint = os.path.join(sam2_dir, "checkpoints", "sam2.1_hiera_large.pt")

    model_cfg = "sam2.1/sam2.1_hiera_l.yaml"   # RELATIVE, NOT ABSOLUTE

    sam2 = build_sam2(model_cfg, checkpoint, device="cuda")

    mask_generator = SAM2AutomaticMaskGenerator(sam2)
    
    print(f"Processing image: {image_path}")
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    print("Generating masks...")
    masks = mask_generator.generate(image_rgb)
    
    print(f"Generated {len(masks)} masks")

    # Sort masks by area (descending) for consistent ordering
    masks_sorted = sorted(masks, key=lambda x: x['area'], reverse=True)

    # Keep all masks - let feature matching rank them by similarity
    # (top_k will be used later by matching script to return best matches)
    top_masks = masks_sorted

    print(f"Keeping all {len(top_masks)} masks for feature matching")
    
    # Save masks
    os.makedirs(output_dir, exist_ok=True)
    
    mask_paths = []
    metadata = []
    
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    for i, mask_data in enumerate(top_masks):
        mask = mask_data['segmentation']
        area = mask_data['area']
        bbox = mask_data['bbox']  # [x, y, w, h]
        
        # Save mask as binary image
        mask_filename = f"{base_name}_mask_{i:02d}.png"
        mask_path = os.path.join(output_dir, mask_filename)
        
        mask_img = (mask * 255).astype(np.uint8)
        Image.fromarray(mask_img).save(mask_path)
        
        # Create bounding box crop info
        x, y, w, h = bbox
        bbox_info = {
            'x': int(x),
            'y': int(y),
            'width': int(w),
            'height': int(h)
        }
        
        # Save crop of original image
        crop_filename = f"{base_name}_crop_{i:02d}.png"
        crop_path = os.path.join(output_dir, crop_filename)
        
        x_max = min(x + w, image_rgb.shape[1])
        y_max = min(y + h, image_rgb.shape[0])
        crop = image_rgb[int(y):int(y_max), int(x):int(x_max)]
        
        Image.fromarray(crop).save(crop_path)
        
        mask_paths.append(mask_path)
        
        metadata.append({
            'mask_id': i,
            'mask_path': mask_path,
            'crop_path': crop_path,
            'area': int(area),
            'bbox': bbox_info
        })
        
        print(f"  Mask {i}: area={area:.0f}, bbox={bbox_info}")
    
    # Save metadata
    metadata_path = os.path.join(output_dir, f"{base_name}_segmentation_metadata.npy")
    np.save(metadata_path, {
        'image_path': image_path,
        'num_masks': len(top_masks),
        'masks': metadata
    })
    
    print(f"Saved {len(mask_paths)} masks and crops to {output_dir}")
    print(f"Metadata saved to {metadata_path}")
    
    return mask_paths, metadata


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python 3_sam2_localization.py <target_image_path> <output_dir> [top_k]")
        print("  target_image_path: Path to RGB image containing target object")
        print("  output_dir: Directory to save segmentation results")
        print("  top_k: (optional) Number of top masks to keep, default=5")
        sys.exit(1)
    
    image_path = sys.argv[1]
    output_dir = sys.argv[2]
    top_k = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    
    if not os.path.exists(image_path):
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)
    
    mask_paths, metadata = run_sam2_segmentation(image_path, output_dir, top_k)
    
    print("\n" + "="*60)
    print("SAM2 Localization Complete!")
    print("="*60)
    print(f"Generated {len(mask_paths)} candidate masks")
    print(f"Results saved to: {output_dir}")