import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import sys
import torch.nn.functional as F

# Args
img_path = sys.argv[1]
mask_path = sys.argv[2]
out_path = sys.argv[3]

print(f"Processing: {img_path}")
print(f"Using mask: {mask_path}")

# Load model
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
model = model.cuda().eval()

# DINOv2 expects images divisible by patch_size (14 for ViT-L/14)
# Standard approach: use 518x518 (518 = 37*14)
IMG_SIZE = 518
PATCH_SIZE = 14

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Mask preprocessing (no normalization, just resize and crop)
mask_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE, interpolation=transforms.InterpolationMode.NEAREST),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
])

# Load image and mask
img = Image.open(img_path).convert("RGB")
mask = Image.open(mask_path).convert("L")

# Get original size before transform
orig_h, orig_w = img.size[1], img.size[0]

# Transform
x = transform(img).unsqueeze(0).cuda()
mask_tensor = mask_transform(mask).cuda()

# Extract DENSE features from patch tokens
with torch.no_grad():
    # Get intermediate layer output (contains patch tokens)
    # For ViT, the output includes: [CLS token, patch_1, patch_2, ..., patch_N]
    output = model.get_intermediate_layers(x, n=1)[0]
    
    # output shape: [batch_size, num_tokens, feature_dim]
    # num_tokens = 1 (CLS) + num_patches
    # For 518x518 input with patch_size=14: num_patches = 37*37 = 1369
    # So output shape: [1, 1370, 1024]
    
    # Remove CLS token, keep only patch tokens
    patch_tokens = output[:, 1:, :]  # [1, num_patches, 1024]
    
    batch_size, num_patches, feature_dim = patch_tokens.shape
    
    print(f"Patch tokens shape: {patch_tokens.shape}")
    print(f"Number of patches: {num_patches}")
    print(f"Expected patches for {IMG_SIZE}x{IMG_SIZE}: {(IMG_SIZE//PATCH_SIZE)**2}")
    
    # Try to find the actual grid dimensions
    # The model might output a non-square grid depending on image dimensions
    grid_size_h = grid_size_w = IMG_SIZE // PATCH_SIZE
    expected_patches = grid_size_h * grid_size_w
    
    print(f"Expected grid: {grid_size_h}x{grid_size_w} = {expected_patches} patches")
    
    if num_patches != expected_patches:
        print(f"Mismatch! Got {num_patches} patches, expected {expected_patches}")
        
        # Try to find factors of num_patches that are close to square
        # Common cases: 36×38=1368, 37×37=1369
        import math
        best_h, best_w = None, None
        min_diff = float('inf')
        
        for h in range(int(math.sqrt(num_patches)), num_patches + 1):
            if num_patches % h == 0:
                w = num_patches // h
                diff = abs(h - w)
                if diff < min_diff:
                    min_diff = diff
                    best_h, best_w = h, w
        
        if best_h is not None:
            grid_size_h, grid_size_w = best_h, best_w
            print(f"Using inferred grid: {grid_size_h}x{grid_size_w} = {grid_size_h * grid_size_w}")
        else:
            # Fallback: try square with padding
            grid_size_h = grid_size_w = int(math.ceil(math.sqrt(num_patches)))
            print(f"Fallback to square grid with padding: {grid_size_h}x{grid_size_w}")
            # Pad the patches
            target_patches = grid_size_h * grid_size_w
            padding_needed = target_patches - num_patches
            padding = torch.zeros(batch_size, padding_needed, feature_dim, device=patch_tokens.device)
            patch_tokens = torch.cat([patch_tokens, padding], dim=1)
            print(f"Padded {padding_needed} patches")
    else:
        grid_size_h = grid_size_w = grid_size_h  # Square grid
    
    # Reshape to spatial grid: [batch, grid_h, grid_w, feature_dim]
    features_spatial = patch_tokens.reshape(batch_size, grid_size_h, grid_size_w, feature_dim)
    print(f"Reshaped to spatial grid: {features_spatial.shape}")
    features_spatial = features_spatial.permute(0, 3, 1, 2)  # [1, 1024, 37, 37]
    
    # Upsample to match input image resolution
    features_upsampled = F.interpolate(
        features_spatial,
        size=(IMG_SIZE, IMG_SIZE),
        mode='bilinear',
        align_corners=False
    )  # [1, 1024, IMG_SIZE, IMG_SIZE]
    
    # Apply mask to features
    # Expand mask to match feature dimensions
    mask_expanded = mask_tensor.unsqueeze(1)  # [1, 1, IMG_SIZE, IMG_SIZE]
    mask_binary = (mask_expanded > 0.5).float()  # Binary mask
    
    # Mask features
    features_masked = features_upsampled * mask_binary  # [1, 1024, IMG_SIZE, IMG_SIZE]
    
    # Extract only the masked region's features
    # Get coordinates where mask is True
    mask_coords = torch.nonzero(mask_binary[0, 0], as_tuple=False)  # [N, 2] where N = num_masked_pixels
    
    if len(mask_coords) == 0:
        print("Warning: Empty mask, no features extracted")
        features_array = np.array([])
    else:
        # Extract features at masked coordinates
        y_coords = mask_coords[:, 0]
        x_coords = mask_coords[:, 1]
        
        features_at_mask = features_upsampled[0, :, y_coords, x_coords]  # [1024, N]
        features_at_mask = features_at_mask.transpose(0, 1)  # [N, 1024]
        
        features_array = features_at_mask.cpu().numpy()
        coords_array = mask_coords.cpu().numpy()
        
        print(f"Extracted features shape: {features_array.shape}")
        print(f"Feature dimension: {feature_dim}")
        print(f"Number of masked pixels: {len(mask_coords)}")

# Save features with metadata
output_data = {
    'features': features_array,  # [N, 1024] where N = number of masked pixels
    'coords': coords_array if len(mask_coords) > 0 else np.array([]),  # [N, 2] pixel coordinates
    'feature_dim': feature_dim,
    'grid_size': (grid_size_h, grid_size_w),
    'image_size': (IMG_SIZE, IMG_SIZE),
    'original_size': (orig_h, orig_w),
    'num_masked_pixels': len(mask_coords) if len(mask_coords) > 0 else 0
}

np.save(out_path, output_data)
print(f"Saved: {out_path}")

print(f"Features shape: {features_array.shape if len(mask_coords) > 0 else 'empty'}")