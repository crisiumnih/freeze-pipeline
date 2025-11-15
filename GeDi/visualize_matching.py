import os
import sys
import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

"""
Visualize the feature matching process:
1. Show projected 3D points on 2D images
2. Show matched feature locations
3. Show point cloud colored by number of views
4. Show PCA visualization of aggregated features
"""

def create_camera_intrinsic(width, height, fov=60):
    fx = fy = width / (2 * np.tan(np.radians(fov) / 2))
    cx = width / 2
    cy = height / 2
    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.set_intrinsics(width, height, fx, fy, cx, cy)
    return intrinsic

def create_extrinsic_matrix(rotation_angles, pcd_center):
    R_render = o3d.geometry.get_rotation_matrix_from_xyz(rotation_angles)
    R_camera = R_render.T
    
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = R_camera
    
    camera_distance = 2.0
    t = -R_camera @ pcd_center
    t[2] += camera_distance
    extrinsic[:3, 3] = t
    
    return extrinsic

def project_points_to_image(points_3d, intrinsic, extrinsic, width, height):
    N = len(points_3d)
    points_homo = np.hstack([points_3d, np.ones((N, 1))])
    points_cam = (extrinsic @ points_homo.T).T
    
    x_cam, y_cam, z_cam = points_cam[:, 0], points_cam[:, 1], points_cam[:, 2]
    
    K = intrinsic.intrinsic_matrix
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    
    valid_depth = z_cam > 0.01
    x_img, y_img = np.zeros(N), np.zeros(N)
    
    x_img[valid_depth] = (fx * x_cam[valid_depth] / z_cam[valid_depth]) + cx
    y_img[valid_depth] = (fy * y_cam[valid_depth] / z_cam[valid_depth]) + cy
    
    valid_x = (x_img >= 0) & (x_img < width)
    valid_y = (y_img >= 0) & (y_img < height)
    valid_mask = valid_depth & valid_x & valid_y
    
    return np.column_stack([x_img, y_img]), z_cam, valid_mask

def visualize_single_view(view_idx, view_name, processed_folder, object_name, 
                         points_3d, pcd_center, intrinsic, view_rotation, output_folder):
    """Create visualization for a single view showing projected points and matched features."""
    
    # Load rendered image
    img_path = os.path.join(processed_folder, f"view_{view_idx:02d}_{view_name}.png")
    mask_path = os.path.join(processed_folder, f"mask_{view_idx:02d}_{view_name}.png")
    feat_path = os.path.join(processed_folder, f"dino_view_{view_idx:02d}_{view_name}.npy")
    
    if not os.path.exists(img_path) or not os.path.exists(feat_path):
        print(f"Skipping {view_name}: files not found")
        return
    
    # Load data
    img = Image.open(img_path).convert('RGB')
    mask = Image.open(mask_path).convert('L')
    feature_data = np.load(feat_path, allow_pickle=True).item()
    
    width, height = img.size
    feat_h, feat_w = feature_data['image_size']
    
    # Project 3D points
    extrinsic = create_extrinsic_matrix(view_rotation, pcd_center)
    pixels, depths, valid_proj = project_points_to_image(points_3d, intrinsic, extrinsic, width, height)
    
    # Scale to feature space
    scale_x, scale_y = feat_w / width, feat_h / height
    pixels_scaled = pixels.copy()
    pixels_scaled[:, 0] *= scale_x
    pixels_scaled[:, 1] *= scale_y
    
    # Get feature coordinates
    feature_coords = feature_data['coords']  # [N, 2] (y, x)
    feature_pixels = feature_coords[:, [1, 0]].astype(np.float32)  # [N, 2] (x, y)
    
    # Create KD-tree for matching
    tree = cKDTree(feature_pixels)
    
    # Find matches
    valid_indices = np.where(valid_proj)[0]
    matched_3d_pixels = []
    matched_feat_pixels = []
    
    for idx in valid_indices[:100]:  # Visualize first 100 for clarity
        px, py = pixels_scaled[idx]
        distances, feat_idx = tree.query([px, py], k=1, distance_upper_bound=10)
        
        if not np.isinf(distances):
            matched_3d_pixels.append([px, py])
            matched_feat_pixels.append(feature_pixels[feat_idx])
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Original rendered image
    axes[0, 0].imshow(img)
    axes[0, 0].set_title(f'View: {view_name}\nOriginal Render ({width}x{height})')
    axes[0, 0].axis('off')
    
    # 2. Mask
    axes[0, 1].imshow(mask, cmap='gray')
    axes[0, 1].set_title(f'Segmentation Mask\nObject pixels: {feature_data["num_masked_pixels"]}')
    axes[0, 1].axis('off')
    
    # 3. All projected 3D points on image
    img_with_proj = img.copy()
    draw = ImageDraw.Draw(img_with_proj)
    for idx in valid_indices:
        px, py = pixels[idx]
        draw.ellipse([px-2, py-2, px+2, py+2], fill='red', outline='red')
    axes[0, 2].imshow(img_with_proj)
    axes[0, 2].set_title(f'Projected 3D Points (Render Space)\n{len(valid_indices)} points visible')
    axes[0, 2].axis('off')
    
    # 4. Feature extraction visualization (in feature space)
    # Create blank image in feature space
    feat_viz = np.ones((feat_h, feat_w, 3), dtype=np.uint8) * 255
    
    # Draw feature pixels in blue
    for fy, fx in feature_coords[:1000]:  # Draw subset for visibility
        fy, fx = int(fy), int(fx)
        if 0 <= fy < feat_h and 0 <= fx < feat_w:
            feat_viz[max(0,fy-1):min(feat_h,fy+2), max(0,fx-1):min(feat_w,fx+2)] = [0, 0, 255]
    
    # Draw projected points in red
    for px, py in pixels_scaled[valid_indices[:1000]]:
        px, py = int(px), int(py)
        if 0 <= py < feat_h and 0 <= px < feat_w:
            feat_viz[max(0,py-1):min(feat_h,py+2), max(0,px-1):min(feat_w,px+2)] = [255, 0, 0]
    
    axes[1, 0].imshow(feat_viz)
    axes[1, 0].set_title(f'Feature Space ({feat_w}x{feat_h})\nBlue: DINOv2 features, Red: Projected points')
    axes[1, 0].axis('off')
    
    # 5. Matched pairs with lines
    match_viz = feat_viz.copy()
    for (p3d_x, p3d_y), (feat_x, feat_y) in zip(matched_3d_pixels[:50], matched_feat_pixels[:50]):
        p3d_x, p3d_y = int(p3d_x), int(p3d_y)
        feat_x, feat_y = int(feat_x), int(feat_y)
        
        # Draw line
        y_coords = np.linspace(p3d_y, feat_y, 20).astype(int)
        x_coords = np.linspace(p3d_x, feat_x, 20).astype(int)
        for x, y in zip(x_coords, y_coords):
            if 0 <= y < feat_h and 0 <= x < feat_w:
                match_viz[y, x] = [0, 255, 0]
        
        # Draw points
        if 0 <= p3d_y < feat_h and 0 <= p3d_x < feat_w:
            match_viz[max(0,p3d_y-2):min(feat_h,p3d_y+3), max(0,p3d_x-2):min(feat_w,p3d_x+3)] = [255, 0, 0]
        if 0 <= feat_y < feat_h and 0 <= feat_x < feat_w:
            match_viz[max(0,feat_y-2):min(feat_h,feat_y+3), max(0,feat_x-2):min(feat_w,feat_x+3)] = [0, 0, 255]
    
    axes[1, 1].imshow(match_viz)
    axes[1, 1].set_title(f'Feature Matching (50 samples)\nGreen lines: matches')
    axes[1, 1].axis('off')
    
    # 6. Statistics
    stats_text = f"""Matching Statistics:
    
Total 3D points: {len(points_3d)}
Projectable points: {len(valid_indices)}
Feature pixels: {len(feature_pixels)}

Projected range (scaled):
  X: [{pixels_scaled[valid_indices, 0].min():.0f}, {pixels_scaled[valid_indices, 0].max():.0f}]
  Y: [{pixels_scaled[valid_indices, 1].min():.0f}, {pixels_scaled[valid_indices, 1].max():.0f}]

Feature range:
  X: [{feature_pixels[:, 0].min():.0f}, {feature_pixels[:, 0].max():.0f}]
  Y: [{feature_pixels[:, 1].min():.0f}, {feature_pixels[:, 1].max():.0f}]

Matches found: {len(matched_3d_pixels)}
Search radius: 10 pixels
"""
    axes[1, 2].text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center', family='monospace')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    output_path = os.path.join(output_folder, f'match_viz_{view_idx:02d}_{view_name}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved visualization: {output_path}")

def visualize_multiview_coverage(processed_folder, object_name, output_folder):
    """Visualize point cloud colored by number of views that see each point."""
    
    # Load visual features
    visual_feat_path = os.path.join(processed_folder, f"{object_name}_visual_features.npy")
    visual_data = np.load(visual_feat_path, allow_pickle=True).item()
    
    points_3d = visual_data['points_3d']
    num_views = visual_data['num_views_per_point']
    features = visual_data['features']
    
    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    
    # Color by number of views
    colors = plt.cm.viridis(num_views / num_views.max())[:, :3]
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Save colored point cloud
    ply_path = os.path.join(output_folder, f'{object_name}_colored_by_views.ply')
    o3d.io.write_point_cloud(ply_path, pcd)
    print(f"Saved colored point cloud: {ply_path}")
    
    # Create histogram
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].hist(num_views, bins=np.arange(num_views.max() + 2) - 0.5, edgecolor='black')
    axes[0].set_xlabel('Number of Views')
    axes[0].set_ylabel('Number of Points')
    axes[0].set_title(f'Multi-view Coverage\nTotal points: {len(points_3d)}')
    axes[0].grid(True, alpha=0.3)
    
    # Show statistics
    stats_text = f"""Coverage Statistics:
    
Total points: {len(points_3d)}
Points with features: {(num_views > 0).sum()} ({100*(num_views > 0).sum()/len(points_3d):.1f}%)

Views per point:
  Min: {num_views.min()}
  Max: {num_views.max()}
  Mean: {num_views.mean():.2f}
  Median: {np.median(num_views):.1f}

Feature dimension: {features.shape[1]}
"""
    axes[1].text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center', family='monospace')
    axes[1].axis('off')
    
    plt.tight_layout()
    hist_path = os.path.join(output_folder, f'{object_name}_coverage_stats.png')
    plt.savefig(hist_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved coverage histogram: {hist_path}")

def visualize_feature_pca(processed_folder, object_name, output_folder):
    """Visualize aggregated features using PCA."""
    from sklearn.decomposition import PCA
    
    # Load visual features
    visual_feat_path = os.path.join(processed_folder, f"{object_name}_visual_features.npy")
    visual_data = np.load(visual_feat_path, allow_pickle=True).item()
    
    points_3d = visual_data['points_3d']
    features = visual_data['features']
    num_views = visual_data['num_views_per_point']
    
    # Filter points with features
    valid_mask = num_views > 0
    valid_points = points_3d[valid_mask]
    valid_features = features[valid_mask]
    
    if valid_features.shape[0] == 0:
        print("No valid features to visualize!")
        return
    
    # PCA to 3D for visualization
    pca = PCA(n_components=3)
    features_3d = pca.fit_transform(valid_features)
    
    # Normalize to [0, 1] for colors
    features_rgb = (features_3d - features_3d.min(axis=0)) / (features_3d.max(axis=0) - features_3d.min(axis=0))
    
    # Create colored point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(valid_points)
    pcd.colors = o3d.utility.Vector3dVector(features_rgb)
    
    # Save
    ply_path = os.path.join(output_folder, f'{object_name}_pca_features.ply')
    o3d.io.write_point_cloud(ply_path, pcd)
    print(f"Saved PCA-colored point cloud: {ply_path}")
    
    # Show explained variance
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(1, 4), pca.explained_variance_ratio_)
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Explained Variance Ratio')
    ax.set_title(f'PCA of Visual Features\nTotal variance explained: {pca.explained_variance_ratio_.sum():.2%}')
    ax.set_xticks(range(1, 4))
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    pca_path = os.path.join(output_folder, f'{object_name}_pca_variance.png')
    plt.savefig(pca_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved PCA variance plot: {pca_path}")

# Main
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python visualize_matching.py <processed_folder> <object_name>")
        sys.exit(1)
    
    processed_folder = sys.argv[1]
    object_name = sys.argv[2]
    
    output_folder = os.path.join(processed_folder, "visualizations")
    os.makedirs(output_folder, exist_ok=True)
    
    # Load point cloud
    ply_path = os.path.join(processed_folder, f"{object_name}_pc.ply")
    pcd = o3d.io.read_point_cloud(ply_path)
    pcd = pcd.voxel_down_sample(0.005)
    points_3d = np.asarray(pcd.points)
    pcd_center = points_3d.mean(axis=0)
    
    # Load camera params
    cam_params = np.load(os.path.join(processed_folder, f"{object_name}_camera_params.npy"), 
                        allow_pickle=True).item()
    width = cam_params['width']
    height = cam_params['height']
    view_rotations = cam_params['view_rotations']
    
    intrinsic = create_camera_intrinsic(width, height, fov=60)
    
    print("="*60)
    print("Creating visualizations...")
    print("="*60)
    
    # Visualize each view
    for i, (view_name, rotation) in enumerate(view_rotations.items()):
        print(f"\n[{i+1}/{len(view_rotations)}] Visualizing {view_name}...")
        visualize_single_view(i, view_name, processed_folder, object_name,
                            points_3d, pcd_center, intrinsic, rotation, output_folder)
    
    # Multi-view coverage
    print("\nCreating multi-view coverage visualization...")
    visualize_multiview_coverage(processed_folder, object_name, output_folder)
    
    # PCA visualization
    print("\nCreating PCA feature visualization...")
    visualize_feature_pca(processed_folder, object_name, output_folder)
    
    print("\n" + "="*60)
    print("Visualization complete!")
    print("="*60)
    print(f"Output folder: {output_folder}")
    print("\nGenerated files:")
    print("  - match_viz_XX_<view>.png : Feature matching for each view")
    print("  - <object>_colored_by_views.ply : Point cloud colored by view count")
    print("  - <object>_coverage_stats.png : Coverage statistics")
    print("  - <object>_pca_features.ply : Point cloud colored by PCA features")
    print("  - <object>_pca_variance.png : PCA explained variance")