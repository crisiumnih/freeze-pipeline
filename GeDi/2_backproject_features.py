import os
import sys
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

def create_camera_intrinsic(width, height, fov=60):
    """Create camera intrinsic matrix."""
    fx = fy = width / (2 * np.tan(np.radians(fov) / 2))
    cx = width / 2
    cy = height / 2
    
    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.set_intrinsics(width, height, fx, fy, cx, cy)
    
    return intrinsic

def create_extrinsic_matrix(rotation_angles, pcd_center):
    """
    Create extrinsic matrix for the INVERSE of the point cloud rotation.
    
    When rendering, we rotate the point cloud by R around its center.
    For back-projection, we need to transform points as if the camera moved,
    so we use the inverse transformation.
    
    Args:
        rotation_angles: [rx, ry, rz] - same rotation used during rendering
        pcd_center: center of the point cloud
    
    Returns:
        extrinsic: 4x4 matrix transforming world coords to camera coords
    """
    # The rotation matrix used during rendering
    R_render = o3d.geometry.get_rotation_matrix_from_xyz(rotation_angles)
    
    # For back-projection, we need the inverse transformation
    # R_inverse transforms the rotated point cloud back to original
    R_camera = R_render.T  # Inverse of rotation matrix is its transpose
    
    # Camera looks at origin from position [0, 0, camera_distance]
    camera_distance = 2.0
    camera_pos_cam = np.array([0, 0, camera_distance])
    
    # The camera position in world coordinates (before rotation)
    # Since we rotated the object, the camera stays at origin looking down -Z
    # But we need to express this in terms of the rotated coordinate system
    
    # Simpler approach: 
    # During rendering, object was rotated by R around its center
    # Camera position in the ORIGINAL coordinate system is at [0, 0, camera_distance]
    # looking at pcd_center
    
    # Build extrinsic matrix: points are in ORIGINAL world coordinates
    # We need to transform them to the view where the object was rotated
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = R_camera
    
    # Translation: camera looks at pcd_center from distance camera_distance along +Z
    # After rotation, the camera's position needs adjustment
    t = -R_camera @ pcd_center
    t[2] += camera_distance
    extrinsic[:3, 3] = t
    
    return extrinsic

def project_points_to_image(points_3d, intrinsic, extrinsic, width, height):
    """
    Project 3D points to 2D image coordinates.
    
    Returns:
        pixels: [N, 2] array of (x, y) pixel coordinates  
        depths: [N] array of depth values (Z in camera coordinates)
        valid_mask: [N] boolean mask for points within image bounds
    """
    N = len(points_3d)
    
    # Convert to homogeneous coordinates
    points_homo = np.hstack([points_3d, np.ones((N, 1))])
    
    # Transform to camera coordinates
    points_cam = (extrinsic @ points_homo.T).T
    
    # Extract coordinates
    x_cam = points_cam[:, 0]
    y_cam = points_cam[:, 1]
    z_cam = points_cam[:, 2]
    
    # Get intrinsic parameters
    K = intrinsic.intrinsic_matrix
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    # Project to image plane
    valid_depth = z_cam > 0.01
    
    x_img = np.zeros(N)
    y_img = np.zeros(N)
    
    x_img[valid_depth] = (fx * x_cam[valid_depth] / z_cam[valid_depth]) + cx
    y_img[valid_depth] = (fy * y_cam[valid_depth] / z_cam[valid_depth]) + cy
    
    # Check if within image bounds
    valid_x = (x_img >= 0) & (x_img < width)
    valid_y = (y_img >= 0) & (y_img < height)
    valid_mask = valid_depth & valid_x & valid_y
    
    pixels = np.column_stack([x_img, y_img])
    
    return pixels, z_cam, valid_mask

def match_features_to_points(points_3d, feature_data, intrinsic, extrinsic,
                             width, height, depth_map=None, search_radius=5):
    """
    Match 2D DINOv2 features to 3D points.
    
    NOTE: depth_map from Open3D is NORMALIZED [0,1], not metric depth!
    We can't use it for depth consistency checks without proper conversion.
    """
    M = len(points_3d)
    features = feature_data['features']
    feature_coords = feature_data['coords']
    D = features.shape[1]
    
    feature_img_size = feature_data['image_size']
    feat_h, feat_w = feature_img_size
    
    print(f"  Render space: {width}x{height}, Feature space: {feat_w}x{feat_h}")
    
    # Project 3D points to 2D
    pixels, depths, valid_proj = project_points_to_image(
        points_3d, intrinsic, extrinsic, width, height
    )
    
    # Scale to feature space
    scale_x = feat_w / width
    scale_y = feat_h / height
    
    pixels_scaled = pixels.copy()
    pixels_scaled[:, 0] *= scale_x
    pixels_scaled[:, 1] *= scale_y
    
    # Update valid mask for feature space bounds
    valid_x = (pixels_scaled[:, 0] >= 0) & (pixels_scaled[:, 0] < feat_w)
    valid_y = (pixels_scaled[:, 1] >= 0) & (pixels_scaled[:, 1] < feat_h)
    valid_proj = valid_proj & valid_x & valid_y
    
    # Initialize output
    point_features = np.zeros((M, D), dtype=np.float32)
    point_valid = np.zeros(M, dtype=bool)
    
    if len(feature_coords) == 0:
        print("Warning: No features available")
        return point_features, point_valid
    
    # Build KD-tree (swap from (y,x) to (x,y))
    feature_pixels = feature_coords[:, [1, 0]].astype(np.float32)
    tree = cKDTree(feature_pixels)
    
    valid_indices = np.where(valid_proj)[0]
    print(f"  Projectable points: {len(valid_indices)}/{M}")
    
    if len(valid_indices) == 0:
        print("  Warning: No points projected to image!")
        return point_features, point_valid
    
    valid_pixels = pixels_scaled[valid_indices]
    print(f"  Projected pixel range: x=[{valid_pixels[:, 0].min():.0f}, {valid_pixels[:, 0].max():.0f}], "
          f"y=[{valid_pixels[:, 1].min():.0f}, {valid_pixels[:, 1].max():.0f}]")
    print(f"  Feature pixel range: x=[{feature_pixels[:, 0].min():.0f}, {feature_pixels[:, 0].max():.0f}], "
          f"y=[{feature_pixels[:, 1].min():.0f}, {feature_pixels[:, 1].max():.0f}]")

    num_matches = 0
    
    # Match without depth check (depth map is normalized, not useful)
    for idx in valid_indices:
        px, py = pixels_scaled[idx]
        
        # Find nearest feature pixel
        distances, feat_indices = tree.query([px, py], k=1, distance_upper_bound=search_radius)
        
        if not np.isinf(distances):
            feat_idx = feat_indices
            point_features[idx] = features[feat_idx]
            point_valid[idx] = True
            num_matches += 1
    
    print(f"  Final matches: {num_matches}/{M} ({100*num_matches/M:.1f}%)")
    
    return point_features, point_valid

def aggregate_multiview_features(all_features, all_valid, method='mean'):
    """Aggregate features from multiple views."""
    M = all_features[0].shape[0]
    D = all_features[0].shape[1]
    
    features_stack = np.stack(all_features, axis=0)
    valid_stack = np.stack(all_valid, axis=0)
    
    num_views_per_point = valid_stack.sum(axis=0)
    
    if method == 'mean':
        valid_expanded = valid_stack[:, :, np.newaxis]
        features_masked = features_stack * valid_expanded
        
        sum_features = features_masked.sum(axis=0)
        count_features = valid_stack.sum(axis=0, keepdims=True).T
        count_features = np.maximum(count_features, 1)
        
        aggregated = sum_features / count_features
    else:
        raise ValueError(f"Unknown aggregation method: {method}")
    
    return aggregated, num_views_per_point

# Main execution
if __name__ == "__main__":
    
    if len(sys.argv) < 3:
        print("Usage: python 2_backproject_features.py <processed_folder> <object_name>")
        sys.exit(1)
    
    processed_folder = sys.argv[1]
    object_name = sys.argv[2]
    
    # Load point cloud
    ply_path = os.path.join(processed_folder, f"{object_name}_pc.ply")
    print(f"Loading point cloud: {ply_path}")
    pcd = o3d.io.read_point_cloud(ply_path)
    pcd = pcd.voxel_down_sample(0.005)
    points_3d = np.asarray(pcd.points)
    M = len(points_3d)
    print(f"Point cloud has {M} points")
    
    pcd_center = points_3d.mean(axis=0)
    print(f"Point cloud center: {pcd_center}")
    
    # Load camera parameters
    cam_params = np.load(os.path.join(processed_folder, f"{object_name}_camera_params.npy"), 
                        allow_pickle=True).item()
    width = cam_params['width']
    height = cam_params['height']
    view_rotations = cam_params['view_rotations']
    
    print(f"Image dimensions: {width}x{height}")
    print(f"Number of views: {len(view_rotations)}")
    
    intrinsic = create_camera_intrinsic(width, height, fov=60)
    
    # Process each view
    view_names = list(view_rotations.keys())
    all_features = []
    all_valid = []
    
    for i, view_name in enumerate(view_names):
        print(f"\n[View {i+1}/{len(view_names)}] Processing {view_name}...")
        
        feat_path = os.path.join(processed_folder, f"dino_view_{i:02d}_{view_name}.npy")
        if not os.path.exists(feat_path):
            print(f"  Warning: {feat_path} not found, skipping")
            continue
        
        feature_data = np.load(feat_path, allow_pickle=True).item()
        print(f"  Loaded {feature_data['num_masked_pixels']} feature pixels")
        
        # Create extrinsic matrix
        rotation_angles = view_rotations[view_name]
        extrinsic = create_extrinsic_matrix(rotation_angles, pcd_center)
        
        # Match features (without depth check)
        point_feats, point_valid = match_features_to_points(
            points_3d, feature_data, intrinsic, extrinsic,
            width, height, depth_map=None, search_radius=10
        )
        
        all_features.append(point_feats)
        all_valid.append(point_valid)
    
    # Aggregate
    print("\n" + "="*60)
    print("Aggregating features from all views...")
    print("="*60)
    
    aggregated_features, num_views_per_point = aggregate_multiview_features(
        all_features, all_valid, method='mean'
    )
    
    points_with_features = (num_views_per_point > 0).sum()
    avg_views = num_views_per_point[num_views_per_point > 0].mean() if points_with_features > 0 else 0
    
    print(f"Points with at least 1 view: {points_with_features}/{M} ({100*points_with_features/M:.1f}%)")
    print(f"Average views per point: {avg_views:.1f}")
    
    # Save
    output_path = os.path.join(processed_folder, f"{object_name}_visual_features.npy")
    output_data = {
        'features': aggregated_features,
        'num_views_per_point': num_views_per_point,
        'points_3d': points_3d,
        'feature_dim': aggregated_features.shape[1]
    }
    np.save(output_path, output_data)
    print(f"\nSaved: {output_path}")
    print("="*60)