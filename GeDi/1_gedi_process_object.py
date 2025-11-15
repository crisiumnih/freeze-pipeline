import os
import sys
import numpy as np
import open3d as o3d
import torch
from gedi import GeDi
from sklearn.decomposition import PCA


obj_dir = sys.argv[1]     # path to folder like: data/20objects/data/Kinfu_Audiobox1_light
out_dir = sys.argv[2]     # where to store pointcloud + descriptors

os.makedirs(out_dir, exist_ok=True)

object_name = os.path.basename(obj_dir.rstrip("/"))
xyz_path = os.path.join(obj_dir, "object.xyz")
ply_path = os.path.join(out_dir, f"{object_name}_pc.ply")
desc_path = os.path.join(out_dir, f"{object_name}_gedi.npy")
output_dir = "outputs/qop"

config = {
    'dim': 32,
    'samples_per_batch': 5000,
    'samples_per_patch_lrf': 6000,
    'samples_per_patch_out': 512,
    'r_lrf': 0.5,
    'fchkpt_gedi_net': '/home/sra/freeze/gedi_test/gedi/data/chkpts/3dmatch/chkpt.tar'
}

gedi = GeDi(config)

os.environ['PYOPENGL_PLATFORM'] = 'egl'

if not os.path.exists(ply_path):
    print("[GeDi] Creating point cloud:", ply_path)

    pts = np.loadtxt(xyz_path).astype(np.float32)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    o3d.io.write_point_cloud(ply_path, pcd)

else:
    print("[GeDi] Using existing point cloud:", ply_path)


pcd = o3d.io.read_point_cloud(ply_path)

# Downsample same as demo
pcd = pcd.voxel_down_sample(0.005)

pts = torch.tensor(np.asarray(pcd.points)).float()


print("[GeDi] Computing descriptors...")
desc = gedi.compute(pts=pts, pcd=pts)


if hasattr(desc, "cpu"):
    desc = desc.cpu().numpy()
elif not isinstance(desc, np.ndarray):
    desc = np.asarray(desc)

pca = PCA(n_components=3)
desc_rgb = pca.fit_transform(desc)
desc_rgb = (desc_rgb - desc_rgb.min()) / (desc_rgb.max() - desc_rgb.min())
pcd.colors = o3d.utility.Vector3dVector(desc_rgb)

def capture_view_with_mask(pcd, output_rgb_path, output_mask_path, output_depth_path, view_rotation=None):
    """
    Capture point cloud view with RGB, segmentation mask, and depth using off-screen rendering.
    
    Args:
        pcd: Open3D point cloud
        output_rgb_path: Path to save RGB image
        output_mask_path: Path to save binary segmentation mask
        output_depth_path: Path to save depth map
        view_rotation: Rotation angles [x, y, z] in radians
    """
    from PIL import Image
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=960, height=720)
    
    pcd_copy = o3d.geometry.PointCloud(pcd)
    if view_rotation is not None:
        R = pcd_copy.get_rotation_matrix_from_xyz(view_rotation)
        pcd_copy.rotate(R, center=pcd_copy.get_center())
    
    vis.add_geometry(pcd_copy)
    
    # Set render options for RGB
    opt = vis.get_render_option()
    opt.background_color = np.array([1, 1, 1])  # White background
    opt.point_size = 5.0  # Larger points for better coverage
    opt.light_on = False
    
    # Set view control
    ctr = vis.get_view_control()
    ctr.set_front([0.0, 0.0, -1.0])
    ctr.set_up([0.0, -1.0, 0.0])
    ctr.set_zoom(0.8)
    
    vis.poll_events()
    vis.update_renderer()
    
    # Capture RGB image
    vis.capture_screen_image(output_rgb_path, do_render=True)
    print(f"Saved RGB: {output_rgb_path}")
    
    # Load RGB to create mask (simple approach)
    rgb_img = Image.open(output_rgb_path)
    rgb_array = np.array(rgb_img)
    
    # Create mask: non-white pixels are the object
    # White background has RGB = (255, 255, 255)
    white_threshold = 250  # Slightly less than 255 to handle compression artifacts
    is_white = np.all(rgb_array >= white_threshold, axis=-1)
    mask = (~is_white).astype(np.uint8) * 255
    
    # Save mask as image
    mask_img = Image.fromarray(mask)
    mask_img.save(output_mask_path)
    print(f"Saved mask: {output_mask_path} - Object pixels: {mask.sum() // 255}")
    
    # Capture depth image
    depth = vis.capture_depth_float_buffer(do_render=True)
    depth_array = np.asarray(depth)
    
    # Debug depth values
    valid_depth_pixels = (depth_array > 0) & (depth_array < 1.0)
    print(f"  Depth range: [{depth_array.min():.3f}, {depth_array.max():.3f}]")
    print(f"  Valid depth pixels: {valid_depth_pixels.sum()}")
    
    # Save depth as numpy array (for later use in back-projection)
    np.save(output_depth_path.replace('.png', '.npy'), depth_array)
    
    # Save depth visualization
    if depth_array.max() > depth_array.min():
        depth_normalized = ((depth_array - depth_array.min()) / 
                           (depth_array.max() - depth_array.min()) * 255).astype(np.uint8)
    else:
        depth_normalized = np.zeros_like(depth_array, dtype=np.uint8)
    
    depth_img = Image.fromarray(depth_normalized)
    depth_img.save(output_depth_path)
    print(f"Saved depth: {output_depth_path}")
    
    vis.destroy_window()

os.makedirs(output_dir, exist_ok=True)

# Capture main view with mask
png_path = os.path.join(output_dir, f"qop_1_GeDi.png")
mask_path = os.path.join(output_dir, f"qop_1_GeDi_mask.png")
depth_path = os.path.join(output_dir, f"qop_1_GeDi_depth.png")
capture_view_with_mask(pcd, png_path, mask_path, depth_path)

# Capture different viewpoints with masks
view_rotations = {
    "front":  [np.pi, 0, 0],
    "right":  [np.pi, -np.pi/2, 0],
    "left":   [np.pi, np.pi/2, 0],
    "back":   [np.pi, np.pi, 0],
    "top":    [np.pi/2, 0, 0],
    "bottom": [-np.pi/2, 0, 0],
}

for i, (name, rot) in enumerate(view_rotations.items()):
    rgb_path = os.path.join(out_dir, f"view_{i:02d}_{name}.png")
    mask_path = os.path.join(out_dir, f"mask_{i:02d}_{name}.png")
    depth_path = os.path.join(out_dir, f"depth_{i:02d}_{name}.png")
    capture_view_with_mask(pcd, rgb_path, mask_path, depth_path, rot)

# Save descriptor
np.save(desc_path, desc)
print("[GeDi] Saved descriptor:", desc_path)

# Save camera parameters for later back-projection
camera_params = {
    'width': 960,  # Match the actual render resolution
    'height': 720,
    'view_rotations': view_rotations
}
np.save(os.path.join(out_dir, f"{object_name}_camera_params.npy"), camera_params)
print("[GeDi] Saved camera parameters")

print("[GeDi] Done.")