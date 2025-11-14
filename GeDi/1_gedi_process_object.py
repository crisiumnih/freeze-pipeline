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

def capture_view(pcd, output_path, view_rotation=None):
    """Capture point cloud view using off-screen rendering."""
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=960, height=720)
    
    pcd_copy = o3d.geometry.PointCloud(pcd)
    if view_rotation is not None:
        R = pcd_copy.get_rotation_matrix_from_xyz(view_rotation)
        pcd_copy.rotate(R, center=pcd_copy.get_center())
    
    vis.add_geometry(pcd_copy)
    
    # Set render options
    opt = vis.get_render_option()
    opt.background_color = np.array([1, 1, 1])
    opt.point_size = 3.0
    opt.light_on = False
    
    # Set view control
    ctr = vis.get_view_control()
    ctr.set_front([0.0, 0.0, -1.0])
    ctr.set_up([0.0, -1.0, 0.0])
    ctr.set_zoom(0.8)
    
    vis.poll_events()
    vis.update_renderer()
    
    # Capture and save
    vis.capture_screen_image(output_path, do_render=True)
    vis.destroy_window()
    print(f"Saved: {output_path}")

os.makedirs(output_dir, exist_ok=True)
png_path = os.path.join(output_dir, f"qop_1_GeDi.png")
capture_view(pcd, png_path)

# Capture different viewpoints
view_rotations = {
    "front":  [np.pi, 0, 0],
    "right":  [np.pi, -np.pi/2, 0],
    "left":   [np.pi, np.pi/2, 0],
    "back":   [np.pi, np.pi, 0],
    "top":    [np.pi/2, 0, 0],
    "bottom": [-np.pi/2, 0, 0],
}

for i, (name, rot) in enumerate(view_rotations.items()):
    out_path = os.path.join(out_dir, f"view_{i:02d}_{name}.png")
    capture_view(pcd, out_path, rot)

# Save descriptor
np.save(desc_path, desc)
print("[GeDi] Saved descriptor:", desc_path)
print("[GeDi] Done.")