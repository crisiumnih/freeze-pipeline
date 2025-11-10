import open3d as o3d
import numpy as np
import torch
import os
from gedi import GeDi

# -----------------------------
# Paths and camera setup
# -----------------------------
base_dir = "data/20objects/data/Kinfu_Samurai1_light"
depth_dir = os.path.join(base_dir, "depth_noseg")
info_dir = os.path.join(base_dir, "info")

intrinsic = o3d.camera.PinholeCameraIntrinsic(
    width=640, height=480,
    fx=525.0, fy=525.0, cx=319.5, cy=239.5
)

# -----------------------------
# Robust parser for info files
# -----------------------------
def parse_info(info_path):
    with open(info_path, "r") as f:
        lines = [l.strip() for l in f if l.strip()]
    rot_idx = next(i for i, l in enumerate(lines) if l.lower().startswith("rotation"))
    center_idx = next(i for i, l in enumerate(lines) if l.lower().startswith("center"))
    rot = np.array([list(map(float, lines[rot_idx + j + 1].split())) for j in range(3)])
    center = np.array(list(map(float, lines[center_idx + 1].split())))
    return rot, center

# -----------------------------
# Create point cloud from depth
# -----------------------------
def create_pcd(depth_filename, info_filename):
    depth_path = os.path.join(depth_dir, depth_filename)
    info_path = os.path.join(info_dir, info_filename)
    depth = o3d.io.read_image(depth_path)
    pcd = o3d.geometry.PointCloud.create_from_depth_image(
        depth, intrinsic, depth_scale=1000.0, depth_trunc=3.0, stride=1
    )
    rot, center = parse_info(info_path)
    transform = np.eye(4)
    transform[:3, :3] = rot
    transform[:3, 3] = center
    pcd.transform(transform)
    return pcd

# -----------------------------
# Create and color point clouds
# -----------------------------
depth_files = ["depth_00019.png", "depth_00361.png"]
info_files = ["info_00019.txt", "info_00361.txt"]

pcd0 = create_pcd(depth_files[0], info_files[0])
pcd1 = create_pcd(depth_files[1], info_files[1])

pcd0.paint_uniform_color([1, 0.706, 0])     # yellowish
pcd1.paint_uniform_color([0, 0.651, 0.929]) # bluish

# -----------------------------
# Initialize GeDi
# -----------------------------
config = {
    'dim': 32,
    'samples_per_batch': 500,
    'samples_per_patch_lrf': 4000,
    'samples_per_patch_out': 512,
    'r_lrf': 0.5,
    'fchkpt_gedi_net': 'data/chkpts/3dmatch/chkpt.tar'
}
gedi = GeDi(config=config)

# -----------------------------
# Prepare samples
# -----------------------------
voxel_size = 0.01
patches_per_pair = 5000

inds0 = np.random.choice(np.asarray(pcd0.points).shape[0], patches_per_pair, replace=False)
inds1 = np.random.choice(np.asarray(pcd1.points).shape[0], patches_per_pair, replace=False)

pts0 = torch.tensor(np.asarray(pcd0.points)[inds0]).float()
pts1 = torch.tensor(np.asarray(pcd1.points)[inds1]).float()

_pcd0 = torch.tensor(np.asarray(pcd0.voxel_down_sample(voxel_size).points)).float()
_pcd1 = torch.tensor(np.asarray(pcd1.voxel_down_sample(voxel_size).points)).float()

# -----------------------------
# Compute GeDi descriptors
# -----------------------------
pcd0_desc = gedi.compute(pts=pts0, pcd=_pcd0)
pcd1_desc = gedi.compute(pts=pts1, pcd=_pcd1)

# -----------------------------
# Open3D RANSAC registration
# -----------------------------
feat0 = o3d.pipelines.registration.Feature()
feat1 = o3d.pipelines.registration.Feature()
feat0.data = pcd0_desc.T
feat1.data = pcd1_desc.T

_pcd0_o3d = o3d.geometry.PointCloud()
_pcd1_o3d = o3d.geometry.PointCloud()
_pcd0_o3d.points = o3d.utility.Vector3dVector(np.asarray(pcd0.points)[inds0])
_pcd1_o3d.points = o3d.utility.Vector3dVector(np.asarray(pcd1.points)[inds1])

result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
    _pcd0_o3d, _pcd1_o3d,
    feat0, feat1,
    mutual_filter=True,
    max_correspondence_distance=0.02,
    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
    ransac_n=3,
    checkers=[
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(0.02)
    ],
    criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(50000, 1000)
)

# -----------------------------
# Apply transformation and visualize
# -----------------------------
pcd0.transform(result.transformation)
fused = pcd0 + pcd1

print("Transformation matrix estimated:\n", result.transformation)

o3d.visualization.draw_geometries([fused], window_name="Fused (GeDi-aligned)")
