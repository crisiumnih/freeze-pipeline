import torch
import numpy as np
import open3d as o3d
from gedi import GeDi
import sys

# ---- args ----
pointcloud_path = sys.argv[1]
out_path = sys.argv[2]

# ---- config ----
config = {
    'dim': 32,
    'samples_per_batch': 500,
    'samples_per_patch_lrf': 4000,
    'samples_per_patch_out': 512,
    'r_lrf': .5,
    'fchkpt_gedi_net': '/home/sra/freeze/gedi_test/gedi/data/chkpts/3dmatch/chkpt.tar'
}

gedi = GeDi(config=config)

# load point cloud
pcd = o3d.io.read_point_cloud(pointcloud_path)
pts = torch.tensor(np.asarray(pcd.points)).float()

# downsample (same as demo)
pcd = pcd.voxel_down_sample(0.01)
_pcd = torch.tensor(np.asarray(pcd.points)).float()

# compute descriptors
desc = gedi.compute(pts=_pcd, pcd=_pcd)   # consistent dims

# ----- save -----
# desc can be torch tensor or numpy array depending on version
if hasattr(desc, "cpu"):
    desc = desc.cpu().numpy()
elif not isinstance(desc, np.ndarray):
    desc = np.asarray(desc)

np.save(out_path, desc)
print("Saved:", out_path)

