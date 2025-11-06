import open3d as o3d
import torch
import numpy as np
import time
import os
import csv
from sklearn.decomposition import PCA
from gedi import GeDi

# -----------------------------
# User Options
# -----------------------------
MODE = "all"     # "single" or "all"
SINGLE_LEVEL = "medium_25k"  # used only if MODE == "single"

# -----------------------------
# GeDi Config
# -----------------------------
config = {
    'dim': 32,
    'samples_per_batch': 2000,
    'samples_per_patch_lrf': 1000,
    'samples_per_patch_out': 256,
    'r_lrf': 0.4,
    'fchkpt_gedi_net': 'data/chkpts/3dmatch/chkpt.tar'
}

path = "data/20objects/data/Kinfu_Audiobox1_light/object.xyz"
output_dir = "outputs/gedi_experiments"
os.makedirs(output_dir, exist_ok=True)

# Sampling levels (voxel size controls point count)
sampling_levels = {
    "high_50k": None,
    "medium_25k": 0.0025,
    "low_12k": 0.004,
    "lower_6k": 0.006,
    "tiny_3k": 0.009
}

# Initialize GeDi
gedi = GeDi(config=config)
gedi.gedi_net.cuda()

# -----------------------------
# Helper: Compute + visualize a single level
# -----------------------------
def process_level(level, voxel_size=None, save_png=False, show_window=True):
    print(f"\n[RUN] Level: {level}")

    # Load + optional downsample
    pcd = o3d.io.read_point_cloud(path)
    if voxel_size is not None:
        pcd = pcd.voxel_down_sample(voxel_size)
    points = np.asarray(pcd.points)
    print(f"[INFO] {points.shape[0]} points")

    pts = torch.tensor(points, dtype=torch.float32)
    pcd_t = torch.tensor(points, dtype=torch.float32)

    # Run GeDi
    torch.cuda.synchronize()
    t0 = time.time()
    desc = gedi.compute(pts=pts, pcd=pcd_t)
    torch.cuda.synchronize()
    t_total = (time.time() - t0) * 1000
    print(f"[INFO] GeDi inference time: {t_total:.1f} ms")

    desc_np = desc.cpu().numpy() if isinstance(desc, torch.Tensor) else desc

    # PCA → RGB
    pca = PCA(n_components=3)
    desc_rgb = pca.fit_transform(desc_np)
    desc_rgb = (desc_rgb - desc_rgb.min()) / (desc_rgb.max() - desc_rgb.min())
    pcd.colors = o3d.utility.Vector3dVector(desc_rgb)

    # Visualize or save PNG
    if show_window:
        o3d.visualization.draw_geometries(
            [pcd],
            window_name=f"GeDi Visualization ({level})",
            width=960,
            height=720
        )
    if save_png:
        vis.clear_geometries()
        vis.add_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
        png_path = os.path.join(output_dir, f"{level}.png")
        vis.capture_screen_image(png_path)
        print(f"[INFO] Saved {png_path}")
        return png_path
    return None

# -----------------------------
# Mode 1: Single interactive visualization
# -----------------------------
if MODE == "single":
    level = SINGLE_LEVEL
    voxel_size = sampling_levels.get(level)
    process_level(level, voxel_size, save_png=False, show_window=True)

# -----------------------------
# Mode 2: Generate PNGs for all levels
# -----------------------------
elif MODE == "all":
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=960, height=720)
    opt = vis.get_render_option()
    opt.background_color = np.asarray([1, 1, 1])
    opt.point_size = 2.0
    ctr = vis.get_view_control()
    ctr.set_front([0.0, 0.0, -1.0])
    ctr.set_up([0.0, -1.0, 0.0])
    ctr.set_zoom(0.8)

    csv_path = os.path.join(output_dir, "gedi_experiment_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Level", "Points", "Time_ms", "PNG_File"])

        for level, voxel_size in sampling_levels.items():
            png_path = process_level(level, voxel_size, save_png=True, show_window=False)
            points = len(o3d.io.read_point_cloud(path).points) if voxel_size is None else len(o3d.io.read_point_cloud(path).voxel_down_sample(voxel_size).points)
            writer.writerow([level, points, "—", png_path])

    vis.destroy_window()
    print(f"\n[DONE] All PNGs saved to {output_dir}")

