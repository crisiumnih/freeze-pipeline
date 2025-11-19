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

# Auto-detect input file format
xyz_path = os.path.join(obj_dir, "object.xyz")
obj_path = os.path.join(obj_dir, "object.obj")
ply_input_path = os.path.join(obj_dir, "object.ply")

# Determine which file exists
if os.path.exists(xyz_path):
    input_file = xyz_path
elif os.path.exists(obj_path):
    input_file = obj_path
elif os.path.exists(ply_input_path):
    input_file = ply_input_path
else:
    raise FileNotFoundError(f"No object file found in {obj_dir}. Expected object.xyz, object.obj, or object.ply")

# Use the detected file
xyz_path = input_file
print(f"[GeDi] Using input file: {xyz_path}")

ply_path = os.path.join(out_dir, f"{object_name}_pc.ply")
desc_path = os.path.join(out_dir, f"{object_name}_gedi.npy")
output_dir = "outputs/qop_2"

config = {
    'dim': 32,
    'samples_per_batch': 2000,  # Reduced from 5000 to avoid OOM
    'samples_per_patch_lrf': 3000,  # Reduced from 6000 to avoid OOM
    'samples_per_patch_out': 512,
    'r_lrf': 0.5,
    'fchkpt_gedi_net': '/home/sra/freeze/gedi_test/gedi/data/chkpts/3dmatch/chkpt.tar'
}

gedi = GeDi(config)

os.environ['PYOPENGL_PLATFORM'] = 'egl'

# Keep reference to textured mesh for rendering
textured_mesh = None

def bake_texture_to_vertex_colors(mesh):
    """
    Convert texture-mapped mesh to vertex-colored mesh.
    Open3D's visualizer doesn't render textures, so we bake them into vertex colors.
    """
    if not mesh.has_textures() or not mesh.has_triangle_uvs():
        print("[GeDi] Warning: Mesh has no texture or UVs, cannot bake texture")
        return mesh

    from PIL import Image
    import numpy as np

    # Get texture image
    if len(mesh.textures) == 0:
        print("[GeDi] Warning: No textures found in mesh")
        return mesh

    # Open3D stores textures as Image objects
    texture_img = mesh.textures[0]

    # Convert Open3D image to numpy array
    texture_array = np.asarray(texture_img)
    height, width = texture_array.shape[:2]

    print(f"[GeDi] Baking texture to vertex colors: {width}x{height}")

    # Sample texture colors at triangle UVs and assign to vertices
    # This is an approximation - we'll average colors from triangles using each vertex
    vertex_colors = np.zeros((len(mesh.vertices), 3))
    vertex_counts = np.zeros(len(mesh.vertices))

    triangles = np.asarray(mesh.triangles)
    triangle_uvs = np.asarray(mesh.triangle_uvs)

    for tri_idx, tri in enumerate(triangles):
        # Get UV coordinates for this triangle's vertices
        uv0 = triangle_uvs[tri_idx * 3]
        uv1 = triangle_uvs[tri_idx * 3 + 1]
        uv2 = triangle_uvs[tri_idx * 3 + 2]

        # Sample texture at UV coordinates
        for i, uv in enumerate([uv0, uv1, uv2]):
            # UV coordinates are in [0, 1], convert to pixel coordinates
            u = int(uv[0] * (width - 1))
            v = int((1.0 - uv[1]) * (height - 1))  # Flip V coordinate

            # Clamp to valid range
            u = np.clip(u, 0, width - 1)
            v = np.clip(v, 0, height - 1)

            # Get color from texture
            color = texture_array[v, u, :3] / 255.0  # Normalize to [0, 1]

            # Add to vertex color
            vertex_idx = tri[i]
            vertex_colors[vertex_idx] += color
            vertex_counts[vertex_idx] += 1

    # Average colors
    for i in range(len(vertex_colors)):
        if vertex_counts[i] > 0:
            vertex_colors[i] /= vertex_counts[i]

    # Set vertex colors
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    print(f"[GeDi] Baked texture to {len(vertex_colors)} vertex colors")

    return mesh

if not os.path.exists(ply_path):
    print("[GeDi] Creating point cloud:", ply_path)

    # Handle different file formats
    if xyz_path.endswith('.xyz'):
        # Load XYZ point cloud
        pts = np.loadtxt(xyz_path).astype(np.float32)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
    elif xyz_path.endswith('.obj') or xyz_path.endswith('.ply'):
        # Load mesh and sample points from it
        print(f"[GeDi] Loading mesh from: {xyz_path}")
        mesh = o3d.io.read_triangle_mesh(xyz_path)

        print(f"[GeDi] Mesh has texture: {mesh.has_textures()}")
        print(f"[GeDi] Mesh has triangle UVs: {mesh.has_triangle_uvs()}")
        print(f"[GeDi] Mesh has vertex colors: {mesh.has_vertex_colors()}")
        print(f"[GeDi] Number of textures: {len(mesh.textures)}")

        # Explicitly load texture if not already loaded
        if not mesh.has_textures() and xyz_path.endswith('.obj'):
            # Try to find and load texture explicitly
            obj_dir = os.path.dirname(xyz_path)
            obj_name = os.path.basename(xyz_path).replace('object.obj', '')

            # Look for texture file
            for ext in ['.jpg', '.png', '.jpeg']:
                # Try multiple naming patterns
                texture_paths = [
                    os.path.join(obj_dir, f"Pineapple{ext}"),  # EXPLICIT: Pineapple.jpg
                    os.path.join(obj_dir, f"{object_name}{ext}"),
                    os.path.join(obj_dir, f"texture{ext}"),
                ]

                for texture_path in texture_paths:
                    if os.path.exists(texture_path):
                        print(f"[GeDi] Explicitly loading texture: {texture_path}")
                        texture_img = o3d.io.read_image(texture_path)
                        mesh.textures = [texture_img]
                        print(f"[GeDi] Loaded texture: {texture_path}")
                        break
                if mesh.has_textures():
                    break

        # Bake texture to vertex colors for rendering
        if mesh.has_textures() and mesh.has_triangle_uvs():
            textured_mesh = bake_texture_to_vertex_colors(mesh)
        else:
            textured_mesh = mesh
            print("[GeDi] Warning: No texture to bake, using mesh as-is")

        # Decimate mesh to reduce memory usage during rendering
        if textured_mesh is not None and isinstance(textured_mesh, o3d.geometry.TriangleMesh):
            num_vertices = len(textured_mesh.vertices)
            if num_vertices > 50000:
                target_vertices = 50000
                target_triangles = int(len(textured_mesh.triangles) * (target_vertices / num_vertices))
                print(f"[GeDi] Decimating mesh: {num_vertices} → {target_vertices} vertices to reduce memory")
                textured_mesh = textured_mesh.simplify_quadric_decimation(target_number_of_triangles=target_triangles)
                print(f"[GeDi] Decimated to {len(textured_mesh.vertices)} vertices, {len(textured_mesh.triangles)} triangles")

        # Sample points from mesh surface for GeDi processing
        pcd = mesh.sample_points_uniformly(number_of_points=100000)
        print(f"[GeDi] Sampled {len(pcd.points)} points from mesh")
    else:
        raise ValueError(f"Unsupported file format: {xyz_path}")

    o3d.io.write_point_cloud(ply_path, pcd)

else:
    print("[GeDi] Using existing point cloud:", ply_path)

    # Try to load textured mesh for rendering
    if xyz_path.endswith('.obj') or xyz_path.endswith('.ply'):
        print(f"[GeDi] Loading textured mesh for rendering: {xyz_path}")
        mesh = o3d.io.read_triangle_mesh(xyz_path)

        print(f"[GeDi] Mesh has texture: {mesh.has_textures()}")
        print(f"[GeDi] Mesh has triangle UVs: {mesh.has_triangle_uvs()}")
        print(f"[GeDi] Number of textures: {len(mesh.textures)}")

        # Explicitly load texture if not already loaded
        if not mesh.has_textures() and xyz_path.endswith('.obj'):
            obj_dir = os.path.dirname(xyz_path)

            # Look for texture file
            for ext in ['.jpg', '.png', '.jpeg']:
                texture_paths = [
                    os.path.join(obj_dir, f"Pineapple{ext}"),  # EXPLICIT: Pineapple.jpg
                    os.path.join(obj_dir, f"{object_name}{ext}"),
                    os.path.join(obj_dir, f"texture{ext}"),
                ]

                for texture_path in texture_paths:
                    if os.path.exists(texture_path):
                        print(f"[GeDi] Explicitly loading texture: {texture_path}")
                        texture_img = o3d.io.read_image(texture_path)
                        mesh.textures = [texture_img]
                        print(f"[GeDi] Loaded texture: {texture_path}")
                        break
                if mesh.has_textures():
                    break

        # Bake texture to vertex colors
        if mesh.has_textures() and mesh.has_triangle_uvs():
            textured_mesh = bake_texture_to_vertex_colors(mesh)
        else:
            textured_mesh = mesh
            print("[GeDi] Warning: No texture to bake")

        # Decimate mesh to reduce memory usage during rendering
        if textured_mesh is not None and isinstance(textured_mesh, o3d.geometry.TriangleMesh):
            num_vertices = len(textured_mesh.vertices)
            if num_vertices > 50000:
                target_vertices = 50000
                target_triangles = int(len(textured_mesh.triangles) * (target_vertices / num_vertices))
                print(f"[GeDi] Decimating mesh: {num_vertices} → {target_vertices} vertices to reduce memory")
                textured_mesh = textured_mesh.simplify_quadric_decimation(target_number_of_triangles=target_triangles)
                print(f"[GeDi] Decimated to {len(textured_mesh.vertices)} vertices, {len(textured_mesh.triangles)} triangles")


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

# Handle NaN values in descriptors
nan_mask = np.isnan(desc).any(axis=1)
num_nan = nan_mask.sum()
if num_nan > 0:
    print(f"[GeDi] Warning: {num_nan}/{len(desc)} descriptors contain NaN values")
    # Replace NaN with zeros
    desc = np.nan_to_num(desc, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"[GeDi] Replaced NaN values with zeros")

pca = PCA(n_components=3)
desc_rgb = pca.fit_transform(desc)
desc_rgb = (desc_rgb - desc_rgb.min()) / (desc_rgb.max() - desc_rgb.min())
pcd.colors = o3d.utility.Vector3dVector(desc_rgb)

def capture_view_with_mask(geometry, output_rgb_path, output_mask_path, output_depth_path, view_rotation=None):
    """
    Capture geometry view with RGB, segmentation mask, and depth using off-screen rendering.

    Args:
        geometry: Open3D point cloud or triangle mesh
        output_rgb_path: Path to save RGB image
        output_mask_path: Path to save binary segmentation mask
        output_depth_path: Path to save depth map
        view_rotation: Rotation angles [x, y, z] in radians
    """
    from PIL import Image

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=1920, height=1080)

    # Make a copy of the geometry (handles both PointCloud and TriangleMesh)
    if isinstance(geometry, o3d.geometry.PointCloud):
        geom_copy = o3d.geometry.PointCloud(geometry)
    elif isinstance(geometry, o3d.geometry.TriangleMesh):
        geom_copy = o3d.geometry.TriangleMesh(geometry)
    else:
        raise TypeError(f"Unsupported geometry type: {type(geometry)}")

    if view_rotation is not None:
        R = geom_copy.get_rotation_matrix_from_xyz(view_rotation)
        geom_copy.rotate(R, center=geom_copy.get_center())

    vis.add_geometry(geom_copy)

    # Set render options for RGB
    opt = vis.get_render_option()
    opt.background_color = np.array([1, 1, 1])  # White background

    # Set different rendering options based on geometry type
    if isinstance(geometry, o3d.geometry.PointCloud):
        opt.point_size = 5.0  # Larger points for better coverage
        opt.light_on = False
    elif isinstance(geometry, o3d.geometry.TriangleMesh):
        opt.mesh_show_back_face = True
        opt.mesh_show_wireframe = False
        opt.mesh_color_option = o3d.visualization.MeshColorOption.Color
        # Enable lighting but keep it subtle to preserve colors
        opt.light_on = True
    
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

    # Clean up
    vis.destroy_window()
    del geom_copy
    del vis

    # Force garbage collection to free memory
    import gc
    gc.collect()

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

    # Use textured mesh for rendering if available, otherwise use PCA-colored point cloud
    geometry_to_render = textured_mesh if textured_mesh is not None else pcd
    capture_view_with_mask(geometry_to_render, rgb_path, mask_path, depth_path, rot)

# Save descriptor
np.save(desc_path, desc)
print("[GeDi] Saved descriptor:", desc_path)

# Save camera parameters for later back-projection
# Using HOPE dataset RealSense D415 camera specifications
camera_params = {
    'width': 1920,
    'height': 1080,
    'fx': 1390.53,
    'fy': 1386.99,
    'cx': 964.957,
    'cy': 522.586,
    'intrinsics': [[1390.53, 0.0, 964.957],
                   [0.0, 1386.99, 522.586],
                   [0.0, 0.0, 1.0]],
    'view_rotations': view_rotations,
    'camera_model': 'RealSense D415'
}
np.save(os.path.join(out_dir, f"{object_name}_camera_params.npy"), camera_params)
print("[GeDi] Saved camera parameters")

print("[GeDi] Done.")