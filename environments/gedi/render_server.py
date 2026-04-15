#!/usr/bin/env python3
"""
Rendering Server for Query Objects
Uses Open3D 0.15.1 in Python 3.8 environment for stable headless rendering
"""
import sys
import json
import os
import numpy as np

# Set OpenGL platform for headless rendering
os.environ['PYOPENGL_PLATFORM'] = 'egl'

# Suppress Open3D warnings that pollute stdout
import open3d as o3d
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

from pathlib import Path


class RenderServer:
    def __init__(self, image_size=(640, 480)):
        self.image_size = image_size
        print(f"RenderServer initialized: {image_size}", file=sys.stderr)
        print(f"Open3D version: {o3d.__version__}", file=sys.stderr)

    def bake_texture_to_vertex_colors(self, mesh, texture_image):
        """Bake texture to vertex colors for rendering."""
        if not mesh.has_triangle_uvs():
            print("Warning: Mesh has no UVs", file=sys.stderr)
            return mesh

        texture_array = np.asarray(texture_image)
        height, width = texture_array.shape[:2]

        print(f"Baking texture: {width}x{height}", file=sys.stderr)

        vertex_colors = np.zeros((len(mesh.vertices), 3))
        vertex_counts = np.zeros(len(mesh.vertices))

        triangles = np.asarray(mesh.triangles)
        triangle_uvs = np.asarray(mesh.triangle_uvs)

        for tri_idx, tri in enumerate(triangles):
            uv0 = triangle_uvs[tri_idx * 3]
            uv1 = triangle_uvs[tri_idx * 3 + 1]
            uv2 = triangle_uvs[tri_idx * 3 + 2]

            for i, uv in enumerate([uv0, uv1, uv2]):
                u = int(uv[0] * (width - 1))
                v = int((1.0 - uv[1]) * (height - 1))
                u = np.clip(u, 0, width - 1)
                v = np.clip(v, 0, height - 1)

                color = texture_array[v, u, :3] / 255.0
                vertex_idx = tri[i]
                vertex_colors[vertex_idx] += color
                vertex_counts[vertex_idx] += 1

        for i in range(len(vertex_colors)):
            if vertex_counts[i] > 0:
                vertex_colors[i] /= vertex_counts[i]

        mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
        print(f"Baked to {len(vertex_colors)} vertex colors", file=sys.stderr)
        return mesh

    def render_views(self, request):
        """Render multiple views of a mesh."""
        mesh_path = Path(request['mesh_path'])
        num_views = request.get('num_views', 8)
        elevation_angles = request.get('elevation_angles', [0, 30])
        output_dir = Path(request['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load mesh
        print(f"Loading mesh: {mesh_path}", file=sys.stderr)
        mesh = o3d.io.read_triangle_mesh(str(mesh_path))

        if not mesh.has_vertices():
            raise ValueError(f"Failed to load mesh from {mesh_path}")

        # Compute normals
        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()

        # Load and bake texture if exists
        texture_path = mesh_path.with_suffix('.jpg')
        if not texture_path.exists():
            texture_path = mesh_path.with_suffix('.png')

        if texture_path.exists():
            print(f"Loading texture: {texture_path}", file=sys.stderr)
            texture_image = o3d.io.read_image(str(texture_path))
            mesh = self.bake_texture_to_vertex_colors(mesh, texture_image)

        # Center mesh
        mesh_center = mesh.get_center()
        mesh.translate(-mesh_center)

        # Get auto distance
        bbox = mesh.get_axis_aligned_bounding_box()
        max_extent = np.max(bbox.get_extent())
        auto_distance = max_extent * 2.0

        rendered_views = []

        # Create visualizer ONCE and reuse it
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=self.image_size[0], height=self.image_size[1])
        vis.add_geometry(mesh)

        # Set render options
        opt = vis.get_render_option()
        opt.mesh_show_back_face = True
        opt.light_on = True

        if mesh.has_vertex_colors():
            opt.mesh_color_option = o3d.visualization.MeshColorOption.Color

        # Set camera
        ctr = vis.get_view_control()
        ctr.set_front([0.0, 0.0, -1.0])
        ctr.set_lookat([0.0, 0.0, 0.0])
        ctr.set_up([0.0, -1.0, 0.0])
        ctr.set_zoom(0.8)

        # Generate views by rotating mesh
        for elevation in elevation_angles:
            azimuth_angles = np.linspace(0, 360, num_views, endpoint=False)

            for azimuth in azimuth_angles:
                # Rotate mesh
                rotation = [np.radians(elevation), 0, np.radians(-azimuth)]
                R = mesh.get_rotation_matrix_from_xyz(rotation)
                center = mesh.get_center()
                mesh.rotate(R, center=center)

                # Update geometry and render
                vis.update_geometry(mesh)
                vis.poll_events()
                vis.update_renderer()

                # Save RGB image
                view_filename = f"view_az{int(azimuth)}_el{int(elevation)}.png"
                view_path = output_dir / view_filename
                vis.capture_screen_image(str(view_path), do_render=True)

                # Capture and save depth map
                depth_filename = f"depth_az{int(azimuth)}_el{int(elevation)}.npy"
                depth_path = output_dir / depth_filename
                depth_buffer = vis.capture_depth_float_buffer(do_render=False)
                depth_array = np.asarray(depth_buffer)
                np.save(str(depth_path), depth_array)

                rendered_views.append({
                    'path': str(view_path),
                    'depth_path': str(depth_path),
                    'azimuth': float(azimuth),
                    'elevation': float(elevation),
                    'distance': float(auto_distance)
                })

                # Rotate back for next view
                R_inv = mesh.get_rotation_matrix_from_xyz([-r for r in rotation])
                mesh.rotate(R_inv, center=center)

        # Destroy visualizer after all views
        vis.destroy_window()

        return {'views': rendered_views, 'num_views': len(rendered_views)}

    def process_file_request(self, request_path):
        """Process file-based rendering request."""
        with open(request_path) as f:
            request = json.load(f)

        result = self.render_views(request)
        return result


def main():
    if len(sys.argv) < 2:
        print("Usage: python render_server.py <request_json_path>", file=sys.stderr)
        sys.exit(1)

    request_path = sys.argv[1]

    # Initialize server
    server = RenderServer()

    # Process request
    result = server.process_file_request(request_path)

    # Print result as JSON
    print(json.dumps(result, default=str))


if __name__ == '__main__':
    main()
