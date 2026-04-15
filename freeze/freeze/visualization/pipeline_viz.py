"""
Pipeline visualization utilities.

Creates visual outputs for each stage of the FreeZe pipeline.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import imageio
from typing import List, Dict, Tuple, Optional
import trimesh
import cv2
import open3d as o3d


class PipelineVisualizer:
    """Visualize FreeZe pipeline stages and create GIFs."""

    def __init__(self, output_dir: Path):
        """
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.frames = []  # Store frame paths for GIF creation

    def add_title_frame(self, title: str, subtitle: str = "", duration: float = 2.0) -> Path:
        """
        Create a title frame.

        Args:
            title: Main title text
            subtitle: Subtitle text
            duration: Duration in seconds (for GIF)

        Returns:
            Path to saved frame
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        # Title
        ax.text(0.5, 0.6, title, ha='center', va='center',
                fontsize=36, fontweight='bold', color='#2c3e50')

        # Subtitle
        if subtitle:
            ax.text(0.5, 0.4, subtitle, ha='center', va='center',
                    fontsize=20, color='#7f8c8d')

        # Save
        frame_path = self.output_dir / f'frame_{len(self.frames):03d}_title.png'
        plt.tight_layout()
        plt.savefig(frame_path, dpi=100, bbox_inches='tight', facecolor='white')
        plt.close()

        self.frames.append({'path': frame_path, 'duration': duration})
        return frame_path

    def visualize_query_mesh(self, mesh_path: Path, duration: float = 2.0) -> Path:
        """
        Visualize the query mesh from multiple angles.

        Args:
            mesh_path: Path to mesh file
            duration: Duration in seconds

        Returns:
            Path to saved frame
        """
        # Load mesh
        mesh = trimesh.load(mesh_path)

        # Create figure with 3 views
        fig = plt.figure(figsize=(15, 5))

        # Camera angles (azimuth, elevation)
        angles = [(0, 0), (90, 0), (45, 30)]
        titles = ['Front View', 'Side View', 'Perspective']

        for i, ((azim, elev), title) in enumerate(zip(angles, titles)):
            ax = fig.add_subplot(1, 3, i + 1, projection='3d')

            # Plot mesh vertices
            vertices = mesh.vertices
            faces = mesh.faces

            ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                           triangles=faces, cmap='viridis', alpha=0.8, edgecolor='none')

            # Set view angle
            ax.view_init(elev=elev, azim=azim)
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')

            # Equal aspect ratio
            max_range = np.array([vertices[:, 0].max() - vertices[:, 0].min(),
                                 vertices[:, 1].max() - vertices[:, 1].min(),
                                 vertices[:, 2].max() - vertices[:, 2].min()]).max() / 2.0
            mid_x = (vertices[:, 0].max() + vertices[:, 0].min()) * 0.5
            mid_y = (vertices[:, 1].max() + vertices[:, 1].min()) * 0.5
            mid_z = (vertices[:, 2].max() + vertices[:, 2].min()) * 0.5
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)

        plt.suptitle(f'Query Object: {mesh_path.name}', fontsize=16, fontweight='bold', y=0.98)

        # Save
        frame_path = self.output_dir / f'frame_{len(self.frames):03d}_query_mesh.png'
        plt.tight_layout()
        plt.savefig(frame_path, dpi=100, bbox_inches='tight', facecolor='white')
        plt.close()

        self.frames.append({'path': frame_path, 'duration': duration})
        return frame_path

    def visualize_query_renders(self, renders: List[Tuple[np.ndarray, np.ndarray, Dict]],
                                duration: float = 2.0) -> Path:
        """
        Visualize rendered views of the query object.

        Args:
            renders: List of (image, depth, metadata) tuples
            duration: Duration in seconds

        Returns:
            Path to saved frame
        """
        n_views = len(renders)  # Show all views
        cols = 3
        rows = (n_views + cols - 1) // cols  # Ceiling division
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
        
        # Flatten axes array for easier indexing
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        axes_flat = axes.flatten()

        for i in range(n_views):
            image, depth, metadata = renders[i]

            # Show RGB view
            axes_flat[i].imshow(image)
            axes_flat[i].set_title(f"View {i+1}\nAz={metadata.get('azimuth', 0):.0f}° "
                                f"El={metadata.get('elevation', 0):.0f}°", fontsize=10)
            axes_flat[i].axis('off')
        
        # Hide unused subplots
        for i in range(n_views, len(axes_flat)):
            axes_flat[i].axis('off')

        plt.suptitle('Multi-View Rendering', fontsize=14, fontweight='bold')

        # Save
        frame_path = self.output_dir / f'frame_{len(self.frames):03d}_query_renders.png'
        plt.tight_layout()
        plt.savefig(frame_path, dpi=100, bbox_inches='tight', facecolor='white')
        plt.close()

        self.frames.append({'path': frame_path, 'duration': duration})
        return frame_path

    def visualize_target_scene(self, rgb_path: Path, depth_path: Path,
                               duration: float = 2.0) -> Path:
        """
        Visualize target scene (RGB + depth).

        Args:
            rgb_path: Path to RGB image
            depth_path: Path to depth image
            duration: Duration in seconds

        Returns:
            Path to saved frame
        """
        rgb = np.array(Image.open(rgb_path))
        depth = np.array(Image.open(depth_path))

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # RGB
        axes[0].imshow(rgb)
        axes[0].set_title('RGB Image', fontsize=14, fontweight='bold')
        axes[0].axis('off')

        # Depth
        im = axes[1].imshow(depth, cmap='viridis')
        axes[1].set_title('Depth Map', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

        plt.suptitle('Target Scene', fontsize=16, fontweight='bold')

        # Save
        frame_path = self.output_dir / f'frame_{len(self.frames):03d}_target_scene.png'
        plt.tight_layout()
        plt.savefig(frame_path, dpi=100, bbox_inches='tight', facecolor='white')
        plt.close()

        self.frames.append({'path': frame_path, 'duration': duration})
        return frame_path

    def visualize_segmentation(self, rgb_path: Path, proposals: List[Dict],
                               duration: float = 2.0) -> Path:
        """
        Visualize SAM2 segmentation proposals.

        Args:
            rgb_path: Path to RGB image
            proposals: List of proposal dicts with 'mask' key
            duration: Duration in seconds

        Returns:
            Path to saved frame
        """
        rgb = np.array(Image.open(rgb_path))

        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Original image
        axes[0].imshow(rgb)
        axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
        axes[0].axis('off')

        # Segmentation overlay
        axes[1].imshow(rgb)

        # Overlay masks with different colors
        colors = plt.cm.Set3(np.linspace(0, 1, len(proposals)))
        for i, proposal in enumerate(proposals):
            mask = proposal.get('mask', proposal.get('segmentation'))
            if mask is None:
                continue

            # Create colored mask
            colored_mask = np.zeros((*mask.shape, 4))
            colored_mask[mask > 0] = colors[i]
            colored_mask[mask > 0, 3] = 0.5  # Alpha

            axes[1].imshow(colored_mask)

            # Draw bounding box if available
            if 'bbox' in proposal:
                x, y, w, h = proposal['bbox']
                rect = Rectangle((x, y), w, h, linewidth=2,
                               edgecolor=colors[i], facecolor='none')
                axes[1].add_patch(rect)
                axes[1].text(x, y - 5, f'#{i+1}', color=colors[i],
                           fontsize=12, fontweight='bold',
                           bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

        axes[1].set_title(f'Segmentation ({len(proposals)} proposals)',
                         fontsize=14, fontweight='bold')
        axes[1].axis('off')

        plt.suptitle('SAM2 Segmentation', fontsize=16, fontweight='bold')

        # Save
        frame_path = self.output_dir / f'frame_{len(self.frames):03d}_segmentation.png'
        plt.tight_layout()
        plt.savefig(frame_path, dpi=100, bbox_inches='tight', facecolor='white')
        plt.close()

        self.frames.append({'path': frame_path, 'duration': duration})
        return frame_path

    def visualize_pose_result(self, rgb_path: Path, mesh_path: Path,
                             pose_result: Dict, proposal_idx: int,
                             camera_intrinsics: Optional[np.ndarray] = None,
                             duration: float = 3.0) -> Path:
        """
        Visualize final pose estimation result by overlaying rendered mesh on scene.

        Args:
            rgb_path: Path to RGB image
            mesh_path: Path to query mesh
            pose_result: Pose estimation result with R, t
            proposal_idx: Which proposal was matched
            camera_intrinsics: 3x3 camera intrinsic matrix (optional)
            duration: Duration in seconds

        Returns:
            Path to saved frame
        """
        rgb = np.array(Image.open(rgb_path))

        if pose_result.get('success'):
            fig = plt.figure(figsize=(18, 6))

            # Panel 1: Original scene
            ax1 = fig.add_subplot(1, 3, 1)
            ax1.imshow(rgb)
            ax1.set_title('Target Scene', fontsize=14, fontweight='bold')
            ax1.axis('off')

            # Panel 2: 3D mesh overlay
            ax2 = fig.add_subplot(1, 3, 2)
            overlay = self._render_mesh_overlay(rgb, mesh_path, pose_result, camera_intrinsics)
            ax2.imshow(overlay)
            ax2.set_title('Estimated Pose (3D Mesh Rendered)', fontsize=14, fontweight='bold')
            ax2.axis('off')

            # Panel 3: Info box
            ax3 = fig.add_subplot(1, 3, 3)
            ax3.axis('off')
            info_text = (
                f"\u2713 Object Detected!\n\n"
                f"Proposal: #{proposal_idx + 1}\n"
                f"Inliers: {pose_result['num_inliers']}\n"
                f"RMSE: {pose_result['rmse']:.4f}m\n\n"
                f"Rotation (R):\n"
                f"{np.array2string(pose_result['R'], precision=3, suppress_small=True)}\n\n"
                f"Translation (t):\n"
                f"{np.array2string(pose_result['t'].flatten(), precision=3)}"
            )
            ax3.text(0.1, 0.5, info_text, fontsize=10, family='monospace',
                    verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

            success_msg = 'SUCCESS!'
            color = 'green'
        else:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.imshow(rgb)
            ax.text(rgb.shape[1] // 2, rgb.shape[0] // 2, '\u2717 Pose Estimation Failed',
                   fontsize=20, fontweight='bold', color='red',
                   ha='center', va='center',
                   bbox=dict(facecolor='black', alpha=0.7))
            ax.set_title('Pose Estimation Result', fontsize=14, fontweight='bold')
            ax.axis('off')
            success_msg = 'FAILED'
            color = 'red'

        plt.suptitle(f'6-DoF Pose Estimation: {success_msg}',
                    fontsize=16, fontweight='bold', color=color)

        frame_path = self.output_dir / f'frame_{len(self.frames):03d}_result.png'
        plt.tight_layout()
        plt.savefig(frame_path, dpi=100, bbox_inches='tight', facecolor='white')
        plt.close()

        self.frames.append({'path': frame_path, 'duration': duration})
        return frame_path

    def _render_mesh_overlay(self, rgb_image: np.ndarray, mesh_path: Path,
                             pose_result: Dict,
                             camera_intrinsics: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Render mesh using Open3D offscreen renderer and composite onto scene image.
        Uses defaultUnlit shader for reliable visibility regardless of normals/lighting.
        """
        h, w = rgb_image.shape[:2]

        if camera_intrinsics is not None:
            fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
            cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]
        else:
            # Match TargetPipeline's auto-estimated intrinsics (FOV 60°)
            # so pose and rendering use the same coordinate system
            fy = h / (2.0 * np.tan(np.radians(30.0)))
            fx = fy
            cx, cy = w / 2.0, h / 2.0

        # Normalize exactly as MeshNormalizer in QueryPipeline:
        # center by bbox center, scale so bbox diagonal = 0.2m
        mesh_o3d = o3d.io.read_triangle_mesh(str(mesh_path))
        mesh_o3d.compute_vertex_normals()
        # 1. Center at bbox center
        centroid = mesh_o3d.get_axis_aligned_bounding_box().get_center()
        mesh_o3d.translate(-centroid)
        # 2. Scale so diagonal = 0.2m (matches MeshNormalizer target_scale=0.2)
        extent = mesh_o3d.get_axis_aligned_bounding_box().get_extent()
        diagonal = float(np.linalg.norm(np.asarray(extent)))
        mesh_o3d.scale(0.2 / diagonal, center=np.array([0.0, 0.0, 0.0]))

        # Apply estimated pose: p_cam = R @ p_obj_normalized + t
        R = pose_result['R']
        t = pose_result['t'].flatten()
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        mesh_o3d.transform(T)

        # Bright green vertex color for unlit rendering
        mesh_o3d.paint_uniform_color([0.1, 0.9, 0.2])

        # Setup Open3D offscreen renderer (black background)
        renderer = o3d.visualization.rendering.OffscreenRenderer(w, h)
        renderer.scene.set_background([0.0, 0.0, 0.0, 1.0])

        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = 'defaultUnlit'  # Color independent of normals/lighting
        renderer.scene.add_geometry('mesh', mesh_o3d, mat)

        intrinsics = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)
        renderer.setup_camera(intrinsics, np.eye(4))

        render_raw = np.asarray(renderer.render_to_image())
        # Handle float [0,1] or uint8 [0,255] output
        if render_raw.dtype != np.uint8:
            render_img = (np.clip(render_raw, 0, 1) * 255).astype(np.uint8)
        else:
            render_img = render_raw
        # Mask: non-black pixels (anything rendered over the black bg)
        mask = (render_img.sum(axis=2) > 10).astype(np.float32)

        # Debug: show where the object center projects to
        t_flat = t.flatten()
        if t_flat[2] > 0:
            u_center = fx * t_flat[0] / t_flat[2] + cx
            v_center = fy * t_flat[1] / t_flat[2] + cy
            print(f"  [Debug] fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}")
            print(f"  [Debug] t={t_flat}, projected center=({u_center:.1f}, {v_center:.1f}) in {w}x{h}")
        print(f"  [Debug] mask pixels={mask.sum():.0f}")

        if mask.sum() < 10:
            print("  [3D Render] mesh off-screen, falling back to vertex projection")
            return self._vertex_projection_fallback(rgb_image, mesh_path, pose_result, fx, fy, cx, cy)

        mask_3 = np.stack([mask, mask, mask], axis=2)

        # Blend rendered mesh over original
        green_overlay = np.zeros_like(rgb_image, dtype=np.float32)
        green_overlay[:, :, 1] = 230
        result = (rgb_image.astype(np.float32) * (1 - mask_3 * 0.45) +
                  render_img.astype(np.float32) * mask_3 * 0.45 +
                  green_overlay * mask_3 * 0.25)
        result = np.clip(result, 0, 255).astype(np.uint8)

        ys, xs = np.where(mask > 0)
        print(f"  [Debug] mesh bbox in image: x=[{xs.min()},{xs.max()}] y=[{ys.min()},{ys.max()}]")
        cv2.rectangle(result, (xs.min(), ys.min()), (xs.max(), ys.max()), (0, 255, 80), 3)

        return result

    def _vertex_projection_fallback(self, rgb_image, mesh_path, pose_result, fx, fy, cx, cy):
        """Project mesh vertices to 2D and draw as dots when 3D render fails."""
        mesh = trimesh.load(str(mesh_path))
        verts = np.array(mesh.vertices)
        bbox_center = (verts.min(axis=0) + verts.max(axis=0)) / 2
        verts -= bbox_center
        bbox_size = verts.max(axis=0) - verts.min(axis=0)
        diagonal = np.linalg.norm(bbox_size)
        verts *= 0.2 / diagonal

        R = pose_result['R']
        t = pose_result['t'].flatten()
        verts_cam = (R @ verts.T).T + t
        print(f"  [Fallback] z range: [{verts_cam[:,2].min():.3f}, {verts_cam[:,2].max():.3f}]")

        valid = verts_cam[:, 2] > 0.01
        if valid.sum() == 0:
            return rgb_image

        vc = verts_cam[valid]
        xs = (fx * vc[:, 0] / vc[:, 2] + cx).astype(int)
        ys = (fy * vc[:, 1] / vc[:, 2] + cy).astype(int)
        h, w = rgb_image.shape[:2]
        inb = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        print(f"  [Fallback] {inb.sum()} vertices in image bounds")

        result = rgb_image.copy()
        for x, y in zip(xs[inb], ys[inb]):
            cv2.circle(result, (x, y), 4, (0, 255, 0), -1)
        if inb.sum() > 4:
            cv2.rectangle(result, (xs[inb].min(), ys[inb].min()),
                         (xs[inb].max(), ys[inb].max()), (0, 255, 0), 3)
        return result

    def visualize_comparison(self, results_geo: Dict, results_fused: Dict,
                            duration: float = 3.0) -> Path:
        """
        Visualize comparison between geometric and fused results.

        Args:
            results_geo: Geometric-only pose result
            results_fused: Fused pose result
            duration: Duration in seconds

        Returns:
            Path to saved frame
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Geometric-only
        geo_text = "Geometric-only\n(32-dim)\n\n"
        if results_geo.get('success'):
            geo_text += f"✓ Success\n"
            geo_text += f"Inliers: {results_geo['num_inliers']}\n"
            geo_text += f"RMSE: {results_geo['rmse']:.4f}m"
            geo_color = 'lightgreen'
        else:
            geo_text += f"✗ Failed\n{results_geo.get('reason', '')}"
            geo_color = 'lightcoral'

        axes[0].text(0.5, 0.5, geo_text, ha='center', va='center',
                    fontsize=16, bbox=dict(facecolor=geo_color, alpha=0.3, boxstyle='round,pad=1'))
        axes[0].set_xlim(0, 1)
        axes[0].set_ylim(0, 1)
        axes[0].axis('off')
        axes[0].set_title('Geometric Features', fontsize=14, fontweight='bold')

        # Fused
        fused_text = "Fused\n(32 geo + 1024 vis)\n\n"
        if results_fused.get('success'):
            fused_text += f"✓ Success\n"
            fused_text += f"Inliers: {results_fused['num_inliers']}\n"
            fused_text += f"RMSE: {results_fused['rmse']:.4f}m"
            fused_color = 'lightgreen'
        else:
            fused_text += f"✗ Failed\n{results_fused.get('reason', '')}"
            fused_color = 'lightcoral'

        axes[1].text(0.5, 0.5, fused_text, ha='center', va='center',
                    fontsize=16, bbox=dict(facecolor=fused_color, alpha=0.3, boxstyle='round,pad=1'))
        axes[1].set_xlim(0, 1)
        axes[1].set_ylim(0, 1)
        axes[1].axis('off')
        axes[1].set_title('Fused Features', fontsize=14, fontweight='bold')

        # Calculate improvement
        if results_geo.get('success') and results_fused.get('success'):
            inlier_improvement = results_fused['num_inliers'] - results_geo['num_inliers']
            rmse_improvement = results_geo['rmse'] - results_fused['rmse']

            improvement_text = f"\nVisual Features Impact:\n"
            improvement_text += f"Inliers: {inlier_improvement:+d}\n"
            improvement_text += f"RMSE: {rmse_improvement:+.4f}m"

            if inlier_improvement > 0:
                improvement_color = 'lightgreen'
                result_emoji = '🎉'
            elif inlier_improvement == 0:
                improvement_color = 'lightyellow'
                result_emoji = '➡️'
            else:
                improvement_color = 'lightcoral'
                result_emoji = '⚠️'

            fig.text(0.5, 0.02, improvement_text, ha='center', fontsize=14,
                    bbox=dict(facecolor=improvement_color, alpha=0.3, boxstyle='round,pad=0.5'))

        plt.suptitle('Geometric vs Fused Comparison', fontsize=16, fontweight='bold')

        # Save
        frame_path = self.output_dir / f'frame_{len(self.frames):03d}_comparison.png'
        plt.tight_layout(rect=[0, 0.1, 1, 1])
        plt.savefig(frame_path, dpi=100, bbox_inches='tight', facecolor='white')
        plt.close()

        self.frames.append({'path': frame_path, 'duration': duration})
        return frame_path

    def create_gif(self, output_path: Optional[Path] = None,
                  fps: int = None) -> Path:
        """
        Create GIF from all saved frames.

        Args:
            output_path: Output GIF path (default: output_dir/pipeline.gif)
            fps: Frames per second (if None, uses duration from frames)

        Returns:
            Path to created GIF
        """
        if output_path is None:
            output_path = self.output_dir / 'pipeline.gif'

        print(f"Creating GIF with {len(self.frames)} frames...")

        images = []
        durations = []
        
        # First pass: determine max dimensions
        max_height = 0
        max_width = 0
        for frame_info in self.frames:
            img = imageio.imread(frame_info['path'])
            max_height = max(max_height, img.shape[0])
            max_width = max(max_width, img.shape[1])

        # Second pass: load and resize all images
        from PIL import Image
        for frame_info in self.frames:
            img = imageio.imread(frame_info['path'])
            
            # Convert to PIL Image for resizing
            pil_img = Image.fromarray(img)
            
            # Create a new image with max dimensions (white background)
            new_img = Image.new('RGBA', (max_width, max_height), (255, 255, 255, 255))
            
            # Paste the original image centered
            x_offset = (max_width - img.shape[1]) // 2
            y_offset = (max_height - img.shape[0]) // 2
            new_img.paste(pil_img, (x_offset, y_offset))
            
            images.append(np.array(new_img))

            if fps is None:
                # Use duration from frame (in seconds)
                duration = frame_info.get('duration', 2.0)
                durations.append(duration)

        if fps is not None:
            # Constant FPS
            imageio.mimsave(output_path, images, fps=fps)
        else:
            # Variable duration per frame (convert seconds to milliseconds)
            durations_ms = [d * 1000 for d in durations]
            imageio.mimsave(output_path, images, duration=durations_ms, loop=0)

        print(f"✓ GIF saved to: {output_path}")
        return output_path
