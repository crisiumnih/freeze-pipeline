#!/usr/bin/env python3
"""
Mesh sampling server using Open3D.
Used to sample point clouds from meshes in isolated environment.
"""
import sys
import json
import numpy as np
import open3d as o3d
from pathlib import Path


def sample_mesh(mesh_path, num_points):
    """Sample points uniformly from mesh."""
    print(f"Loading mesh: {mesh_path}", file=sys.stderr)
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))

    if not mesh.has_vertices():
        raise ValueError(f"Failed to load mesh from {mesh_path}")

    print(f"Sampling {num_points} points...", file=sys.stderr)
    pcd = mesh.sample_points_uniformly(number_of_points=num_points)
    points = np.asarray(pcd.points, dtype=np.float32)

    print(f"Sampled {len(points)} points", file=sys.stderr)
    return points


def main():
    if len(sys.argv) < 2:
        print("Usage: python mesh_sampler.py <request_json_path>", file=sys.stderr)
        sys.exit(1)

    request_path = sys.argv[1]

    with open(request_path) as f:
        request = json.load(f)

    mesh_path = request['mesh_path']
    num_points = request.get('num_points', 10000)
    output_path = request['output_path']

    # Sample points
    points = sample_mesh(mesh_path, num_points)

    # Save as numpy array
    np.save(output_path, points)

    # Return result
    result = {
        'output_path': output_path,
        'num_points': len(points),
        'shape': list(points.shape)
    }

    print(json.dumps(result, default=str))


if __name__ == '__main__':
    main()
