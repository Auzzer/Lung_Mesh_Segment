"""
Utility functions for mesh operations in the lung CT pipeline
"""

import numpy as np
import meshio
import nibabel as nib
from pathlib import Path
from typing import Tuple, Optional, Union
from skimage import measure
import logging

logger = logging.getLogger(__name__)

def load_mesh_safe(mesh_file: Union[str, Path]) -> Optional[meshio.Mesh]:
    """Safely load a mesh file with error handling"""
    try:
        mesh = meshio.read(mesh_file)
        logger.info(f"Loaded mesh: {mesh_file} ({len(mesh.points)} vertices)")
        return mesh
    except Exception as e:
        logger.error(f"Failed to load mesh {mesh_file}: {e}")
        return None

def extract_lung_surface(mask_data: np.ndarray, affine: np.ndarray, 
                        level: float = 0.5, step_size: int = 1) -> meshio.Mesh:
    """
    Extract lung surface using marching cubes algorithm
    
    Args:
        mask_data: 3D binary mask array
        affine: 4x4 affine transformation matrix
        level: Iso-surface value for marching cubes
        step_size: Step size for marching cubes (reduce for higher resolution)
    
    Returns:
        meshio.Mesh: Surface mesh
    """
    # Apply marching cubes with step size
    vertices, faces, normals, values = measure.marching_cubes(
        mask_data, level=level, step_size=step_size
    )
    
    # Transform vertices to physical coordinates
    vertices_homo = np.column_stack([vertices, np.ones(len(vertices))])
    vertices_phys = (affine @ vertices_homo.T).T[:, :3]
    
    # Create mesh with normals
    mesh = meshio.Mesh(
        points=vertices_phys,
        cells=[("triangle", faces)],
        point_data={"normals": normals},
        cell_data={}
    )
    
    return mesh

def smooth_mesh(mesh: meshio.Mesh, iterations: int = 5) -> meshio.Mesh:
    """
    Apply Laplacian smoothing to mesh
    
    Args:
        mesh: Input mesh
        iterations: Number of smoothing iterations
    
    Returns:
        meshio.Mesh: Smoothed mesh
    """
    points = mesh.points.copy()
    triangles = mesh.cells[0].data
    
    for _ in range(iterations):
        # Build adjacency information
        adjacency = build_vertex_adjacency(triangles, len(points))
        
        # Apply Laplacian smoothing
        new_points = np.zeros_like(points)
        for i in range(len(points)):
            neighbors = adjacency[i]
            if len(neighbors) > 0:
                new_points[i] = np.mean(points[neighbors], axis=0)
            else:
                new_points[i] = points[i]
        
        points = new_points
    
    # Create smoothed mesh
    smoothed_mesh = meshio.Mesh(
        points=points,
        cells=mesh.cells,
        point_data=mesh.point_data,
        cell_data=mesh.cell_data
    )
    
    return smoothed_mesh

def build_vertex_adjacency(triangles: np.ndarray, num_vertices: int) -> dict:
    """Build vertex adjacency list from triangles"""
    adjacency = {i: set() for i in range(num_vertices)}
    
    for triangle in triangles:
        for i in range(3):
            for j in range(3):
                if i != j:
                    adjacency[triangle[i]].add(triangle[j])
    
    return {k: list(v) for k, v in adjacency.items()}

def calculate_mesh_quality(mesh: meshio.Mesh) -> dict:
    """
    Calculate mesh quality metrics
    
    Args:
        mesh: Input mesh
    
    Returns:
        dict: Quality metrics
    """
    points = mesh.points
    # Handle different mesh cell types
    triangles = None
    for cell in mesh.cells:
        if cell.type == "triangle":
            triangles = cell.data
            break
    
    if triangles is None:
        return {"error": "No triangular cells found in mesh"}
    
    # Calculate triangle areas
    areas = []
    for triangle in triangles:
        p1, p2, p3 = points[triangle[0]], points[triangle[1]], points[triangle[2]]
        area = 0.5 * np.linalg.norm(np.cross(p2 - p1, p3 - p1))
        areas.append(area)
    
    areas = np.array(areas)
    
    # Calculate edge lengths
    edge_lengths = []
    for triangle in triangles:
        p1, p2, p3 = points[triangle[0]], points[triangle[1]], points[triangle[2]]
        edge_lengths.extend([
            np.linalg.norm(p2 - p1),
            np.linalg.norm(p3 - p2),
            np.linalg.norm(p1 - p3)
        ])
    
    edge_lengths = np.array(edge_lengths)
    
    # Calculate angles
    angles = []
    for triangle in triangles:
        p1, p2, p3 = points[triangle[0]], points[triangle[1]], points[triangle[2]]
        v1, v2, v3 = p2 - p1, p3 - p2, p1 - p3
        
        # Calculate angles using dot product
        angle1 = np.arccos(np.clip(np.dot(v1, -v3) / (np.linalg.norm(v1) * np.linalg.norm(v3)), -1, 1))
        angle2 = np.arccos(np.clip(np.dot(v2, -v1) / (np.linalg.norm(v2) * np.linalg.norm(v1)), -1, 1))
        angle3 = np.arccos(np.clip(np.dot(v3, -v2) / (np.linalg.norm(v3) * np.linalg.norm(v2)), -1, 1))
        
        angles.extend([angle1, angle2, angle3])
    
    angles = np.array(angles)
    
    quality_metrics = {
        "num_vertices": len(points),
        "num_triangles": len(triangles),
        "total_area": np.sum(areas),
        "mean_area": np.mean(areas),
        "min_area": np.min(areas),
        "max_area": np.max(areas),
        "mean_edge_length": np.mean(edge_lengths),
        "min_edge_length": np.min(edge_lengths),
        "max_edge_length": np.max(edge_lengths),
        "mean_angle": np.mean(angles),
        "min_angle": np.min(angles),
        "max_angle": np.max(angles),
    }
    
    return quality_metrics

def remesh_surface(mesh: meshio.Mesh, target_edge_length: float = 2.0) -> meshio.Mesh:
    """
    Remesh surface to target edge length (requires pymeshlab)
    
    Args:
        mesh: Input mesh
        target_edge_length: Target edge length for remeshing
    
    Returns:
        meshio.Mesh: Remeshed surface
    """
    try:
        import pymeshlab
        
        # Create MeshSet
        ms = pymeshlab.MeshSet()
        
        # Create mesh from points and faces
        pymesh = pymeshlab.Mesh(
            vertex_matrix=mesh.points,
            face_matrix=mesh.cells[0].data
        )
        ms.add_mesh(pymesh)
        
        # Apply remeshing
        ms.apply_filter('meshing_isotropic_explicit_remeshing', 
                       targetlen=pymeshlab.Percentage(target_edge_length))
        
        # Get result
        result_mesh = ms.current_mesh()
        
        # Create meshio mesh
        remeshed = meshio.Mesh(
            points=result_mesh.vertex_matrix(),
            cells=[("triangle", result_mesh.face_matrix())],
            point_data={},
            cell_data={}
        )
        
        return remeshed
        
    except ImportError:
        logger.warning("pymeshlab not available, returning original mesh")
        return mesh
    except Exception as e:
        logger.error(f"Remeshing failed: {e}")
        return mesh

def apply_displacement_field(mesh: meshio.Mesh, displacement_field: np.ndarray, 
                           affine: np.ndarray) -> meshio.Mesh:
    """
    Apply displacement field to mesh vertices
    
    Args:
        mesh: Input mesh
        displacement_field: 4D array (x, y, z, 3) with displacement vectors
        affine: 4x4 affine transformation matrix
    
    Returns:
        meshio.Mesh: Displaced mesh
    """
    from scipy.interpolate import RegularGridInterpolator
    
    # Get mesh vertices
    vertices = mesh.points.copy()
    
    # Transform vertices to voxel coordinates
    vertices_homo = np.column_stack([vertices, np.ones(len(vertices))])
    voxel_coords = (np.linalg.inv(affine) @ vertices_homo.T).T[:, :3]
    
    # Create interpolators for each displacement component
    x_grid = np.arange(displacement_field.shape[0])
    y_grid = np.arange(displacement_field.shape[1])
    z_grid = np.arange(displacement_field.shape[2])
    
    displacements = np.zeros((len(vertices), 3))
    
    for i in range(3):  # x, y, z components
        interpolator = RegularGridInterpolator(
            (x_grid, y_grid, z_grid),
            displacement_field[:, :, :, i],
            method='cubic',
            bounds_error=False,
            fill_value=0.0
        )
        displacements[:, i] = interpolator(voxel_coords)
    
    # Apply displacements
    displaced_vertices = vertices + displacements
    
    # Create new mesh
    displaced_mesh = meshio.Mesh(
        points=displaced_vertices,
        cells=mesh.cells,
        point_data=mesh.point_data.copy(),
        cell_data=mesh.cell_data.copy()
    )
    
    # Add displacement as point data
    displaced_mesh.point_data['displacement'] = displacements
    displaced_mesh.point_data['displacement_magnitude'] = np.linalg.norm(displacements, axis=1)
    
    return displaced_mesh

def save_mesh_with_metadata(mesh: meshio.Mesh, output_file: Path, 
                          metadata: dict = None):
    """
    Save mesh with optional metadata
    
    Args:
        mesh: Mesh to save
        output_file: Output file path
        metadata: Optional metadata dictionary
    """
    # Save the mesh
    try:
        mesh.write(output_file)
        logger.info(f"Saved mesh: {output_file}")
    except Exception as e:
        logger.error(f"Failed to save mesh {output_file}: {e}")
        return
    
    # Save metadata if provided
    if metadata:
        metadata_file = output_file.with_suffix('.json')
        import json
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved metadata: {metadata_file}")

def compute_mesh_volume(mesh: meshio.Mesh) -> float:
    """
    Compute volume of a closed mesh using divergence theorem
    
    Args:
        mesh: Input mesh (should be closed)
    
    Returns:
        float: Volume
    """
    points = mesh.points
    # Handle different mesh cell types
    triangles = None
    for cell in mesh.cells:
        if cell.type == "triangle":
            triangles = cell.data
            break
    
    if triangles is None:
        return 0.0
    
    volume = 0.0
    for triangle in triangles:
        p1, p2, p3 = points[triangle[0]], points[triangle[1]], points[triangle[2]]
        
        # Vector from origin to vertices
        v1, v2, v3 = p1, p2, p3
        
        # Cross product for triangle area and normal
        cross = np.cross(v2 - v1, v3 - v1)
        
        # Contribution to volume (1/6 * dot product with centroid)
        centroid = (v1 + v2 + v3) / 3
        volume += np.dot(centroid, cross) / 6
    
    return abs(volume)

def validate_mesh(mesh: meshio.Mesh) -> dict:
    """
    Validate mesh and return diagnostic information
    
    Args:
        mesh: Input mesh
    
    Returns:
        dict: Validation results
    """
    validation = {
        "is_valid": True,
        "warnings": [],
        "errors": []
    }
    
    points = mesh.points
    # Handle different mesh cell types
    triangles = None
    for cell in mesh.cells:
        if cell.type == "triangle":
            triangles = cell.data
            break
    
    if triangles is None:
        validation["errors"].append("No triangular cells found in mesh")
        validation["is_valid"] = False
        return validation
    
    # Check for duplicate vertices
    unique_points = np.unique(points, axis=0)
    if len(unique_points) != len(points):
        validation["warnings"].append(f"Duplicate vertices found: {len(points) - len(unique_points)}")
    
    # Check for degenerate triangles
    degenerate_count = 0
    for triangle in triangles:
        p1, p2, p3 = points[triangle[0]], points[triangle[1]], points[triangle[2]]
        area = 0.5 * np.linalg.norm(np.cross(p2 - p1, p3 - p1))
        if area < 1e-10:
            degenerate_count += 1
    
    if degenerate_count > 0:
        validation["errors"].append(f"Degenerate triangles found: {degenerate_count}")
        validation["is_valid"] = False
    
    # Check for invalid triangle indices
    max_vertex_index = np.max(triangles)
    if max_vertex_index >= len(points):
        validation["errors"].append(f"Invalid triangle indices: max index {max_vertex_index} >= {len(points)}")
        validation["is_valid"] = False
    
    # Check mesh manifoldness (simplified check)
    edge_count = {}
    for triangle in triangles:
        for i in range(3):
            edge = tuple(sorted([triangle[i], triangle[(i+1)%3]]))
            edge_count[edge] = edge_count.get(edge, 0) + 1
    
    non_manifold_edges = sum(1 for count in edge_count.values() if count > 2)
    if non_manifold_edges > 0:
        validation["warnings"].append(f"Non-manifold edges found: {non_manifold_edges}")
    
    return validation