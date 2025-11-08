"""
Displacement Field Loader for 4DCT Data Integration

This module handles loading NIfTI displacement fields and interpolating them
to mesh node positions for the ADAMSS lung simulation.
"""

import numpy as np
import nibabel as nib
from scipy.interpolate import RegularGridInterpolator
import meshio
import os

class DisplacementFieldLoader:
    def __init__(self):
        self.displacement_data = None
        self.affine_matrix = None
        self.voxel_size = None
        self.image_shape = None
        self.interpolators = None
        
    def load_nifti_displacement(self, nifti_path):
        """
        Load displacement field from NIfTI file
        
        Args:
            nifti_path: Path to .nii.gz displacement field file
            
        Returns:
            dict: Information about loaded displacement field
        """
        if not os.path.exists(nifti_path):
            raise FileNotFoundError(f"Displacement file not found: {nifti_path}")
            
        try:
            # Load NIfTI file
            nifti_img = nib.load(nifti_path)
            
            # Get displacement data (typically 4D: x, y, z, components)
            self.displacement_data = nifti_img.get_fdata()
            self.affine_matrix = nifti_img.affine
            self.image_shape = self.displacement_data.shape[:3]  # Spatial dimensions
            
            # Extract voxel size from affine matrix
            self.voxel_size = np.abs(np.diag(self.affine_matrix)[:3])
            
            print(f"Loaded displacement field:")
            print(f"  Shape: {self.displacement_data.shape}")
            print(f"  Voxel size: {self.voxel_size} mm")
            print(f"  Data range: [{self.displacement_data.min():.3f}, {self.displacement_data.max():.3f}]")
            
            # Create spatial coordinate grids for interpolation
            self._setup_interpolators()
            
            return {
                'shape': self.displacement_data.shape,
                'voxel_size': self.voxel_size,
                'data_range': [self.displacement_data.min(), self.displacement_data.max()],
                'affine': self.affine_matrix
            }
            
        except Exception as e:
            raise RuntimeError(f"Error loading NIfTI displacement field: {str(e)}")
    
    def _setup_interpolators(self):
        """Setup interpolators for each displacement component"""
        # Create coordinate grids in world coordinates (mm)
        nx, ny, nz = self.image_shape
        
        # Grid coordinates in voxel space
        x_voxel = np.arange(nx)
        y_voxel = np.arange(ny) 
        z_voxel = np.arange(nz)
        
        # Convert to world coordinates using affine matrix
        # Simplified assuming axis-aligned images
        x_world = x_voxel * self.voxel_size[0] + self.affine_matrix[0, 3]
        y_world = y_voxel * self.voxel_size[1] + self.affine_matrix[1, 3]
        z_world = z_voxel * self.voxel_size[2] + self.affine_matrix[2, 3]
        
        # Create interpolators for each displacement component
        self.interpolators = {}
        
        if len(self.displacement_data.shape) == 4:
            # 4D displacement field (x, y, z, components)
            n_components = self.displacement_data.shape[3]
            
            for comp in range(n_components):
                self.interpolators[f'u{comp}'] = RegularGridInterpolator(
                    (x_world, y_world, z_world),
                    self.displacement_data[:, :, :, comp],
                    method='linear',
                    bounds_error=False,
                    fill_value=0.0
                )
        else:
            # Handle other formats if needed
            raise ValueError(f"Unsupported displacement data shape: {self.displacement_data.shape}")
    
    def interpolate_to_mesh_nodes(self, mesh_nodes_mm):
        """
        Interpolate displacement field to mesh node positions
        
        Args:
            mesh_nodes_mm: Array of shape (N, 3) with mesh node positions in mm
            
        Returns:
            np.ndarray: Displacement vectors of shape (N, 3)
        """
        if self.interpolators is None:
            raise RuntimeError("Displacement field not loaded. Call load_nifti_displacement() first.")
        
        n_nodes = mesh_nodes_mm.shape[0]
        displacement_vectors = np.zeros((n_nodes, 3))
        
        # Interpolate each component
        for comp in range(3):
            if f'u{comp}' in self.interpolators:
                displacement_vectors[:, comp] = self.interpolators[f'u{comp}'](mesh_nodes_mm)
        
        # Convert from mm to meters (mesh is typically in meters)
        displacement_vectors *= 1e-3
        
        print(f"Interpolated displacements to {n_nodes} mesh nodes")
        print(f"  Displacement range: [{displacement_vectors.min():.6f}, {displacement_vectors.max():.6f}] m")
        print(f"  Mean displacement magnitude: {np.linalg.norm(displacement_vectors, axis=1).mean():.6f} m")
        
        return displacement_vectors

def load_and_interpolate_displacement(nifti_path, mesh_path):
    """
    Convenience function to load displacement field and interpolate to mesh
    
    Args:
        nifti_path: Path to NIfTI displacement field
        mesh_path: Path to mesh file
        
    Returns:
        tuple: (mesh_points, mesh_tets, mesh_labels, displacement_vectors)
    """
    # Load mesh
    if not os.path.exists(mesh_path):
        raise FileNotFoundError(f"Mesh file not found: {mesh_path}")
    
    mesh = meshio.read(mesh_path)
    pts = mesh.points  # Already in mm from mesh file
    
    # Find tetrahedra
    tid = next((i for i, c in enumerate(mesh.cells) if c.type == "tetra"), None)
    if tid is None:
        raise ValueError("No tetrahedra found in mesh file")
    
    tets = mesh.cells[tid].data
    
    # Get labels
    if "c_labels" not in mesh.cell_data:
        print("Warning: No cell labels found, using default label 0")
        lbls = np.zeros(tets.shape[0], dtype=np.int32)
    else:
        lbls = mesh.cell_data["c_labels"][tid]
    
    # Load and interpolate displacement field
    loader = DisplacementFieldLoader()
    loader.load_nifti_displacement(nifti_path)
    displacement_vectors = loader.interpolate_to_mesh_nodes(pts)
    
    print(f"Loaded mesh: {pts.shape[0]} nodes, {tets.shape[0]} tetrahedra")
    
    return pts, tets, lbls, displacement_vectors

if __name__ == "__main__":
    # Test loading with non-zero displacement (T00 to T50 - significant respiratory phase difference)
    displacement_path = "/home/haozhe/Lung_Mesh_Segment/data/Case1Pack/CorrField/case1_T00_T50.nii.gz"
    mesh_path = "/home/haozhe/Lung_Mesh_Segment/data/Case1Pack/pygalmesh/case1_T00_lung_regions_11.xdmf"
    
    try:
        pts, tets, lbls, displ = load_and_interpolate_displacement(displacement_path, mesh_path)
        print("✓ Successfully loaded and interpolated displacement field")
        
        # Print some statistics
        displ_magnitudes = np.linalg.norm(displ, axis=1)
        print(f"Displacement statistics:")
        print(f"  Max magnitude: {displ_magnitudes.max():.6f} m")
        print(f"  Mean magnitude: {displ_magnitudes.mean():.6f} m")
        print(f"  Std magnitude: {displ_magnitudes.std():.6f} m")
        print(f"  Non-zero nodes: {np.sum(displ_magnitudes > 1e-9)}/{len(displ_magnitudes)}")
        
        # Save test displacement for verification
        np.save("test_displacement_field.npy", displ)
        print("Saved displacement field to test_displacement_field.npy")
        
    except Exception as e:
        print(f"Error: {e}")