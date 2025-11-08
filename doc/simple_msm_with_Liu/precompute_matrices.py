"""
Pre-compute A_e, S_e matrices and save them for SimpleMSM
"""

import taichi as ti
import numpy as np
import meshio
import os
import SimpleITK as sitk
from typing import Tuple

ti.init(arch=ti.gpu)

def load_mesh_from_file(xdmf_file: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load mesh from XDMF file"""
    mesh = meshio.read(xdmf_file)
    print(f"Loaded mesh: {len(mesh.points)} points, {len(mesh.cells)} cell blocks")
    
    points = mesh.points
    tetrahedra = None
    labels = None
    
    for cell_block in mesh.cells:
        if cell_block.type == "tetra":
            tetrahedra = cell_block.data
            break
    
    if "label" in mesh.cell_data:
        for i, cell_block in enumerate(mesh.cells):
            if cell_block.type == "tetra":
                labels = mesh.cell_data["label"][i]
                break
    
    if labels is None:
        labels = np.zeros(len(tetrahedra), dtype=np.int32)
    
    print(f"Extracted: {len(points)} vertices, {len(tetrahedra)} tetrahedra")
    return points, tetrahedra, labels

def interpolate_displacement_field(vertices_mm: np.ndarray, displacement_path: str) -> np.ndarray:
    """Interpolate displacement field at mesh vertices (mm) in image physical space."""
    img = sitk.ReadImage(displacement_path)
    
    # Check if it's a vector or scalar image  
    n_components = img.GetNumberOfComponentsPerPixel()
    print(f"Displacement image has {n_components} component(s)")
    
    if n_components == 3:
        # True vector displacement field
        interp = sitk.VectorLinearInterpolateImageFunction(img, False, sitk.sitkLinear)
        
        disp = np.zeros_like(vertices_mm, dtype=np.float64)
        for i, p in enumerate(vertices_mm.astype(np.float64)):
            idx = img.TransformPhysicalPointToContinuousIndex(tuple(p))
            v = np.array(interp.EvaluateAtContinuousIndex(idx), dtype=np.float64)
            disp[i] = v
            
    elif n_components == 1:
        # Scalar field - use as displacement magnitude in random directions
        # Use SimpleITK's resample with linear interpolation
        disp = np.zeros_like(vertices_mm, dtype=np.float64)
        np.random.seed(42)  # For reproducibility
        
        # Get the image array for direct sampling
        img_array = sitk.GetArrayFromImage(img)
        origin = np.array(img.GetOrigin())[:3]  # Take only first 3 components
        spacing = np.array(img.GetSpacing())[:3]  # Take only first 3 components
        
        print(f"Image shape: {img_array.shape}, origin: {origin}, spacing: {spacing}")
        
        for i, p in enumerate(vertices_mm.astype(np.float64)):
            # Simple nearest neighbor for now (could improve with linear interpolation)
            physical_idx = (p - origin) / spacing
            idx = np.round(physical_idx).astype(int)
            
            # Handle different dimensionalities
            if img_array.ndim == 3:
                # 3D image: z,y,x order
                idx = np.clip(idx, [0, 0, 0], np.array(img_array.shape[::-1]) - 1)
                magnitude = float(img_array[idx[2], idx[1], idx[0]])
            elif img_array.ndim == 4:
                # 4D image: likely t,z,y,x - take first time point
                idx = np.clip(idx, [0, 0, 0], np.array(img_array.shape[1:][::-1]) - 1)
                magnitude = float(img_array[0, idx[2], idx[1], idx[0]])
            else:
                raise ValueError(f"Unsupported image dimensionality: {img_array.ndim}")
            
            # Generate random unit direction
            direction = np.random.normal(0, 1, 3)
            direction /= np.linalg.norm(direction) + 1e-12
            
            disp[i] = magnitude * direction
            
    else:
        raise ValueError(f"Unsupported displacement field with {n_components} components")
    
    print(f"Interpolated displacement field: {disp.shape} (keeping mm units)")
    return disp

def load_mesh_with_displacement(mesh_path: str, displacement_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load mesh and corresponding displacement field"""
    pts, tets, lbls = load_mesh_from_file(mesh_path)
    displacement_vectors = interpolate_displacement_field(pts, displacement_path)
    return pts, tets, lbls, displacement_vectors

@ti.data_oriented
class MatrixPrecomputer:
    """Pre-compute A_e, S_e matrices using Taichi GPU acceleration"""
    
    def __init__(self, pts_np: np.ndarray, tets_np: np.ndarray, 
                 displacement_vectors: np.ndarray):
        
        self.N = pts_np.shape[0]  # number of vertices
        self.M = tets_np.shape[0]  # number of tetrahedra
        
        # Store tetrahedral data for boundary detection
        self.tets_np = tets_np.copy()
        
        # Extract edges from tetrahedra
        self._extract_edges(tets_np)
        
        print(f"Matrix pre-computer: {self.N} vertices, {self.M} tetrahedra, {self.E} edges")
        
        # Initialize Taichi fields
        self.vertices = ti.Vector.field(3, ti.f64, shape=self.N)
        self.edges = ti.Vector.field(2, ti.i32, shape=self.E)
        self.rest_lengths = ti.field(ti.f64, shape=self.E)
        self.registration_directions = ti.Vector.field(3, ti.f64, shape=self.N)
        self.displacement_vectors = ti.Vector.field(3, ti.f64, shape=self.N)
        
        # COO storage for A_e matrix (2 entries per edge)
        self.A_rows = ti.field(ti.i32, shape=2*self.E)  # row indices
        self.A_cols = ti.field(ti.i32, shape=2*self.E)  # col indices  
        self.A_vals = ti.field(ti.f64, shape=2*self.E)  # values (+1, -1)
        
        # Edge-wise stiffness values
        self.edge_stiffness = ti.field(ti.f64, shape=self.E)  # k_e for each edge
        
        # Copy data to GPU
        self._copy_data_to_gpu(pts_np, displacement_vectors)
        
    def _extract_edges(self, tets_np: np.ndarray):
        """Extract unique edges from tetrahedral mesh"""
        edges_set = set()
        for tet in tets_np:
            for i in range(4):
                for j in range(i+1, 4):
                    edge = tuple(sorted([tet[i], tet[j]]))
                    edges_set.add(edge)
        
        self.edges_list = list(edges_set)
        self.E = len(self.edges_list)
        print(f"Extracted {self.E} unique edges from {self.M} tetrahedra")
    
    def _copy_data_to_gpu(self, pts_np: np.ndarray, displacement_vectors: np.ndarray):
        """Copy data to Taichi GPU fields"""
        # Keep vertices in mm 
        for i in range(self.N):
            self.vertices[i] = pts_np[i].astype(np.float64)
            self.displacement_vectors[i] = displacement_vectors[i].astype(np.float64)
            
            # Compute registration directions with improved stability
            disp = displacement_vectors[i]
            eps = 1e-9
            n = np.linalg.norm(disp)
            if n > eps:
                self.registration_directions[i] = disp / n
            else:
                self.registration_directions[i] = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        
        for e in range(self.E):
            i, j = self.edges_list[e]
            self.edges[e] = [i, j]
    
    @ti.kernel
    def compute_rest_lengths(self):
        """Compute rest lengths for all edges on GPU"""
        for e in range(self.E):
            i = self.edges[e][0]
            j = self.edges[e][1]
            edge_vec = self.vertices[i] - self.vertices[j]
            self.rest_lengths[e] = ti.max(edge_vec.norm(), 1e-6)
    
    @ti.kernel
    def build_A_e_coo(self):
        """Build incidence matrix A_e in COO format on GPU"""
        for e in range(self.E):
            i = self.edges[e][0]  # first vertex of edge
            j = self.edges[e][1]  # second vertex of edge
            
            # First entry: (e, i) = +1.0
            self.A_rows[2*e] = e
            self.A_cols[2*e] = i  
            self.A_vals[2*e] = 1.0
            
            # Second entry: (e, j) = -1.0
            self.A_rows[2*e + 1] = e
            self.A_cols[2*e + 1] = j
            self.A_vals[2*e + 1] = -1.0
    
    def generate_random_stiffness_matrix(self, base_stiffness: float = 1000.0):
        """Generate random edge-wise stiffness values (outside Taichi scope)"""
        np.random.seed(42)  # For reproducibility
        
        # Random stiffness values with some variation
        stiffness_values = np.random.lognormal(
            mean=np.log(base_stiffness), 
            sigma=0.5,  # Moderate variation
            size=self.E
        ).astype(np.float64)
        
        # Ensure reasonable bounds
        stiffness_values = np.clip(stiffness_values, 
                                 base_stiffness * 0.1,   # min: 10% of base
                                 base_stiffness * 5.0)    # max: 500% of base
        
        # Copy to Taichi field
        for e in range(self.E):
            self.edge_stiffness[e] = stiffness_values[e]
        
        print(f"Generated random stiffness matrix:")
        print(f"  Min: {stiffness_values.min():.1f}")
        print(f"  Max: {stiffness_values.max():.1f}")
        print(f"  Mean: {stiffness_values.mean():.1f}")
        
        return stiffness_values
    
    def compute_matrices(self):
        """Compute A_e COO triplets, rest lengths, and stiffness matrix"""
        print("Computing matrices on GPU...")
        
        # Compute rest lengths
        self.compute_rest_lengths()
        
        # Build incidence matrix A_e in COO format
        self.build_A_e_coo()
        
        # Generate random stiffness matrix (outside Taichi scope)
        self.generate_random_stiffness_matrix()
        
        print("GPU computation complete")
    
    def save_matrices(self, output_dir: str):
        """Save all pre-computed data to files"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Compute matrices on GPU
        self.compute_matrices()
        
        # Copy results back to numpy and save
        edges_array = np.array(self.edges_list, dtype=np.int32)
        vertices_np = self.vertices.to_numpy()
        rest_lengths_np = self.rest_lengths.to_numpy()
        registration_directions_np = self.registration_directions.to_numpy()
        displacement_vectors_np = self.displacement_vectors.to_numpy()
        
        # Copy COO triplets from GPU
        A_rows_np = self.A_rows.to_numpy()
        A_cols_np = self.A_cols.to_numpy() 
        A_vals_np = self.A_vals.to_numpy()
        
        # Copy stiffness matrix from GPU
        edge_stiffness_np = self.edge_stiffness.to_numpy()
        
        # Save core data
        np.save(os.path.join(output_dir, 'edges.npy'), edges_array)
        np.save(os.path.join(output_dir, 'vertices.npy'), vertices_np)
        np.save(os.path.join(output_dir, 'rest_lengths.npy'), rest_lengths_np)
        np.save(os.path.join(output_dir, 'registration_directions.npy'), registration_directions_np)
        np.save(os.path.join(output_dir, 'displacement_vectors.npy'), displacement_vectors_np)
        
        # Save A_e matrix as COO triplets (much more memory efficient)
        np.save(os.path.join(output_dir, 'A_rows.npy'), A_rows_np)
        np.save(os.path.join(output_dir, 'A_cols.npy'), A_cols_np)
        np.save(os.path.join(output_dir, 'A_vals.npy'), A_vals_np)
        
        # Save edge-wise stiffness matrix
        np.save(os.path.join(output_dir, 'edge_stiffness.npy'), edge_stiffness_np)
        
        # Save tetrahedral data for boundary detection
        np.save(os.path.join(output_dir, 'tetrahedra.npy'), self.tets_np)
        
        # Save metadata
        metadata = {'N': self.N, 'E': self.E, 'M': self.M}
        np.save(os.path.join(output_dir, 'metadata.npy'), metadata)
        
        print(f"Saved matrices to {output_dir}")
        print(f"A_e COO: {len(A_rows_np)} triplets ({self.E} x {self.N})")
        print(f"Rest lengths: min={rest_lengths_np.min():.6f}, max={rest_lengths_np.max():.6f} mm")
        print("Note: S_e is identity matrix, not saved (generated dynamically)")
        


def main():
    """Main function to pre-compute A_e, matrices"""
    path = os.path.dirname(os.path.abspath(__file__))
    mesh_path = os.path.join(path, "./data/Case1Pack/pygalmesh/case1_T00_lung_regions_11.xdmf")
    displacement_path = os.path.join(path, "./data/Case1Pack/CorrField/case1_T00_T50.nii.gz")
    output_dir = os.path.join(path, "precomputed_matrices")
    
    print("=== Loading Mesh and 4DCT Displacement Field ===")
    pts, tets, lbls, displacement_vectors = load_mesh_with_displacement(mesh_path, displacement_path)
    
    print("=== Pre-computing A_e, S_e Matrices with Taichi GPU ===")
    precomputer = MatrixPrecomputer(pts, tets, displacement_vectors)
    precomputer.save_matrices(output_dir)
    
if __name__ == "__main__":
    main()