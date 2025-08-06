"""
Deformation Processor for Mesh and Image Registration

This script processes mesh and image registration input by:
1. Calculating deformation axes from SVD decomposition 
2. Computing particle deformation coefficients
3. Saving results for further analysis

output will be used for the lung mesh segmentation project's static equilibrium solver.
"""

import taichi as ti
import meshio
import numpy as np

ti.init(arch=ti.cuda)

# Tolerance constants
EPS_ZERO = 1e-10
EPS_DISTANCE = 1e-6
EPS_PARALLEL = 1e-10
EPS_BARYCENTRIC = 1e-8

# Physical constants
DEFAULT_RHO = {
    0: 1_050.0,   # normal tissue (kg/m³)
    1:   250.0,   # air-rich
    2: 1_100.0,   # fibrotic
}

@ti.data_oriented
class DeformationProcessor:
    def __init__(self, pts_np, tets_np, labels_np):
        self.N = pts_np.shape[0]      # #nodes
        self.M = tets_np.shape[0]     # #tets
        
        # Core data fields
        self.x = ti.Vector.field(3, ti.f64, shape=self.N)           # current positions
        self.tets = ti.Vector.field(4, ti.i32, shape=self.M)        # tetrahedron indices
        self.label = ti.field(ti.i32, shape=self.M)                 # region labels
        
        # Initial configuration
        self.initial_positions = ti.Vector.field(3, ti.f64, shape=self.N)
        
        # Displacement and deformation fields
        self.displacement_field = ti.Vector.field(3, ti.f64, shape=self.N)        # u_i from 4DCT
        self.deformed_positions = ti.Vector.field(3, ti.f64, shape=self.N)        # x_i = X_i + u_i
        self.deformation_gradient = ti.Matrix.field(3, 3, ti.f64, shape=self.M)   # alpha matrix per tet
        self.reference_edge_matrix = ti.Matrix.field(3, 3, ti.f64, shape=self.M)  # D_m matrix per tet
        self.deformed_edge_matrix = ti.Matrix.field(3, 3, ti.f64, shape=self.M)   # d_x matrix per tet
        
        # SVD components for anisotropy axis extraction
        self.svd_U = ti.Matrix.field(3, 3, ti.f64, shape=self.M)             # U matrix from SVD
        self.svd_sigma = ti.Vector.field(3, ti.f64, shape=self.M)            # Singular values
        self.svd_V = ti.Matrix.field(3, 3, ti.f64, shape=self.M)             # V matrix from SVD
        self.principal_stretches = ti.Vector.field(3, ti.f64, shape=self.M)  # Principal stretch values
        self.anisotropy_axes = ti.Vector.field(3, ti.f64, shape=(self.M, 3)) # 3 axes per tet
        
        # Particle deformation coefficients
        self.deformation_coeffs = ti.Matrix.field(3, 4, ti.f64, shape=self.M)    # 3x4 matrix per tet
        self.stretch_anisotropy = ti.field(ti.f64, shape=self.M)                # anisotropy measure
        
        # Mass-spring system preprocessing fields
        self.tets = ti.Vector.field(4, ti.i32, shape=self.M)                    # tetrahedron indices
        self.label = ti.field(ti.i32, shape=self.M)                             # region labels
        self.rho = ti.field(ti.f64, shape=self.M)                               # density per tet
        self.vol = ti.field(ti.f64, shape=self.M)                               # volume per tet
        self.mass = ti.field(ti.f64, shape=self.N)                              # lumped mass per node
        
        # Intersection and coefficient fields
        self.intersection_points = ti.Vector.field(3, ti.f64, shape=(self.M, 6))  # 6 points per tet
        self.intersection_valid = ti.field(ti.i32, shape=(self.M, 6))             # validity flag
        self.intersection_face = ti.field(ti.i32, shape=(self.M, 6))              # face index
        self.C_k = ti.field(ti.f64, shape=(self.M, 4, 6))                        # Coefficient matrix
        
        # Spring system fields
        self.rest_lengths = ti.field(ti.f64, shape=(self.M, 3))                  # rest lengths per axis
        self.stiffness = ti.field(ti.f64, shape=(self.M, 3))                     # stiffness per axis
        self.torsion_stiffness = ti.field(ti.f64, shape=(self.M, 3))             # torsion stiffness
        self.rest_cos_angles = ti.field(ti.f64, shape=(self.M, 3))               # rest cosine angles
        
        # Boundary detection fields
        self.boundary_nodes = ti.field(ti.i32, shape=self.N)                     # boundary node flags
        
        # Initial configuration storage
        self.initial_positions = ti.Vector.field(3, ti.f64, shape=self.N)
        self.initial_intersection_points = ti.Vector.field(3, ti.f64, shape=(self.M, 6))
        self.initial_axis_vectors = ti.Vector.field(3, ti.f64, shape=(self.M, 3))
        
        # Flags
        self.is_degenerate = ti.field(ti.i32, shape=self.M)
        self.is_near_rigid = ti.field(ti.i32, shape=self.M)
        
        # Initialize data
        self._initialize_data(pts_np, tets_np, labels_np)
        
    def _initialize_data(self, pts_np, tets_np, labels_np):
        """Initialize data from numpy arrays"""
        # Convert positions from mm to m
        self.x.from_numpy(pts_np.astype(np.float64))
        self.initial_positions.from_numpy(pts_np.astype(np.float64))
        self.tets.from_numpy(tets_np.astype(np.int32))
        self.label.from_numpy(labels_np.astype(np.int32))
        
        # Pre-compute density
        rho_np = np.vectorize(lambda lbl: DEFAULT_RHO.get(int(lbl), DEFAULT_RHO[0]),
                              otypes=[np.float64])(labels_np)
        self.rho.from_numpy(rho_np)
        
        # Initialize all preprocessing components
        self._initialize_deformation_fields()
        self._compute_tet_volume()
        self._compute_node_mass()
        self._initialize_anisotropy()
        self._initialize_arrays()
        self._initialize_stiffness()
        self._initialize_torsion_springs()
        
    @ti.kernel
    def _initialize_deformation_fields(self):
        """Initialize deformation-related fields"""
        for i in range(self.N):
            self.displacement_field[i] = ti.Vector([0.0, 0.0, 0.0])
            self.deformed_positions[i] = self.initial_positions[i]
        
        for k in range(self.M):
            # Initialize matrices to identity
            for i, j in ti.static(ti.ndrange(3, 3)):
                if i == j:
                    self.deformation_gradient[k][i, j] = 1.0
                    self.svd_U[k][i, j] = 1.0
                    self.svd_V[k][i, j] = 1.0
                else:
                    self.deformation_gradient[k][i, j] = 0.0
                    self.svd_U[k][i, j] = 0.0
                    self.svd_V[k][i, j] = 0.0
                    
                self.reference_edge_matrix[k][i, j] = 0.0
                self.deformed_edge_matrix[k][i, j] = 0.0
            
            # Initialize other fields
            self.svd_sigma[k] = ti.Vector([1.0, 1.0, 1.0])
            self.principal_stretches[k] = ti.Vector([1.0, 1.0, 1.0])
            self.stretch_anisotropy[k] = 0.0
            self.is_degenerate[k] = 0
            self.is_near_rigid[k] = 0
            
            # Initialize anisotropy axes to world coordinates
            self.anisotropy_axes[k, 0] = ti.Vector([1.0, 0.0, 0.0])
            self.anisotropy_axes[k, 1] = ti.Vector([0.0, 1.0, 0.0])
            self.anisotropy_axes[k, 2] = ti.Vector([0.0, 0.0, 1.0])
            
            # Initialize deformation coefficients
            for i, j in ti.static(ti.ndrange(3, 4)):
                self.deformation_coeffs[k][i, j] = 0.0
    
    def set_displacement_field(self, displacement_np):
        """Set displacement field from 4DCT registration data"""
        if displacement_np.shape != (self.N, 3):
            raise ValueError(f"Displacement field shape {displacement_np.shape} doesn't match mesh nodes {(self.N, 3)}")
        
        self.displacement_field.from_numpy(displacement_np.astype(np.float64))
        self._update_deformed_positions()
        self._compute_edge_matrices()
        self._compute_deformation_gradients()
        self._compute_anisotropy_axes_from_svd()
        self._compute_particle_deformation_coefficients()
        
        # Complete preprocessing after deformation analysis
        self._compute_intersections()  
        self._compute_coefficients()
        self._store_initial_configuration()
        self._store_rest_lengths()
        self._detect_boundary_nodes()
        
    @ti.kernel
    def _update_deformed_positions(self):
        """Compute deformed positions: x_i = X_i + u_i"""
        for i in range(self.N):
            self.deformed_positions[i] = self.initial_positions[i] + self.displacement_field[i]
    
    @ti.kernel 
    def _compute_edge_matrices(self):
        """Compute reference and deformed edge matrices D_m and d_x for each tetrahedron"""
        for k in range(self.M):
            # Get vertex indices
            v0_idx = self.tets[k][0]
            v1_idx = self.tets[k][1] 
            v2_idx = self.tets[k][2]
            v3_idx = self.tets[k][3]
            
            # Reference positions (initial)
            X0 = self.initial_positions[v0_idx]
            X1 = self.initial_positions[v1_idx]
            X2 = self.initial_positions[v2_idx]
            X3 = self.initial_positions[v3_idx]
            
            # Deformed positions
            x0 = self.deformed_positions[v0_idx]
            x1 = self.deformed_positions[v1_idx]
            x2 = self.deformed_positions[v2_idx]
            x3 = self.deformed_positions[v3_idx]
            
            # Compute edge vectors from vertex 0
            # D_m = [X1-X0, X2-X0, X3-X0] (3x3 matrix)
            edge1_ref = X1 - X0
            edge2_ref = X2 - X0  
            edge3_ref = X3 - X0
            
            edge1_def = x1 - x0
            edge2_def = x2 - x0
            edge3_def = x3 - x0
            
            # Store as column vectors in matrices
            for i in ti.static(range(3)):
                self.reference_edge_matrix[k][i, 0] = edge1_ref[i]
                self.reference_edge_matrix[k][i, 1] = edge2_ref[i]
                self.reference_edge_matrix[k][i, 2] = edge3_ref[i]
                
                self.deformed_edge_matrix[k][i, 0] = edge1_def[i]
                self.deformed_edge_matrix[k][i, 1] = edge2_def[i]
                self.deformed_edge_matrix[k][i, 2] = edge3_def[i]
    
    @ti.kernel
    def _compute_deformation_gradients(self):
        """Compute deformation gradient alpha = d_x · D_m^(-1) for each tetrahedron"""
        for k in range(self.M):
            # Get reference edge matrix D_m
            D_m = self.reference_edge_matrix[k]
            d_x = self.deformed_edge_matrix[k]
            
            # Check for degeneracy using determinant
            det_D_m = D_m.determinant()
            
            if ti.abs(det_D_m) > EPS_ZERO:
                D_m_inv = D_m.inverse()# D_m^(-1) 
                
                # Compute deformation gradient: alpha = d_x · D_m^(-1)
                alpha = d_x @ D_m_inv
                self.deformation_gradient[k] = alpha
                
                # Check for degenerate deformation (negative determinant)
                det_alpha = alpha.determinant()
                self.is_degenerate[k] = 1 if det_alpha <= 0.0 else 0
                
            else:
                # Degenerate reference tetrahedron - set to identity
                self.is_degenerate[k] = 1
                for i, j in ti.static(ti.ndrange(3, 3)):
                    if i == j:
                        self.deformation_gradient[k][i, j] = 1.0
                    else:
                        self.deformation_gradient[k][i, j] = 0.0
    
    @ti.kernel
    def _compute_anisotropy_axes_from_svd(self):
        """Compute anisotropy axes from deformation gradient using SVD: alpha = U·Sigma·V^T"""
        for k in range(self.M):
            if self.is_degenerate[k] == 0:
                # Get deformation gradient alpha
                alpha = self.deformation_gradient[k]
                
                # Perform SVD: alpha = U·Sigma·V^T
                U, sigma, V = ti.svd(alpha)
                
                # Store SVD components
                self.svd_U[k] = U
                self.svd_V[k] = V
                
                # Extract singular values (diagonal of Sigma matrix)
                lambda0 = sigma[0, 0]  # Largest singular value
                lambda1 = sigma[1, 1]  # Second singular value  
                lambda2 = sigma[2, 2]  # Smallest singular value
                
                # Store principal stretches (sorted lambda0 ≥ lambda1 ≥ lambda2)
                self.principal_stretches[k] = ti.Vector([lambda0, lambda1, lambda2])
                self.svd_sigma[k] = ti.Vector([lambda0, lambda1, lambda2])
                
                # Extract principal stretch directions from V matrix columns
                v0 = ti.Vector([V[0, 0], V[1, 0], V[2, 0]])  # First column (largest stretch direction)
                v1 = ti.Vector([V[0, 1], V[1, 1], V[2, 1]])  # Second column
                v2 = ti.Vector([V[0, 2], V[1, 2], V[2, 2]])  # Third column (smallest stretch direction)
                
                # Ensure right-handed coordinate system
                cross_product = v0.cross(v1)
                if cross_product.dot(v2) < 0.0:
                    # Swap v1 and v2 to maintain right-handed system
                    temp = v1
                    v1 = v2  
                    v2 = temp
                    
                    # Also swap corresponding singular values
                    temp_lambda = lambda1
                    lambda1 = lambda2
                    lambda2 = temp_lambda
                    self.principal_stretches[k] = ti.Vector([lambda0, lambda1, lambda2])
                
                # Set anisotropy axes
                self.anisotropy_axes[k, 0] = v0  # e0: largest stretch direction
                self.anisotropy_axes[k, 1] = v1  # e1: second stretch direction  
                self.anisotropy_axes[k, 2] = v2  # e2: third stretch direction
                
                # Compute stretch anisotropy measure: (lambda_max - lambda_min) / lambda_max
                if lambda0 > EPS_ZERO:
                    self.stretch_anisotropy[k] = (lambda0 - lambda2) / lambda0
                else:
                    self.stretch_anisotropy[k] = 0.0
                
                # Check for near-rigid motion
                eps_rigid = 1e-6
                lambda_diffs = ti.Vector([ti.abs(lambda0 - 1.0), ti.abs(lambda1 - 1.0), ti.abs(lambda2 - 1.0)])
                max_diff = ti.max(lambda_diffs[0], ti.max(lambda_diffs[1], lambda_diffs[2]))
                
                if max_diff < eps_rigid:
                    self.is_near_rigid[k] = 1
                else:
                    self.is_near_rigid[k] = 0
            else:
                # Handle degenerate elements using displacement-based fallback
                self._handle_degenerate_tetrahedron(k)
    
    @ti.func
    def _handle_degenerate_tetrahedron(self, k):
        """Handle degenerate tetrahedra using displacement-based fallback strategy"""
        # Get vertex indices
        v0_idx = self.tets[k][0]
        v1_idx = self.tets[k][1] 
        v2_idx = self.tets[k][2]
        v3_idx = self.tets[k][3]
        
        # Compute mean nodal displacement
        u_bar = (self.displacement_field[v0_idx] + self.displacement_field[v1_idx] + 
                 self.displacement_field[v2_idx] + self.displacement_field[v3_idx]) / 4.0
        
        u_bar_norm = u_bar.norm()
        eps_displacement = 1e-6
        
        # Initialize axes
        e0 = ti.Vector([1.0, 0.0, 0.0])
        e1 = ti.Vector([0.0, 1.0, 0.0])
        e2 = ti.Vector([0.0, 0.0, 1.0])
        
        if u_bar_norm > eps_displacement:
            # Use displacement-based frame
            e0 = u_bar.normalized()
            
            # Orthogonalize reference edge against e0
            X0 = self.initial_positions[v0_idx]
            X1 = self.initial_positions[v1_idx]
            r = X1 - X0
            
            r_proj = r - r.dot(e0) * e0
            if r_proj.norm() > eps_displacement:
                e1 = r_proj.normalized()
            else:
                # Fallback: use world coordinate
                if ti.abs(e0[0]) < 0.9:
                    world_vec = ti.Vector([1.0, 0.0, 0.0])
                    e1 = (world_vec - world_vec.dot(e0) * e0).normalized()
                else:
                    world_vec = ti.Vector([0.0, 1.0, 0.0])
                    e1 = (world_vec - world_vec.dot(e0) * e0).normalized()
            
            # Third axis: e2 = e0 × e1
            e2 = e0.cross(e1)
        
        # Set anisotropy axes
        self.anisotropy_axes[k, 0] = e0
        self.anisotropy_axes[k, 1] = e1  
        self.anisotropy_axes[k, 2] = e2
    
    @ti.kernel
    def _compute_particle_deformation_coefficients(self):
        """Compute particle deformation coefficients for each tetrahedron"""
        for k in range(self.M):
            if self.is_degenerate[k] == 0:
                # Get deformation gradient
                alpha = self.deformation_gradient[k]
                
                # Compute shape functions derivatives in reference configuration
                # For linear tetrahedron: nablaN_i = (reference_edge_matrix^-1)^T · e_i
                D_m = self.reference_edge_matrix[k]
                det_D_m = D_m.determinant()
                
                if ti.abs(det_D_m) > EPS_ZERO:
                    D_m_inv = D_m.inverse()
                    D_m_inv_T = D_m_inv.transpose()
                    
                    # Standard shape function gradients for tetrahedron
                    # nablaN_0 = -(nablaN_1 + nablaN_2 + nablaN_3)
                    grad_N1 = ti.Vector([D_m_inv_T[0, 0], D_m_inv_T[1, 0], D_m_inv_T[2, 0]])
                    grad_N2 = ti.Vector([D_m_inv_T[0, 1], D_m_inv_T[1, 1], D_m_inv_T[2, 1]])
                    grad_N3 = ti.Vector([D_m_inv_T[0, 2], D_m_inv_T[1, 2], D_m_inv_T[2, 2]])
                    grad_N0 = -(grad_N1 + grad_N2 + grad_N3)

                    # Compute deformation coefficients: C_ij = alpha_ik * (nablaN_j)_k
                    # This gives the contribution of node j to the deformation in direction i
                    for i in ti.static(range(3)):  # spatial directions
                        self.deformation_coeffs[k][i, 0] = (alpha[i, 0] * grad_N0[0] + 
                                                           alpha[i, 1] * grad_N0[1] + 
                                                           alpha[i, 2] * grad_N0[2])
                        self.deformation_coeffs[k][i, 1] = (alpha[i, 0] * grad_N1[0] + 
                                                           alpha[i, 1] * grad_N1[1] + 
                                                           alpha[i, 2] * grad_N1[2])
                        self.deformation_coeffs[k][i, 2] = (alpha[i, 0] * grad_N2[0] + 
                                                           alpha[i, 1] * grad_N2[1] + 
                                                           alpha[i, 2] * grad_N2[2])
                        self.deformation_coeffs[k][i, 3] = (alpha[i, 0] * grad_N3[0] + 
                                                           alpha[i, 1] * grad_N3[1] + 
                                                           alpha[i, 2] * grad_N3[2])
                else:
                    # Degenerate case - set to zero
                    for i, j in ti.static(ti.ndrange(3, 4)):
                        self.deformation_coeffs[k][i, j] = 0.0
            else:
                # Degenerate case - set to zero
                for i, j in ti.static(ti.ndrange(3, 4)):
                    self.deformation_coeffs[k][i, j] = 0.0
    
    @ti.kernel
    def _compute_tet_volume(self):
        """Compute volume for each tetrahedron"""
        for k in range(self.M):
            a = self.x[self.tets[k][0]]
            b = self.x[self.tets[k][1]]
            c = self.x[self.tets[k][2]]
            d = self.x[self.tets[k][3]]
            vol = ti.abs((b - a).dot((c - a).cross(d - a))) / 6.0
            self.vol[k] = ti.max(vol, EPS_ZERO)
    
    @ti.kernel
    def _compute_node_mass(self):
        """Compute lumped mass at each node"""
        for i in self.mass:
            self.mass[i] = 0.0
        
        for k in range(self.M):
            mass_per_node = self.rho[k] * self.vol[k] / 4.0
            for l in ti.static(range(4)):
                node_idx = self.tets[k][l]
                ti.atomic_add(self.mass[node_idx], mass_per_node)
    
    @ti.kernel
    def _initialize_anisotropy(self):
        """Initialize anisotropy axes to world coordinates"""
        for k in range(self.M):
            self.anisotropy_axes[k, 0] = ti.Vector([1.0, 0.0, 0.0])
            self.anisotropy_axes[k, 1] = ti.Vector([0.0, 1.0, 0.0])
            self.anisotropy_axes[k, 2] = ti.Vector([0.0, 0.0, 1.0])
    
    @ti.kernel
    def _initialize_arrays(self):
        """Initialize intersection and coefficient arrays"""
        for k, i in ti.ndrange(self.M, 6):
            self.intersection_valid[k, i] = 0
            self.intersection_face[k, i] = -1
            self.intersection_points[k, i] = ti.Vector([0.0, 0.0, 0.0])
            
        for k, i, j in ti.ndrange(self.M, 4, 6):
            self.C_k[k, i, j] = 0.0
    
    @ti.kernel
    def _initialize_stiffness(self):
        """Initialize stiffness parameters for each axis of each tetrahedron"""
        for k in range(self.M):
            base_stiffness = 0.5  # N/m
            
            if self.label[k] == 1:  # Air-rich tissue
                base_stiffness *= 0.2
            elif self.label[k] == 2:  # Fibrotic tissue
                base_stiffness *= 2.0
            
            self.stiffness[k, 0] = base_stiffness
            self.stiffness[k, 1] = base_stiffness  
            self.stiffness[k, 2] = base_stiffness
    
    @ti.kernel
    def _initialize_torsion_springs(self):
        """Initialize torsion spring parameters"""
        for k in range(self.M):
            if self.vol[k] > 1e-10:
                base_torsion_stiffness = 0.1  # N⋅m/rad
                
                if self.label[k] == 1:  # Air-rich tissue
                    base_torsion_stiffness *= 0.2
                elif self.label[k] == 2:  # Fibrotic tissue 
                    base_torsion_stiffness *= 200.0
                
                self.torsion_stiffness[k, 0] = base_torsion_stiffness
                self.torsion_stiffness[k, 1] = base_torsion_stiffness
                self.torsion_stiffness[k, 2] = base_torsion_stiffness
                
                # Initialize rest cosine angles
                self.rest_cos_angles[k, 0] = 0.0  # Will be updated after intersections
                self.rest_cos_angles[k, 1] = 0.0
                self.rest_cos_angles[k, 2] = 0.0
    
    @ti.func
    def compute_barycentric(self, p, v0, v1, v2):
        """Compute barycentric coordinates"""
        v01 = v1 - v0
        v02 = v2 - v0
        v0p = p - v0
        
        dot00 = v02.dot(v02)
        dot01 = v02.dot(v01)
        dot02 = v02.dot(v0p)
        dot11 = v01.dot(v01)
        dot12 = v01.dot(v0p)
        
        denom = dot00 * dot11 - dot01 * dot01
        u = 0.0
        v = 0.0
        w = 0.0
        
        if ti.abs(denom) > EPS_ZERO:
            inv_denom = 1.0 / denom
            u = (dot11 * dot02 - dot01 * dot12) * inv_denom
            v = (dot00 * dot12 - dot01 * dot02) * inv_denom
            w = 1.0 - u - v
        
        return u, v, w

    @ti.func
    def point_in_triangle(self, p, v0, v1, v2):
        """Test if point is inside triangle"""
        u, v, w = self.compute_barycentric(p, v0, v1, v2)
        return (u >= -EPS_BARYCENTRIC and 
                v >= -EPS_BARYCENTRIC and 
                w >= -EPS_BARYCENTRIC)

    @ti.kernel
    def _compute_intersections(self):
        """Compute intersection points for each tetrahedron and axis"""
        for k in range(self.M):
            if self.vol[k] > 1e-12:
                # Get tetrahedron vertices
                v0 = self.x[self.tets[k][0]]
                v1 = self.x[self.tets[k][1]]
                v2 = self.x[self.tets[k][2]]
                v3 = self.x[self.tets[k][3]]
                
                # Calculate barycenter
                barycenter = (v0 + v1 + v2 + v3) / 4.0
                
                # Process each axis
                for axis_idx in ti.static(range(3)):
                    ray_dir = self.anisotropy_axes[k, axis_idx]
                    intersections_found = 0
                    
                    # Check both directions along the axis
                    for sign in ti.static([-1.0, 1.0]):
                        dir = sign * ray_dir
                        
                        # Check all four faces
                        for face_idx in ti.static(range(4)):
                            # Define face vertices
                            va = ti.Vector([0.0, 0.0, 0.0])
                            vb = ti.Vector([0.0, 0.0, 0.0])
                            vc = ti.Vector([0.0, 0.0, 0.0])
                            
                            if face_idx == 0:
                                va, vb, vc = v0, v1, v2
                            elif face_idx == 1:
                                va, vb, vc = v0, v2, v3
                            elif face_idx == 2:
                                va, vb, vc = v0, v3, v1
                            else:
                                va, vb, vc = v1, v3, v2
                            
                            face_normal = (vb - va).cross(vc - va)
                            if face_normal.norm() > EPS_ZERO:
                                face_normal = face_normal.normalized()
                                denom = dir.dot(face_normal)
                                if ti.abs(denom) > EPS_PARALLEL:
                                    t = (va - barycenter).dot(face_normal) / denom
                                    p = barycenter + t * dir
                                    if self.point_in_triangle(p, va, vb, vc):
                                        if intersections_found < 2:
                                            idx = axis_idx * 2 + intersections_found
                                            self.intersection_points[k, idx] = p
                                            self.intersection_valid[k, idx] = 1
                                            self.intersection_face[k, idx] = face_idx
                                            intersections_found += 1

    @ti.kernel
    def _compute_coefficients(self):
        """Compute coefficient matrix"""
        for k in range(self.M):
            if self.vol[k] > 1e-12:
                # Get vertices
                v0 = self.x[self.tets[k][0]]
                v1 = self.x[self.tets[k][1]]
                v2 = self.x[self.tets[k][2]]
                v3 = self.x[self.tets[k][3]]
                
                # Process each intersection point
                for pt_idx in range(6):
                    if self.intersection_valid[k, pt_idx] == 1:
                        p = self.intersection_points[k, pt_idx]
                        face_idx = self.intersection_face[k, pt_idx]
                        
                        # Compute barycentric coordinates based on face
                        if face_idx == 0:  # Face 0,1,2
                            u, v, w = self.compute_barycentric(p, v0, v1, v2)
                            self.C_k[k, 0, pt_idx] = w
                            self.C_k[k, 1, pt_idx] = v
                            self.C_k[k, 2, pt_idx] = u
                            self.C_k[k, 3, pt_idx] = 0.0
                        elif face_idx == 1:  # Face 0,2,3
                            u, v, w = self.compute_barycentric(p, v0, v2, v3)
                            self.C_k[k, 0, pt_idx] = w
                            self.C_k[k, 1, pt_idx] = 0.0
                            self.C_k[k, 2, pt_idx] = v
                            self.C_k[k, 3, pt_idx] = u
                        elif face_idx == 2:  # Face 0,3,1
                            u, v, w = self.compute_barycentric(p, v0, v3, v1)
                            self.C_k[k, 0, pt_idx] = w
                            self.C_k[k, 1, pt_idx] = u
                            self.C_k[k, 2, pt_idx] = 0.0
                            self.C_k[k, 3, pt_idx] = v
                        else:  # Face 1,3,2
                            u, v, w = self.compute_barycentric(p, v1, v3, v2)
                            self.C_k[k, 0, pt_idx] = 0.0
                            self.C_k[k, 1, pt_idx] = w
                            self.C_k[k, 2, pt_idx] = u
                            self.C_k[k, 3, pt_idx] = v
    
    @ti.kernel
    def _store_initial_configuration(self):
        """Store all initial configuration data"""
        # Store initial intersections
        for k, i in ti.ndrange(self.M, 6):
            self.initial_intersection_points[k, i] = self.intersection_points[k, i]
        
        # Store initial axis vectors
        for k, i in ti.ndrange(self.M, 3):
            self.initial_axis_vectors[k, i] = self.anisotropy_axes[k, i]
    
    @ti.kernel
    def _store_rest_lengths(self):
        """Store initial axis lengths as rest lengths"""
        for k in range(self.M):
            if self.vol[k] > 1e-12:
                for axis_idx in ti.static(range(3)):
                    pt1_idx = axis_idx * 2
                    pt2_idx = axis_idx * 2 + 1
                    
                    if (self.intersection_valid[k, pt1_idx] == 1 and 
                        self.intersection_valid[k, pt2_idx] == 1):
                        p1 = self.intersection_points[k, pt1_idx]
                        p2 = self.intersection_points[k, pt2_idx]
                        rest_length = (p1 - p2).norm()
                        self.rest_lengths[k, axis_idx] = rest_length
                    else:
                        self.rest_lengths[k, axis_idx] = 1.0
    
    def _detect_boundary_nodes(self):
        """Automatic boundary detection"""
        # Initialize boundary detection
        self._initialize_boundary_detection()
        
        # Detect boundary faces and mark boundary nodes
        self._mark_boundary_faces_and_nodes()
        
        boundary_count = self._count_boundary_nodes()
        print(f"Boundary detection: {boundary_count} boundary nodes detected")

    @ti.kernel
    def _initialize_boundary_detection(self):
        """Initialize boundary detection fields"""
        for i in range(self.N):
            self.boundary_nodes[i] = 0

    def _mark_boundary_faces_and_nodes(self):
        """Mark boundary faces and their constituent nodes using face counting"""
        # Create face-to-tetrahedron mapping
        face_dict = {}
        
        # For each tetrahedron, extract its 4 faces
        tets_np = self.tets.to_numpy()
        
        for tet_idx in range(self.M):
            tet = tets_np[tet_idx]
            
            # Four faces of tetrahedron (node indices sorted for consistency)
            faces = [
                tuple(sorted([tet[0], tet[1], tet[2]])),  # Face opposite to node 3
                tuple(sorted([tet[0], tet[1], tet[3]])),  # Face opposite to node 2
                tuple(sorted([tet[0], tet[2], tet[3]])),  # Face opposite to node 1
                tuple(sorted([tet[1], tet[2], tet[3]])),  # Face opposite to node 0
            ]
            
            for face in faces:
                if face in face_dict:
                    face_dict[face] += 1
                else:
                    face_dict[face] = 1
        
        # Faces with count = 1 are boundary faces
        boundary_faces = [face for face, count in face_dict.items() if count == 1]
        
        # Mark nodes that belong to boundary faces
        boundary_node_set = set()
        for face in boundary_faces:
            for node in face:
                boundary_node_set.add(node)
        
        # Update Taichi fields
        boundary_nodes_array = np.array(list(boundary_node_set), dtype=np.int32)
        self._set_boundary_nodes_from_array(boundary_nodes_array)

    @ti.kernel
    def _set_boundary_nodes_from_array(self, boundary_nodes_list: ti.types.ndarray()):
        """Set boundary nodes from numpy array"""
        for i in range(boundary_nodes_list.shape[0]):
            node_idx = boundary_nodes_list[i]
            self.boundary_nodes[node_idx] = 1

    def _count_boundary_nodes(self):
        """Count number of boundary nodes"""
        return int(self.boundary_nodes.to_numpy().sum())
    
    def get_results(self):
        """Get all computed results as numpy arrays for solver preprocessing"""
        results = {
            # Mesh topology and properties
            'mesh_points': self.x.to_numpy(),
            'tetrahedra': self.tets.to_numpy(),
            'labels': self.label.to_numpy(),
            'density': self.rho.to_numpy(),
            'volume': self.vol.to_numpy(),
            'mass': self.mass.to_numpy(),
            
            # Deformation analysis
            'anisotropy_axes': self.anisotropy_axes.to_numpy(),
            'deformation_gradient': self.deformation_gradient.to_numpy(),
            'deformation_coeffs': self.deformation_coeffs.to_numpy(),
            'principal_stretches': self.principal_stretches.to_numpy(),
            'stretch_anisotropy': self.stretch_anisotropy.to_numpy(),
            'svd_U': self.svd_U.to_numpy(),
            'svd_sigma': self.svd_sigma.to_numpy(),
            'svd_V': self.svd_V.to_numpy(),
            
            # Spring system preprocessing
            'intersection_points': self.intersection_points.to_numpy(),
            'intersection_valid': self.intersection_valid.to_numpy(),
            'intersection_face': self.intersection_face.to_numpy(),
            'coefficient_matrix': self.C_k.to_numpy(),
            'rest_lengths': self.rest_lengths.to_numpy(),
            'stiffness': self.stiffness.to_numpy(),
            'torsion_stiffness': self.torsion_stiffness.to_numpy(),
            'rest_cos_angles': self.rest_cos_angles.to_numpy(),
            
            # Boundary conditions
            'boundary_nodes': self.boundary_nodes.to_numpy(),
            
            # Initial configuration
            'initial_positions': self.initial_positions.to_numpy(),
            'initial_intersection_points': self.initial_intersection_points.to_numpy(),
            'initial_axis_vectors': self.initial_axis_vectors.to_numpy(),
            
            # Displacement field
            'displacement_field': self.displacement_field.to_numpy(),
            
            # Flags
            'is_degenerate': self.is_degenerate.to_numpy(),
            'is_near_rigid': self.is_near_rigid.to_numpy(),
            
            # Metadata
            'mesh_info': {
                'n_nodes': self.N,
                'n_tetrahedra': self.M,
                'has_displacement': True if np.any(self.displacement_field.to_numpy()) else False
            }
        }
        return results
    
    def save_results(self, output_path):
        """Save results to file"""
        results = self.get_results()
        np.savez(f"{output_path}.npz", **results)
        print(f"Results saved to {output_path}.npz")
    
    def print_summary(self):
        """Print summary of deformation analysis"""
        print(f"\n=== Deformation Analysis Summary ===")
        print(f"Mesh: {self.N} nodes, {self.M} tetrahedra")
        
        # Displacement statistics
        displacement_magnitudes = np.linalg.norm(self.displacement_field.to_numpy(), axis=1)
        print(f"\nDisplacement Field:")
        print(f"  Max displacement: {displacement_magnitudes.max():.6f} m")
        print(f"  Mean displacement: {displacement_magnitudes.mean():.6f} m")
        print(f"  Nodes with displacement > 1m: {np.sum(displacement_magnitudes > 1)}/{len(displacement_magnitudes)}")
        
        # Deformation statistics
        stretches = self.principal_stretches.to_numpy()
        anisotropy = self.stretch_anisotropy.to_numpy()
        
        print(f"\nDeformation Analysis:")
        print(f"  Principal stretch lambda1: {stretches[:, 0].mean():.4f} ± {stretches[:, 0].std():.4f}")
        print(f"  Principal stretch lambda2: {stretches[:, 1].mean():.4f} ± {stretches[:, 1].std():.4f}")
        print(f"  Principal stretch lambda3: {stretches[:, 2].mean():.4f} ± {stretches[:, 2].std():.4f}")
        print(f"  Stretch anisotropy: {anisotropy.mean():.4f} ± {anisotropy.std():.4f}")
        
        # Degeneracy statistics
        n_degenerate = self.is_degenerate.to_numpy().sum()
        n_rigid = self.is_near_rigid.to_numpy().sum()
        print(f"\nDegenerate elements: {n_degenerate}/{self.M} ({100*n_degenerate/self.M:.1f}%)")
        print(f"Near-rigid elements: {n_rigid}/{self.M} ({100*n_rigid/self.M:.1f}%)")


def load_mesh_from_file(mesh_path):
    """Load mesh from file"""
    mesh = meshio.read(mesh_path)
    pts = mesh.points
    tid = next((i for i, c in enumerate(mesh.cells) if c.type == "tetra"), None)
    if tid is None:
        raise ValueError("No tetrahedral cells found in mesh")
    tets = mesh.cells[tid].data
    lbls = mesh.cell_data.get("c_labels", [np.zeros(tets.shape[0], dtype=np.int32)])[tid] if "c_labels" in mesh.cell_data else np.zeros(tets.shape[0], dtype=np.int32)
    return pts, tets, lbls


def load_mesh_with_displacement(mesh_path, displacement_path=None):
    """Load mesh and optionally apply displacement field"""
    from displacement_loader import load_and_interpolate_displacement
    pts, tets, lbls, displacement_vectors = load_and_interpolate_displacement(displacement_path, mesh_path)
    return pts, tets, lbls, displacement_vectors


def process_case_data(case_folder):
    """Process all mesh and displacement combinations for a case"""
    from pathlib import Path
    
    case_path = Path(case_folder)
    case_name = case_path.name
    
    # Create output folder
    output_base = Path("data_processed_deformation") / case_name
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Find mesh files
    mesh_files = list(case_path.glob("pygalmesh/*_lung_regions_11.xdmf"))
    
    # Find displacement files  
    displacement_files = list(case_path.glob("CorrField/*.nii.gz"))
    
    print(f"\n=== Processing {case_name} ===")
    print(f"Found {len(mesh_files)} mesh files")
    print(f"Found {len(displacement_files)} displacement files")
    
    processed_count = 0
    
    for mesh_file in mesh_files:
        # Extract time point from mesh filename (e.g., case1_T00_lung_regions_11.xdmf -> T00)
        mesh_time = mesh_file.stem.split('_')[1]  # T00, T10, etc.
        
        for displacement_file in displacement_files:
            # Extract time points from displacement filename (e.g., case1_T00_T50.nii.gz -> T00_T50)
            disp_name = displacement_file.stem.replace('.nii', '')  # Remove .nii from .nii.gz to extract name
            disp_times = '_'.join(disp_name.split('_')[-2:])  # T00_T50
            
            # Check if mesh time matches the first time in displacement
            if mesh_time in disp_times.split('_')[0]:
                output_name = f"{case_name}_{mesh_time}_to_{disp_times.split('_')[1]}_deformation"
                output_path = output_base / output_name
                
                # Skip if already processed
                if (output_path.parent / f"{output_path.name}.npz").exists():
                    print(f"Skipping (already exists): {output_path.name}.npz")
                    processed_count += 1
                    continue
                
                print(f"\nProcessing: {mesh_file.name} + {displacement_file.name}")
                print(f"Output: {output_path}.npz")
                
                # Load data
                pts, tets, lbls, displacement_vectors = load_mesh_with_displacement(
                    str(mesh_file), str(displacement_file)
                )
                
                # Create processor
                processor = DeformationProcessor(pts, tets, lbls)
                
                # Compute deformation
                processor.set_displacement_field(displacement_vectors)
                
                # Save results as numpy
                processor.save_results(str(output_path))
                processed_count += 1
                    

    
    print(f"\n=== {case_name} Complete: {processed_count} files processed ===")
    return processed_count


def main():
    """Main function - automatically process all available case data"""
    from pathlib import Path
    
    print("=== Automatic Deformation Processor ===")
    print("Processing all available case data...")
    
    # Find all case folders
    data_path = Path("data")
    case_folders = list(data_path.glob("Case*Pack"))
    
    
    print(f"Found {len(case_folders)} case folders:")
    for folder in case_folders:
        print(f"  - {folder.name}")
    
    total_processed = 0
    
    # Process each case
    for case_folder in case_folders:
        count = process_case_data(case_folder)
        total_processed += count

    print(f"\n=== All Cases Complete ===")
    print(f"Total files processed: {total_processed}")
    print(f"Results saved in: data_processed_deformation/")
    
    # Show output structure
    output_path = Path("data_processed_deformation")  
    if output_path.exists():
        print(f"\nOutput structure:")
        for case_dir in output_path.iterdir():
            if case_dir.is_dir():
                npz_files = list(case_dir.glob("*.npz"))
                print(f"  {case_dir.name}/: {len(npz_files)} files")


if __name__ == "__main__":
    main()