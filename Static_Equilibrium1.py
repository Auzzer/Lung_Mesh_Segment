"""
Pure Mass-Spring approach (2.4.2.1) with simplified volume springs

This implementation follows:
- Section 2.4.2.1: Spring-based forces along anisotropy axes (Hooke's law)
- Section 2.4.2.3/2.4.2.4: Simplified SMS volume springs (no tensor computation)

Key features:
- Pure spring forces along 3 anisotropy axes per tetrahedron
- Simplified volume preservation using volume gradient approach
- No continuum mechanics tensors (F, E, S) - pure mass-spring system

differnce between ti.kernel and ti.func:
- ti.kernel: could be excuted in python scope
- ti.func: can only be called from other ti.func or ti.kernel, not in python scope but taichi scope
"""

import taichi as ti
import meshio
import numpy as np
import sys
import os

ti.init(arch=ti.cuda)

# --------------------------------------------
# CONSTANTS
# --------------------------------------------
DEFAULT_RHO = {
    0: 1_050.0,   # normal tissue (kg/m³)
    1:   250.0,   # air-rich
    2: 1_100.0,   # fibrotic
}

# Tolerance constants
EPS_PARALLEL = 1e-10
EPS_BARYCENTRIC = 1e-8
EPS_DISTANCE = 1e-6
EPS_ZERO = 1e-10

# --------------------------------------------
@ti.data_oriented
class Adamss:
    def __init__(self, pts_np, tets_np, labels_np):
        self.N = pts_np.shape[0]      # #nodes
        self.M = tets_np.shape[0]     # #tets

        # -------------------------------Core Taichi  data fields ------------------------------------
        self.x     = ti.Vector.field(3, ti.f64, shape=self.N)     # positions
        self.tets  = ti.Vector.field(4, ti.i32, shape=self.M)     # indices
        self.label = ti.field(ti.i32, shape=self.M)               # region id
        self.rho   = ti.field(ti.f64, shape=self.M)               # density
        self.vol   = ti.field(ti.f64, shape=self.M)               # volume
        self.mass  = ti.field(ti.f64, shape=self.N)               # lumped mass
        
        #--------------------------------------  Linear Spring Forces-------------------------------------- 
        ## Anisotropy and intersection fields
        self.anisotropy_axes = ti.Vector.field(3, ti.f64, shape=(self.M, 3))      # 3 axes per tet
        self.intersection_points = ti.Vector.field(3, ti.f64, shape=(self.M, 6))  # 6 points per tet
        self.intersection_valid = ti.field(ti.i32, shape=(self.M, 6))             # validity flag
        self.intersection_face = ti.field(ti.i32, shape=(self.M, 6))              # face index
        self.C_k = ti.field(ti.f64, shape=(self.M, 4, 6))                        # Coefficient matrix
        
        ## Force and dynamics fields
        self.force = ti.Vector.field(3, ti.f64, shape=self.N)    # forces on nodes
        self.vel = ti.Vector.field(3, ti.f64, shape=self.N)      # velocities
        self.rest_lengths = ti.field(ti.f64, shape=(self.M, 3))  # rest lengths for each axis
        self.stiffness = ti.field(ti.f64, shape=(self.M, 3))     # stiffness per axis per tet
        self.damping = 0.90                                      # Further reduced for stability
        
        ## Boundary detection and constraints (FEniCS-style)
        self.boundary_nodes = ti.field(ti.i32, shape=self.N)     # 1 if boundary node, 0 otherwise
        self.boundary_displacement = ti.Vector.field(3, ti.f64, shape=self.N)  # prescribed displacements
        self.is_boundary_constrained = ti.field(ti.i32, shape=self.N)  # 1 if Dirichlet BC applied
        
        #------------------------------ Static equilibrium solver fields----------------------------------------
        self.stiffness_matrix = ti.field(ti.f64, shape=(3*self.N, 3*self.N))    # Global stiffness matrix K
        self.residual = ti.Vector.field(3, ti.f64, shape=self.N)                # Residual vector r = g_int + F_ext
        self.solution_increment = ti.Vector.field(3, ti.f64, shape=self.N)      # Solution increment Δq
        self.temp_b = ti.field(ti.f64, shape=3*self.N)                          # Flattened RHS vector for linear solver
        self.temp_x = ti.field(ti.f64, shape=3*self.N)                          # Flattened solution vector for linear solver
        
        ## Torsion springs (Section 2.4.2.1) - 3 torsion springs between axis pairs
        self.torsion_stiffness = ti.field(ti.f64, shape=(self.M, 3))  # k_lm for each pair (0,1), (1,2), (2,0)
        self.rest_cos_angles = ti.field(ti.f64, shape=(self.M, 3))    # cos(alpha_l m^0) for rest angles
        self.torsion_damping = 0.05  # c_tau for torsion damping
        
        ## Barycentric volume springs (Eq. 2.76/2.77)
        self.barycentric_rest_lengths = ti.field(ti.f64, shape=(self.M, 4))  # |xi_j^0|
        self.barycentric_rest_center = ti.Vector.field(3, ti.f64, shape=self.M)  # x_b^0
        self.barycentric_ks = ti.field(ti.f64, shape=self.M)  # Adaptive spring constant
        self.barycentric_mu = 1e-6  # LMS step size for adaptive update (reduced for stability)
        self.barycentric_damping = 0.1  # damping coefficient for barycentric springs
        
        #----------Volume preservation/control fields (Section 2.4.2.3 & 2.4.2.4)----------
        self.volume_preservation_enabled = ti.field(ti.i32, shape=self.M)  # Enable flag
        self.volume_control_enabled = ti.field(ti.i32, shape=self.M)       # Volume control flag
        self.bulk_modulus = ti.field(ti.f64, shape=self.M)                # p parameter (Pa)
        self.target_volume_ratio = ti.field(ti.f64, shape=self.M)         # r = V^inf/V^0
        self.current_volume = ti.field(ti.f64, shape=self.M)              # Current volume
        
        ## Initial configuration storage
        self.initial_positions = ti.Vector.field(3, ti.f64, shape=self.N)         # Initial vertices
        self.initial_volume = ti.field(ti.f64, shape=self.M)                      # Initial volumes
        self.initial_intersection_points = ti.Vector.field(3, ti.f64, shape=(self.M, 6))
        self.initial_axis_vectors = ti.Vector.field(3, ti.f64, shape=(self.M, 3))
        
        # ----------Section 2.4: Deformation gradient and anisotropy axis computation from 4DCT data----------
        self.displacement_field = ti.Vector.field(3, ti.f64, shape=self.N)        # u_i displacement from 4DCT
        self.deformed_positions = ti.Vector.field(3, ti.f64, shape=self.N)        # x_i = X_i + u_i
        self.deformation_gradient = ti.Matrix.field(3, 3, ti.f64, shape=self.M)   # mapping alpha matrix per tetrahedron
        self.reference_edge_matrix = ti.Matrix.field(3, 3, ti.f64, shape=self.M)  # D_m matrix per tetrahedron
        self.deformed_edge_matrix = ti.Matrix.field(3, 3, ti.f64, shape=self.M)   # d_x matrix per tetrahedron
        
        ## SVD components for anisotropy axis extraction
        self.svd_U = ti.Matrix.field(3, 3, ti.f64, shape=self.M)             # U matrix from SVD
        self.svd_sigma = ti.Vector.field(3, ti.f64, shape=self.M)            # Singular values ( lambda0, lambda1, lambda2)
        self.svd_V = ti.Matrix.field(3, 3, ti.f64, shape=self.M)             # V matrix from SVD
        self.principal_stretches = ti.Vector.field(3, ti.f64, shape=self.M)  # Principal stretch values
        
        ## Flags for degenerate case handling
        self.is_degenerate = ti.field(ti.i32, shape=self.M)                       # Degenerate element flag
        self.is_near_rigid = ti.field(ti.i32, shape=self.M)                       # Near-rigid element flag

        # -------------------------------Some Tools-----------------------------
        ## Copy data from NumPy
        self.x.from_numpy(pts_np.astype(np.float64) * 1e-3)  # mm → m
        self.tets.from_numpy(tets_np.astype(np.int32))
        self.label.from_numpy(labels_np.astype(np.int32))

        ## Validate mesh
        self._validate_mesh()

        ## Pre-compute density
        rho_np = np.vectorize(lambda lbl: DEFAULT_RHO.get(int(lbl), DEFAULT_RHO[0]),
                              otypes=[np.float64])(labels_np)
        self.rho.from_numpy(rho_np)

        # ---------- Initialize simulation---------------
        self._compute_tet_volume()
        self._check_degenerate_tets()
        self._compute_node_mass()
        self._initialize_anisotropy()
        self._initialize_arrays()
        self._compute_intersections()
        self._compute_coefficients()
        self._initialize_stiffness()
        self._initialize_volume_preservation()
        self._initialize_dynamics()
        self._store_initial_configuration()
        self._initialize_torsion_springs()  # After storing initial configuration
        self._initialize_barycentric_springs()
        self._store_rest_lengths()
        self._initialize_deformation_fields()
        self._detect_boundary_nodes()  # FEniCS-style automatic boundary detection

    def _validate_mesh(self):
        """Basic mesh validation"""
        if self.N < 4:
            raise ValueError(f"Mesh has too few nodes: {self.N}")
        if self.M < 1:
            raise ValueError(f"Mesh has no tetrahedra: {self.M}")

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

    def _check_degenerate_tets(self):
        """Check for degenerate tetrahedra"""
        volumes = self.vol.to_numpy()
        degenerate_mask = volumes < 1e-12
        degenerate_count = np.sum(degenerate_mask)
        
        if degenerate_count > 0:
            print(f"Warning: Found {degenerate_count} degenerate tetrahedra out of {self.M}")

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
        """Initialize anisotropy axes to world coordinates (will be updated from 4DCT data)"""
        for k in range(self.M):
            # Initialize to world coordinate axes - these will be replaced when displacement field is set
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
            # Spring stiffness values - much lower for stability
            base_stiffness = 0.5  # N/m (reduced from 1.0)
            
            if self.label[k] == 1:  # Air-rich tissue
                base_stiffness *= 0.2
            elif self.label[k] == 2:  # Fibrotic tissue
                base_stiffness *= 2.0  # Reduced from 3.0
            
            self.stiffness[k, 0] = base_stiffness  # First axis stiffness
            self.stiffness[k, 1] = base_stiffness  # Second axis stiffness  
            self.stiffness[k, 2] = base_stiffness  # Third axis stiffness
    
    @ti.kernel
    def _initialize_torsion_springs(self):
        """Initialize torsion spring parameters (Section 2.4.2.1) using initial intersection vectors"""
        for k in range(self.M):
            if self.vol[k] > EPS_ZERO:
                # Torsion spring stiffness - typically smaller than axial stiffness
                base_torsion_stiffness = 0.1  # N⋅m/rad
                
                if self.label[k] == 1:  # Air-rich tissue
                    base_torsion_stiffness *= 0.2
                elif self.label[k] == 2:  # Fibrotic tissue 
                    base_torsion_stiffness *= 200.0
                
                # Three torsion springs between axis pairs: (0,1), (1,2), (2,0)
                self.torsion_stiffness[k, 0] = base_torsion_stiffness  # Between axes 0 and 1
                self.torsion_stiffness[k, 1] = base_torsion_stiffness  # Between axes 1 and 2
                self.torsion_stiffness[k, 2] = base_torsion_stiffness  # Between axes 2 and 0
                
                # Compute rest cosine angles from INITIAL intersection vectors: zeta_l^0 = q_{l,1}^0 - q_{l,2}^0
                # Check if initial intersection points are valid
                all_valid = True
                for i in range(6):
                    if self.intersection_valid[k, i] != 1:
                        all_valid = False
                        break
                
                if all_valid:
                    # Compute zeta vectors using loop
                    zeta = ti.Matrix.zero(ti.f32, 3, 3)  # 3x3 matrix to store 3 vectors as rows
                    
                    for i in range(3):
                        # Calculate zeta_i = (p_{2i} - p_{2i+1})
                        zeta_vec = (self.initial_intersection_points[k, 2*i] - 
                                   self.initial_intersection_points[k, 2*i+1])
                        
                        if zeta_vec.norm() > EPS_ZERO:
                            for j in range(3):
                                zeta[i, j] = zeta_vec.normalized()[j]
                        else:
                            for j in range(3):
                                zeta[i, j] = self.anisotropy_axes[k, i].normalized()[j]  # Fallback
                    
                    # Compute rest cosine angles
                    zeta0 = ti.Vector([zeta[0, 0], zeta[0, 1], zeta[0, 2]])
                    zeta1 = ti.Vector([zeta[1, 0], zeta[1, 1], zeta[1, 2]])
                    zeta2 = ti.Vector([zeta[2, 0], zeta[2, 1], zeta[2, 2]])
                    self.rest_cos_angles[k, 0] = zeta0.dot(zeta1)  # cos(alpha_01^0)
                    self.rest_cos_angles[k, 1] = zeta1.dot(zeta2)  # cos(alpha_12^0)
                    self.rest_cos_angles[k, 2] = zeta2.dot(zeta0)  # cos(alpha_20^0)
                else:
                    # Fallback to world axes if intersection points are invalid
                    axis0_hat = self.anisotropy_axes[k, 0].normalized()
                    axis1_hat = self.anisotropy_axes[k, 1].normalized()
                    axis2_hat = self.anisotropy_axes[k, 2].normalized()
                    
                    self.rest_cos_angles[k, 0] = axis0_hat.dot(axis1_hat)  # cos(alpha_01^0)
                    self.rest_cos_angles[k, 1] = axis1_hat.dot(axis2_hat)  # cos(alpha_12^0)
                    self.rest_cos_angles[k, 2] = axis2_hat.dot(axis0_hat)  # cos(alpha_20^0)


    @ti.kernel
    def _initialize_volume_preservation(self):
        """Initialize volume preservation parameters"""
        for k in range(self.M):
            self.volume_preservation_enabled[k] = 1  # Enable by default
            self.volume_control_enabled[k] = 0       # Disabled by default
            self.bulk_modulus[k] = 1e3               # 10 Pa (reduced from 1 kPa)
            self.target_volume_ratio[k] = 1.0        # Preserve initial volume
            self.current_volume[k] = self.vol[k]
    
    @ti.kernel
    def _initialize_barycentric_springs(self):
        """Initialize barycentric volume springs (Eq. 2.76/2.77)"""
        for k in range(self.M):
            if self.vol[k] > EPS_ZERO:
                # Compute initial barycenter: x_b^0 = (1/4) * sum x_j^0
                x_b_0 = ti.Vector([0.0, 0.0, 0.0])
                for j in ti.static(range(4)):
                    x_b_0 += self.initial_positions[self.tets[k][j]]
                x_b_0 /= 4.0
                self.barycentric_rest_center[k] = x_b_0
                
                # Compute rest lengths: |xi_j^0| = |x_b^0 - x_j^0|
                for j in ti.static(range(4)):
                    xi_j_0 = x_b_0 - self.initial_positions[self.tets[k][j]]
                    self.barycentric_rest_lengths[k, j] = xi_j_0.norm()
                
                # Initialize adaptive spring constant
                self.barycentric_ks[k] = self.bulk_modulus[k]  # Start with full bulk modulus strength

    @ti.kernel
    def _initialize_dynamics(self):
        """Initialize dynamic simulation parameters"""
        for i in range(self.N):
            self.force[i] = ti.Vector([0.0, 0.0, 0.0])
            self.vel[i] = ti.Vector([0.0, 0.0, 0.0])

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
        """Compute coefficient matrix (Eq. 2.17)"""
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
        # Store initial positions
        for i in range(self.N):
            self.initial_positions[i] = self.x[i]
        
        # Store initial volumes
        for k in range(self.M):
            self.initial_volume[k] = self.vol[k]
        
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
            
            self.svd_sigma[k] = ti.Vector([1.0, 1.0, 1.0])
            self.principal_stretches[k] = ti.Vector([1.0, 1.0, 1.0])
            self.is_degenerate[k] = 0
            self.is_near_rigid[k] = 0

    def _detect_boundary_nodes(self):
        """
        Automatic boundary detection (FEniCS-style 'on_boundary')
        A node is on the boundary if it belongs to at least one face that is shared by only one tetrahedron
        """
        # First pass: initialize boundary detection
        self._initialize_boundary_detection()
        
        # Second pass: detect boundary faces and mark boundary nodes
        self._mark_boundary_faces_and_nodes()
        
        print(f"Boundary detection complete: {self._count_boundary_nodes()} boundary nodes detected")

    @ti.kernel
    def _initialize_boundary_detection(self):
        """Initialize boundary detection fields"""
        for i in range(self.N):
            self.boundary_nodes[i] = 0
            self.boundary_displacement[i] = ti.Vector([0.0, 0.0, 0.0])
            self.is_boundary_constrained[i] = 0

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
    
    def apply_dirichlet_bc_on_boundary(self, displacement_field=None):
        """
        Apply Dirichlet boundary conditions on automatically detected boundary
        (FEniCS-style: u_bc_surface = fa.DirichletBC(self.V, u_bc, 'on_boundary'))
        
        Args:
            displacement_field: If provided, use this displacement field for boundary conditions
                              If None, use the existing displacement_field from 4DCT
        """
        if displacement_field is not None:
            # Update displacement field if provided
            self.displacement_field.from_numpy(displacement_field.astype(np.float64))
        
        # Apply boundary conditions to all detected boundary nodes
        self._apply_boundary_displacements()
        
        # Count constrained nodes
        constrained_count = self._count_constrained_nodes()
        print(f"Applied Dirichlet BC to {constrained_count} boundary nodes")

    @ti.kernel
    def _apply_boundary_displacements(self):
        """Apply displacement boundary conditions to detected boundary nodes"""
        for i in range(self.N):
            if self.boundary_nodes[i] == 1:  # This is a boundary node
                # Set prescribed displacement from 4DCT data
                self.boundary_displacement[i] = self.displacement_field[i]
                # Mark as constrained
                self.is_boundary_constrained[i] = 1
                # Update position to satisfy Dirichlet condition: q = X + u
                self.x[i] = self.initial_positions[i] + self.displacement_field[i]

    def _count_constrained_nodes(self):
        """Count number of nodes with Dirichlet constraints applied"""
        return int(self.is_boundary_constrained.to_numpy().sum())

    @ti.kernel
    def _assemble_stiffness_matrix(self):
        """
        Assemble global stiffness matrix from spring network
        K[3*i+d, 3*j+d'] = contribution from springs connecting nodes i and j
        """
        # Initialize stiffness matrix to zero
        for i, j in ti.ndrange(3*self.N, 3*self.N):
            self.stiffness_matrix[i, j] = 0.0
        
        # Add contributions from all springs in the network
        for k in range(self.M):
            if self.vol[k] > EPS_ZERO:
                self._add_axial_spring_stiffness(k)
                self._add_torsion_spring_stiffness(k)
                self._add_barycentric_spring_stiffness(k)

    @ti.func
    def _add_axial_spring_stiffness(self, k):
        """Add axial spring contributions to global stiffness matrix"""
        for axis_idx in ti.static(range(3)):
            pt1_idx = axis_idx * 2
            pt2_idx = axis_idx * 2 + 1
            
            if (self.intersection_valid[k, pt1_idx] == 1 and 
                self.intersection_valid[k, pt2_idx] == 1):
                
                # Get current spring properties
                p1 = self.intersection_points[k, pt1_idx]
                p2 = self.intersection_points[k, pt2_idx]
                axis_vector = p1 - p2
                current_length = axis_vector.norm()
                
                if current_length > EPS_ZERO:
                    # Spring stiffness and direction
                    k_spring = self.stiffness[k, axis_idx]
                    axis_hat = axis_vector.normalized()
                    
                    # For each vertex of the tetrahedron
                    for i in ti.static(range(4)):
                        for j in ti.static(range(4)):
                            # Get coefficients for this spring
                            c1_i = self.C_k[k, i, pt1_idx]
                            c2_i = self.C_k[k, i, pt2_idx]
                            c1_j = self.C_k[k, j, pt1_idx]
                            c2_j = self.C_k[k, j, pt2_idx]
                            
                            # Spring stiffness contribution: k * hat_axis ⊗ hat_axis
                            for d1 in ti.static(range(3)):
                                for d2 in ti.static(range(3)):
                                    stiff_contrib = k_spring * axis_hat[d1] * axis_hat[d2]
                                    
                                    node_i = self.tets[k][i]
                                    node_j = self.tets[k][j]
                                    
                                    # Assembly coefficient from intersection point interpolation
                                    assembly_coeff = (c1_i - c2_i) * (c1_j - c2_j)
                                    
                                    # Add to global stiffness matrix
                                    ti.atomic_add(self.stiffness_matrix[3*node_i + d1, 3*node_j + d2], 
                                                assembly_coeff * stiff_contrib)

    @ti.func  
    def _add_torsion_spring_stiffness(self, k):
        """Add torsion spring contributions to global stiffness matrix (simplified)"""
        # Simplified torsion stiffness - adds small coupling between axes
        base_torsion = 0.1 * self.stiffness[k, 0]  # Small torsion coupling
        
        for axis1 in ti.static(range(3)):
            for axis2 in ti.static(range(3)):
                if axis1 != axis2:
                    pt1_idx = axis1 * 2 
                    pt2_idx = axis2 * 2
                    
                    if (self.intersection_valid[k, pt1_idx] == 1 and 
                        self.intersection_valid[k, pt2_idx] == 1):
                        
                        # Add small off-diagonal coupling
                        for i in ti.static(range(4)):
                            for j in ti.static(range(4)):
                                c1 = self.C_k[k, i, pt1_idx]
                                c2 = self.C_k[k, j, pt2_idx]
                                
                                node_i = self.tets[k][i]
                                node_j = self.tets[k][j]
                                
                                # Add small coupling terms
                                for d in ti.static(range(3)):
                                    ti.atomic_add(self.stiffness_matrix[3*node_i + d, 3*node_j + d], 
                                                c1 * c2 * base_torsion)

    @ti.func
    def _add_barycentric_spring_stiffness(self, k):
        """Add barycentric volume spring contributions to global stiffness matrix"""
        if self.volume_preservation_enabled[k] == 1:
            k_barycentric = self.barycentric_ks[k]
            
            # Barycentric springs connect all nodes to the barycenter
            for i in ti.static(range(4)):
                for j in ti.static(range(4)):
                    node_i = self.tets[k][i]
                    node_j = self.tets[k][j]
                    
                    # Barycentric spring stiffness pattern
                    if i == j:
                        # Diagonal terms: stiffer
                        for d in ti.static(range(3)):
                            ti.atomic_add(self.stiffness_matrix[3*node_i + d, 3*node_j + d], 
                                        k_barycentric * 0.75)
                    else:
                        # Off-diagonal terms: coupling
                        for d in ti.static(range(3)):
                            ti.atomic_add(self.stiffness_matrix[3*node_i + d, 3*node_j + d], 
                                        -k_barycentric * 0.25)

    @ti.kernel
    def _compute_residual(self):
        """Compute residual vector r = g_int + F_ext (F_ext = 0 in our case)"""
        # Initialize residual to zero
        for i in range(self.N):
            self.residual[i] = ti.Vector([0.0, 0.0, 0.0])
        
        # Compute internal forces (same as existing method but store in residual)
        for k in range(self.M):
            if self.vol[k] > EPS_ZERO:
                self._add_residual_from_axial_springs(k)
                self._add_residual_from_torsion_springs(k)
                self._add_residual_from_barycentric_springs(k)

    @ti.func
    def _add_residual_from_axial_springs(self, k):
        """Add axial spring force contributions to residual"""
        for axis_idx in ti.static(range(3)):
            pt1_idx = axis_idx * 2
            pt2_idx = axis_idx * 2 + 1
            
            if (self.intersection_valid[k, pt1_idx] == 1 and 
                self.intersection_valid[k, pt2_idx] == 1):
                
                # Get intersection points
                p1 = self.intersection_points[k, pt1_idx]
                p2 = self.intersection_points[k, pt2_idx]
                
                # Current axis vector
                axis_vector = p1 - p2
                current_length = axis_vector.norm()
                
                if current_length > EPS_ZERO:
                    # Get rest length and stiffness
                    rest_length = self.rest_lengths[k, axis_idx]
                    k_spring = self.stiffness[k, axis_idx]
                    
                    # Calculate strain and force
                    strain = (current_length - rest_length) / rest_length if rest_length > EPS_ZERO else 0.0
                    spring_force_magnitude = k_spring * strain
                    force_direction = axis_vector.normalized()
                    
                    # Spring forces at intersection points
                    f1_spring = spring_force_magnitude * force_direction
                    f2_spring = -spring_force_magnitude * force_direction
                    
                    # Distribute to vertices using coefficient matrix  
                    for vertex_idx in ti.static(range(4)):
                        c1 = self.C_k[k, vertex_idx, pt1_idx]
                        c2 = self.C_k[k, vertex_idx, pt2_idx]
                        
                        vertex_force = c1 * f1_spring + c2 * f2_spring
                        node_idx = self.tets[k][vertex_idx]
                        
                        ti.atomic_add(self.residual[node_idx], vertex_force)

    @ti.func
    def _add_residual_from_torsion_springs(self, k):
        """Add torsion spring force contributions to residual (simplified)"""
        # Simplified torsion forces - small coupling between nodes
        for i in ti.static(range(4)):
            for j in ti.static(range(4)):
                if i != j:
                    node_i = self.tets[k][i]
                    node_j = self.tets[k][j]
                    
                    # Small torsion coupling force
                    pos_diff = self.x[node_i] - self.x[node_j]
                    torsion_force = -0.01 * self.torsion_stiffness[k, 0] * pos_diff
                    
                    ti.atomic_add(self.residual[node_i], torsion_force)

    @ti.func
    def _add_residual_from_barycentric_springs(self, k):
        """Add barycentric spring force contributions to residual"""
        if self.volume_preservation_enabled[k] == 1:
            # Compute current barycenter
            x_b_t = ti.Vector([0.0, 0.0, 0.0])
            for j in ti.static(range(4)):
                x_b_t += self.x[self.tets[k][j]]
            x_b_t /= 4.0
            
            # Compute current and rest lengths
            current_lengths = ti.Vector([0.0, 0.0, 0.0, 0.0])
            sum_current_lengths = 0.0
            sum_rest_lengths = 0.0
            
            for j in ti.static(range(4)):
                xi_j_t = x_b_t - self.x[self.tets[k][j]]
                current_lengths[j] = xi_j_t.norm()
                sum_current_lengths += current_lengths[j]
                sum_rest_lengths += self.barycentric_rest_lengths[k, j]
            
            # Length difference
            delta_L = sum_current_lengths - sum_rest_lengths
            k_s = self.barycentric_ks[k]
            
            # Apply forces
            for j in ti.static(range(4)):
                if current_lengths[j] > EPS_ZERO:
                    xi_j_t = x_b_t - self.x[self.tets[k][j]]
                    xi_j_hat = xi_j_t / current_lengths[j]
                    
                    # Barycentric spring force
                    f_j_barycentric = -k_s * delta_L * xi_j_hat
                    node_idx = self.tets[k][j]
                    
                    ti.atomic_add(self.residual[node_idx], f_j_barycentric)

    def solve_static_equilibrium(self, max_iterations=50, tolerance=1e-6):
        """
        Solve static equilibrium: g_int(q) + F_ext = 0 (F_ext = 0)
        Using Newton-Raphson with Taichi linear system solver
        """
        print(f"=== Static Equilibrium Solver ===")
        print(f"Max iterations: {max_iterations}, Tolerance: {tolerance}")
        
        for iteration in range(max_iterations):
            # Update intersections and coefficients for current configuration
            self._update_intersections_coefficients()
            
            # Assemble stiffness matrix and compute residual
            self._assemble_stiffness_matrix()
            self._compute_residual()
            
            # Check convergence
            residual_norm = self._compute_residual_norm()
            print(f"Iteration {iteration}: Residual norm = {residual_norm:.6e}")
            
            if residual_norm < tolerance:
                print(f"Converged in {iteration} iterations")
                return True
            
            # Apply boundary constraints and solve linear system
            self._apply_boundary_constraints_to_system()
            success = self._solve_linear_system_with_taichi()
            
            if not success:
                print(f"Linear solver failed at iteration {iteration}")
                return False
            
        
        print(f"Failed to converge in {max_iterations} iterations")
        return False

    def _compute_residual_norm(self):
        """Compute L2 norm of residual vector for free nodes"""
        residual_np = self.residual.to_numpy()
        constrained_np = self.is_boundary_constrained.to_numpy()
        
        # Only compute norm for free (unconstrained) nodes
        free_residual = residual_np[constrained_np == 0]
        return np.linalg.norm(free_residual.flatten())

    @ti.kernel  
    def _apply_boundary_constraints_to_system(self):
        """Apply Dirichlet boundary constraints to stiffness matrix and residual"""
        # Zero out rows and columns for constrained DOFs, set diagonal to 1
        for i in range(self.N):
            if self.is_boundary_constrained[i] == 1:
                for d in ti.static(range(3)):
                    dof = 3 * i + d
                    
                    # Zero out row and column
                    for j in range(3 * self.N):
                        self.stiffness_matrix[dof, j] = 0.0
                        self.stiffness_matrix[j, dof] = 0.0
                    
                    # Set diagonal to 1
                    self.stiffness_matrix[dof, dof] = 1.0
                    
                    # Set residual to zero (no change needed for constrained nodes)
                    self.residual[i][d] = 0.0

    def _solve_linear_system_with_taichi(self):
        """Solve K * Δq = -r using Taichi sparse solver"""
        try:
            # Convert dense matrix to sparse format
            K_sparse = ti.linalg.SparseMatrixBuilder(3*self.N, 3*self.N, max_num_triplets=9*self.N*self.N)
            self._build_sparse_matrix(K_sparse)
            A = K_sparse.build()
            
            # Flatten residual vector
            self._flatten_residual_to_vector(self.temp_b)
            
            # Initialize solution to zero
            self.temp_x.fill(0.0)
            
            # Create sparse solver
            solver = ti.linalg.SparseSolver(solver_type="LLT")
            solver.analyze_pattern(A)
            solver.factorize(A)
            
            # Solve the system
            solver.solve(self.temp_b, self.temp_x)
            
            # Convert solution back to 3D vector field
            self._unflatten_vector_to_solution(self.temp_x)
            
            # Update positions: x = x + Δx (for free nodes only)
            self._update_positions_from_solution()
            
            return True
            
        except Exception as e:
            print(f"Taichi sparse solver error: {e}")
            return False

    @ti.kernel
    def _build_sparse_matrix(self, builder: ti.types.sparse_matrix_builder()):
        """Build sparse matrix from dense stiffness matrix"""
        for i in range(3*self.N):
            for j in range(3*self.N):
                if abs(self.stiffness_matrix[i, j]) > 1e-12:
                    builder[i, j] += self.stiffness_matrix[i, j]

    @ti.kernel
    def _flatten_residual_to_vector(self, b: ti.template()):
        """Flatten residual vector to 1D for solver"""
        for i in range(self.N):
            b[3*i + 0] = -self.residual[i][0]
            b[3*i + 1] = -self.residual[i][1]
            b[3*i + 2] = -self.residual[i][2]

    @ti.kernel
    def _unflatten_vector_to_solution(self, x: ti.template()):
        """Convert 1D solution back to 3D vector field"""
        for i in range(self.N):
            self.solution_increment[i] = ti.Vector([
                x[3*i + 0], 
                x[3*i + 1], 
                x[3*i + 2]
            ])


    @ti.kernel
    def _update_positions_from_solution(self):
        """Update nodal positions: x = x + Δx (for free nodes only)"""
        for i in range(self.N):
            if self.is_boundary_constrained[i] == 0:  # Free node
                self.x[i] += self.solution_increment[i]
            # Constrained nodes remain at their prescribed positions


    def set_displacement_field(self, displacement_np):
        """Set displacement field from 4DCT registration data
        Args:
            displacement_np: numpy array of shape (N, 3) containing displacement vectors u_i
        """
        if displacement_np.shape != (self.N, 3):
            raise ValueError(f"Displacement field shape {displacement_np.shape} doesn't match mesh nodes {(self.N, 3)}")
        
        self.displacement_field.from_numpy(displacement_np.astype(np.float64))
        self._update_deformed_positions()
        self._compute_edge_matrices()
        self._compute_deformation_gradients()
        self._compute_anisotropy_axes_from_svd()

    @ti.kernel
    def _update_deformed_positions(self):
        """Compute deformed positions: x_i = X_i + u_i"""
        for i in range(self.N):
            self.deformed_positions[i] = self.initial_positions[i] + self.displacement_field[i]

    @ti.kernel 
    def _compute_edge_matrices(self):
        """Compute reference and deformed edge matrices D_m and d_x for each tetrahedron"""
        for k in range(self.M):
            if self.vol[k] > EPS_ZERO:
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
        """Compute deformation gradient α = d_x · D_m^(-1) for each tetrahedron using Taichi's built-in inverse"""
        for k in range(self.M):
            if self.vol[k] > EPS_ZERO:
                # Get reference edge matrix D_m
                D_m = self.reference_edge_matrix[k]
                d_x = self.deformed_edge_matrix[k]
                
                # Check for degeneracy using determinant
                det_D_m = D_m.determinant()
                
                if ti.abs(det_D_m) > EPS_ZERO:
                    # Compute D_m^(-1) using Taichi's built-in inverse
                    D_m_inv = D_m.inverse()
                    
                    # Compute deformation gradient: α = d_x · D_m^(-1)
                    alpha = d_x @ D_m_inv
                    self.deformation_gradient[k] = alpha
                    
                    # Check for degenerate deformation (negative determinant or extreme compression)
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
        """Compute anisotropy axes from deformation gradient using SVD: α = U·Σ·V^T"""
        for k in range(self.M):
            if self.vol[k] > EPS_ZERO and self.is_degenerate[k] == 0:
                # Get deformation gradient α
                alpha = self.deformation_gradient[k]
                
                # Perform SVD: α = U·Σ·V^T
                U, sigma, V = ti.svd(alpha)
                
                # Store SVD components
                self.svd_U[k] = U
                self.svd_V[k] = V
                
                # Extract singular values (diagonal of Σ matrix)
                # sigma is returned as a 3x3 diagonal matrix
                lambda0 = sigma[0, 0]  # Largest singular value
                lambda1 = sigma[1, 1]  # Second singular value  
                lambda2 = sigma[2, 2]  # Smallest singular value
                
                # Store principal stretches (sorted λ0 ≥ λ1 ≥ λ2)
                self.principal_stretches[k] = ti.Vector([lambda0, lambda1, lambda2])
                self.svd_sigma[k] = ti.Vector([lambda0, lambda1, lambda2])
                
                # Extract principal stretch directions from V matrix columns
                # V columns are orthonormal principal-stretch directions in reference frame
                v0 = ti.Vector([V[0, 0], V[1, 0], V[2, 0]])  # First column (largest stretch direction)
                v1 = ti.Vector([V[0, 1], V[1, 1], V[2, 1]])  # Second column
                v2 = ti.Vector([V[0, 2], V[1, 2], V[2, 2]])  # Third column (smallest stretch direction)
                
                # Ensure right-handed coordinate system: if (e0 × e1) · e2 < 0, swap e1 and e2
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
                
                # Set anisotropy axes (these will be used for intersection point computation)
                self.anisotropy_axes[k, 0] = v0  # e0: largest stretch direction
                self.anisotropy_axes[k, 1] = v1  # e1: second stretch direction  
                self.anisotropy_axes[k, 2] = v2  # e2: third stretch direction
                
                # Check for near-rigid motion (all stretches close to unity)
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
        """Handle degenerate or near-rigid tetrahedra using displacement-based fallback strategy"""
        # Get vertex indices
        v0_idx = self.tets[k][0]
        v1_idx = self.tets[k][1] 
        v2_idx = self.tets[k][2]
        v3_idx = self.tets[k][3]
        
        # Compute mean nodal displacement: ū = (1/4) * (u₀ + u₁ + u₂ + u₃)
        u_bar = (self.displacement_field[v0_idx] + self.displacement_field[v1_idx] + 
                 self.displacement_field[v2_idx] + self.displacement_field[v3_idx]) / 4.0
        
        u_bar_norm = u_bar.norm()
        eps_displacement = 1e-6
        
        # Initialize axes
        e0 = ti.Vector([1.0, 0.0, 0.0])
        e1 = ti.Vector([0.0, 1.0, 0.0])
        e2 = ti.Vector([0.0, 0.0, 1.0])
        
        if u_bar_norm > eps_displacement:
            # Case 1: Significant mean displacement - use displacement-based frame
            
            # First axis: e₀ = ū / |ū| (dominant displacement direction)
            e0 = u_bar.normalized()
            
            # Second axis: orthogonalize reference edge against e₀
            # Use first edge of tetrahedron: r = X₁ - X₀
            X0 = self.initial_positions[v0_idx]
            X1 = self.initial_positions[v1_idx]
            r = X1 - X0
            
            # Remove component along e₀: e₁ = (r - (r·e₀)e₀) / |r - (r·e₀)e₀|
            r_proj = r - r.dot(e0) * e0
            if r_proj.norm() > eps_displacement:
                e1 = r_proj.normalized()
            else:
                # Fallback: use world coordinate if projection is too small
                if ti.abs(e0[0]) < 0.9:
                    world_vec = ti.Vector([1.0, 0.0, 0.0])
                    e1 = (world_vec - world_vec.dot(e0) * e0).normalized()
                else:
                    world_vec = ti.Vector([0.0, 1.0, 0.0])
                    e1 = (world_vec - world_vec.dot(e0) * e0).normalized()
            
            # Third axis: e₂ = e₀ × e₁ (right-hand rule)
            e2 = e0.cross(e1)
        
        # Set anisotropy axes for all cases (either computed or default)
        self.anisotropy_axes[k, 0] = e0
        self.anisotropy_axes[k, 1] = e1  
        self.anisotropy_axes[k, 2] = e2

    @ti.kernel
    def _clear_intersections(self):
        """Clear only intersection data"""
        for k, i in ti.ndrange(self.M, 6):
            self.intersection_valid[k, i] = 0
            self.intersection_face[k, i] = -1
            self.intersection_points[k, i] = ti.Vector([0.0, 0.0, 0.0])
    
    @ti.kernel
    def _clear_coefficients(self):
        """Clear only coefficient data"""
        for k, i, j in ti.ndrange(self.M, 4, 6):
            self.C_k[k, i, j] = 0.0
    
    @ti.kernel
    def _clear_intersections_coefficients(self):
        """Clear intersection and coefficient data"""
        for k, i in ti.ndrange(self.M, 6):
            self.intersection_valid[k, i] = 0
            self.intersection_face[k, i] = -1
            self.intersection_points[k, i] = ti.Vector([0.0, 0.0, 0.0])
            
        for k, i, j in ti.ndrange(self.M, 4, 6):
            self.C_k[k, i, j] = 0.0
    
    def update_intersections(self):
        """Update only intersection points"""
        # Clear intersection data
        self._clear_intersections()
        
        # Recompute intersections
        self._compute_intersections()
    
    def update_coefficients(self):
        """Update only coefficient matrix"""
        # Clear coefficient data
        self._clear_coefficients()
        
        # Recompute coefficients
        self._compute_coefficients()
    
    def _update_intersections_coefficients(self):
        """Update intersections and coefficients by calling separate functions"""
        # Clear previous data
        self._clear_intersections_coefficients()
        
        # Recompute intersections and coefficients
        self._compute_intersections()
        self._compute_coefficients()

    @ti.func
    def _get_current_axis_direction_normalized(self, k, axis_idx):
        """Get current normalized axis direction from intersection points"""
        pt1_idx = axis_idx * 2
        pt2_idx = axis_idx * 2 + 1
        
        result = self.anisotropy_axes[k, axis_idx].normalized()  # Default fallback
        
        if (self.intersection_valid[k, pt1_idx] == 1 and 
            self.intersection_valid[k, pt2_idx] == 1):
            p1 = self.intersection_points[k, pt1_idx]
            p2 = self.intersection_points[k, pt2_idx]
            axis_vector = p1 - p2
            if axis_vector.norm() > EPS_ZERO:
                result = axis_vector.normalized()
        
        return result
    
    @ti.func
    def _get_axis_velocity_normalized(self, k, axis_idx):
        """Get directional change velocity of unit axis vector using improved formula"""
        pt1_idx = axis_idx * 2
        pt2_idx = axis_idx * 2 + 1
        
        result = ti.Vector([0.0, 0.0, 0.0])  # Default fallback
        
        if (self.intersection_valid[k, pt1_idx] == 1 and 
            self.intersection_valid[k, pt2_idx] == 1):
            # Get current axis vector and its length
            p1 = self.intersection_points[k, pt1_idx]
            p2 = self.intersection_points[k, pt2_idx]
            zeta = p1 - p2
            zeta_norm = zeta.norm()
            
            if zeta_norm > EPS_ZERO:
                # Unit axis vector: zeta = zeta/|zeta|
                zeta_hat = zeta / zeta_norm
                
                # Estimate velocities at intersection points using weighted vertex velocities
                v1 = ti.Vector([0.0, 0.0, 0.0])
                v2 = ti.Vector([0.0, 0.0, 0.0])
                
                for vertex_idx in ti.static(range(4)):
                    c1 = self.C_k[k, vertex_idx, pt1_idx]
                    c2 = self.C_k[k, vertex_idx, pt2_idx]
                    vertex_vel = self.vel[self.tets[k][vertex_idx]]
                    
                    v1 += c1 * vertex_vel
                    v2 += c2 * vertex_vel
                
                # Relative velocity: zeta = v1 - v2
                zeta_dot = v1 - v2
                
                # Compute directional change: zetȧ = (I - zeta zeta^T)/|zeta| * zeta
                # This gives the true directional change of the unit vector
                I_minus_zeta_zeta_T_times_zeta_dot = zeta_dot - zeta_hat.dot(zeta_dot) * zeta_hat
                result = I_minus_zeta_zeta_T_times_zeta_dot / zeta_norm
        
        return result
    
    @ti.func
    def _apply_torsion_springs(self, k):
        """Apply torsion spring forces for tetrahedron k (Eq. 2.44/2.45)"""
        # Get current normalized axis directions
        axis0_hat = self._get_current_axis_direction_normalized(k, 0)
        axis1_hat = self._get_current_axis_direction_normalized(k, 1)
        axis2_hat = self._get_current_axis_direction_normalized(k, 2)
        
        # Get axis velocities for damping
        axis0_vel = self._get_axis_velocity_normalized(k, 0)
        axis1_vel = self._get_axis_velocity_normalized(k, 1)
        axis2_vel = self._get_axis_velocity_normalized(k, 2)
        
        # Torsion spring 0: between axes 0 and 1
        current_cos_angle_01 = ti.max(-1.0, ti.min(1.0, axis0_hat.dot(axis1_hat)))
        rest_cos_angle_01 = self.rest_cos_angles[k, 0]
        cos_diff_01 = current_cos_angle_01 - rest_cos_angle_01
        k_01 = self.torsion_stiffness[k, 0]
        c_tau = self.torsion_damping
        
        # Elastic torsion forces
        torsion_elastic_0_from_1 = -k_01 * cos_diff_01 * axis1_hat  # Force on axis 0 from axis 1
        torsion_elastic_1_from_0 = -k_01 * cos_diff_01 * axis0_hat  # Force on axis 1 from axis 0
        
        # Damping forces: f_damp = -c_tau * ((zeta l·zeta m) + (zeta l·zeta m)) * zeta m
        # Simplified: use relative axis velocities for damping
        damping_factor_01 = axis0_hat.dot(axis1_vel) + axis0_vel.dot(axis1_hat)
        torsion_damp_0_from_1 = -c_tau * damping_factor_01 * axis1_hat
        torsion_damp_1_from_0 = -c_tau * damping_factor_01 * axis0_hat
        
        # Total forces
        torsion_force_0_from_1 = torsion_elastic_0_from_1 + torsion_damp_0_from_1
        torsion_force_1_from_0 = torsion_elastic_1_from_0 + torsion_damp_1_from_0
        
        # Apply forces for axes 0 and 1
        # For axis 0 (intersection points 0 and 1)
        if (self.intersection_valid[k, 0] == 1 and self.intersection_valid[k, 1] == 1):
            for vertex_idx in ti.static(range(4)):
                c1 = self.C_k[k, vertex_idx, 0]
                c2 = self.C_k[k, vertex_idx, 1]
                vertex_force = c1 * torsion_force_0_from_1 + c2 * (-torsion_force_0_from_1)
                node_idx = self.tets[k][vertex_idx]
                ti.atomic_add(self.force[node_idx], vertex_force)
        
        # For axis 1 (intersection points 2 and 3)
        if (self.intersection_valid[k, 2] == 1 and self.intersection_valid[k, 3] == 1):
            for vertex_idx in ti.static(range(4)):
                c1 = self.C_k[k, vertex_idx, 2]
                c2 = self.C_k[k, vertex_idx, 3]
                vertex_force = c1 * torsion_force_1_from_0 + c2 * (-torsion_force_1_from_0)
                node_idx = self.tets[k][vertex_idx]
                ti.atomic_add(self.force[node_idx], vertex_force)
        
        # Torsion spring 1: between axes 1 and 2
        current_cos_angle_12 = ti.max(-1.0, ti.min(1.0, axis1_hat.dot(axis2_hat)))
        rest_cos_angle_12 = self.rest_cos_angles[k, 1]
        cos_diff_12 = current_cos_angle_12 - rest_cos_angle_12
        k_12 = self.torsion_stiffness[k, 1]
        
        # Elastic torsion forces
        torsion_elastic_1_from_2 = -k_12 * cos_diff_12 * axis2_hat  # Force on axis 1 from axis 2
        torsion_elastic_2_from_1 = -k_12 * cos_diff_12 * axis1_hat  # Force on axis 2 from axis 1
        
        # Damping forces
        damping_factor_12 = axis1_hat.dot(axis2_vel) + axis1_vel.dot(axis2_hat)
        torsion_damp_1_from_2 = -c_tau * damping_factor_12 * axis2_hat
        torsion_damp_2_from_1 = -c_tau * damping_factor_12 * axis1_hat
        
        # Total forces
        torsion_force_1_from_2 = torsion_elastic_1_from_2 + torsion_damp_1_from_2
        torsion_force_2_from_1 = torsion_elastic_2_from_1 + torsion_damp_2_from_1
        
        # Apply forces for axes 1 and 2
        # For axis 1 (intersection points 2 and 3) - add to existing forces
        if (self.intersection_valid[k, 2] == 1 and self.intersection_valid[k, 3] == 1):
            for vertex_idx in ti.static(range(4)):
                c1 = self.C_k[k, vertex_idx, 2]
                c2 = self.C_k[k, vertex_idx, 3]
                vertex_force = c1 * torsion_force_1_from_2 + c2 * (-torsion_force_1_from_2)
                node_idx = self.tets[k][vertex_idx]
                ti.atomic_add(self.force[node_idx], vertex_force)
        
        # For axis 2 (intersection points 4 and 5)
        if (self.intersection_valid[k, 4] == 1 and self.intersection_valid[k, 5] == 1):
            for vertex_idx in ti.static(range(4)):
                c1 = self.C_k[k, vertex_idx, 4]
                c2 = self.C_k[k, vertex_idx, 5]
                vertex_force = c1 * torsion_force_2_from_1 + c2 * (-torsion_force_2_from_1)
                node_idx = self.tets[k][vertex_idx]
                ti.atomic_add(self.force[node_idx], vertex_force)
        
        # Torsion spring 2: between axes 2 and 0
        current_cos_angle_20 = ti.max(-1.0, ti.min(1.0, axis2_hat.dot(axis0_hat)))
        rest_cos_angle_20 = self.rest_cos_angles[k, 2]
        cos_diff_20 = current_cos_angle_20 - rest_cos_angle_20
        k_20 = self.torsion_stiffness[k, 2]
        
        # Elastic torsion forces
        torsion_elastic_2_from_0 = -k_20 * cos_diff_20 * axis0_hat  # Force on axis 2 from axis 0
        torsion_elastic_0_from_2 = -k_20 * cos_diff_20 * axis2_hat  # Force on axis 0 from axis 2
        
        # Damping forces
        damping_factor_20 = axis2_hat.dot(axis0_vel) + axis2_vel.dot(axis0_hat)
        torsion_damp_2_from_0 = -c_tau * damping_factor_20 * axis0_hat
        torsion_damp_0_from_2 = -c_tau * damping_factor_20 * axis2_hat
        
        # Total forces
        torsion_force_2_from_0 = torsion_elastic_2_from_0 + torsion_damp_2_from_0
        torsion_force_0_from_2 = torsion_elastic_0_from_2 + torsion_damp_0_from_2
        
        # Apply forces for axes 2 and 0
        # For axis 2 (intersection points 4 and 5) - add to existing forces
        if (self.intersection_valid[k, 4] == 1 and self.intersection_valid[k, 5] == 1):
            for vertex_idx in ti.static(range(4)):
                c1 = self.C_k[k, vertex_idx, 4]
                c2 = self.C_k[k, vertex_idx, 5]
                vertex_force = c1 * torsion_force_2_from_0 + c2 * (-torsion_force_2_from_0)
                node_idx = self.tets[k][vertex_idx]
                ti.atomic_add(self.force[node_idx], vertex_force)
        
        # For axis 0 (intersection points 0 and 1) - add to existing forces
        if (self.intersection_valid[k, 0] == 1 and self.intersection_valid[k, 1] == 1):
            for vertex_idx in ti.static(range(4)):
                c1 = self.C_k[k, vertex_idx, 0]
                c2 = self.C_k[k, vertex_idx, 1]
                vertex_force = c1 * torsion_force_0_from_2 + c2 * (-torsion_force_0_from_2)
                node_idx = self.tets[k][vertex_idx]
                ti.atomic_add(self.force[node_idx], vertex_force)
    
    @ti.func
    def _apply_axial_springs(self, k):
        """Apply axial spring forces for tetrahedron k (Eq. 2.35 to 2.29)"""
        for axis_idx in ti.static(range(3)):
            pt1_idx = axis_idx * 2
            pt2_idx = axis_idx * 2 + 1
            
            if (self.intersection_valid[k, pt1_idx] == 1 and 
                self.intersection_valid[k, pt2_idx] == 1):
                
                # Get intersection points
                p1 = self.intersection_points[k, pt1_idx]
                p2 = self.intersection_points[k, pt2_idx]
                
                # Current axis vector
                axis_vector = p1 - p2
                current_length = axis_vector.norm()
                
                if current_length > EPS_ZERO:
                    # Get rest length
                    rest_length = self.rest_lengths[k, axis_idx]
                    
                    # Calculate strain (deformation from rest state)
                    # Using strain form: f = -k * (|zeta| - |zeta^0|)/|zeta^0| * hat zeta
                    # where |zeta| is current length, |zeta^0| is rest length,
                    # and hat zeta is the normalized axis vector.
                    # This rescales effective k by 1/|zeta^0| compared to raw length difference
                    strain = (current_length - rest_length) / rest_length if rest_length > EPS_ZERO else 0.0
                    
                    # Calculate spring force magnitude (Hooke's law)
                    spring_force_magnitude = self.stiffness[k, axis_idx] * strain
                    
                    # Force direction (normalized current axis)
                    force_direction = axis_vector.normalized()
                    
                    # Spring forces
                    f1_spring = spring_force_magnitude * force_direction
                    f2_spring = -spring_force_magnitude * force_direction
                    
                    # Distribute to vertices using coefficient matrix
                    for vertex_idx in ti.static(range(4)):
                        c1 = self.C_k[k, vertex_idx, pt1_idx]
                        c2 = self.C_k[k, vertex_idx, pt2_idx]
                        
                        vertex_force = c1 * f1_spring + c2 * f2_spring
                        
                        node_idx = self.tets[k][vertex_idx]
                        ti.atomic_add(self.force[node_idx], vertex_force)
    
    @ti.func
    def _apply_barycentric_volume_springs(self, k):
        """Apply barycentric volume springs for tetrahedron k (Eq. 2.76/2.77)"""
        delta_L = 0.0
        sum_current_lengths = 0.0
        
        if (self.volume_preservation_enabled[k] == 1 or self.volume_control_enabled[k] == 1):
            # Compute current barycenter: x_b^t = (1/4) * sum x_j^t
            x_b_t = ti.Vector([0.0, 0.0, 0.0])
            for j in ti.static(range(4)):
                x_b_t += self.x[self.tets[k][j]]
            x_b_t /= 4.0
            
            # Compute current lengths: |xi_j^t| = |x_b^t - x_j^t|
            current_lengths = ti.Vector([0.0, 0.0, 0.0, 0.0])
            sum_current_lengths = 0.0
            for j in ti.static(range(4)):
                xi_j_t = x_b_t - self.x[self.tets[k][j]]
                current_lengths[j] = xi_j_t.norm()
                sum_current_lengths += current_lengths[j]
            
            # Compute rest lengths sum
            sum_rest_lengths = 0.0
            for j in ti.static(range(4)):
                sum_rest_lengths += self.barycentric_rest_lengths[k, j]
            
            # Length difference: delta L = sum|xi_j^t| - sum|xi_j^0|
            delta_L = sum_current_lengths - sum_rest_lengths
            
            # Compute barycenter velocity for damping
            v_b_t = ti.Vector([0.0, 0.0, 0.0])
            for j in ti.static(range(4)):
                v_b_t += self.vel[self.tets[k][j]]
            v_b_t /= 4.0
            
            # Apply forces: f_j = -k_s * delta L * (xi_j^t / |xi_j^t|) + damping
            k_s = self.barycentric_ks[k]
            c = self.barycentric_damping
            for j in ti.static(range(4)):
                if current_lengths[j] > EPS_ZERO:
                    xi_j_t = x_b_t - self.x[self.tets[k][j]]
                    xi_j_hat = xi_j_t / current_lengths[j]  # Unit vector
                    
                    # Elastic barycentric spring force
                    f_j_elastic = -k_s * delta_L * xi_j_hat
                    
                    # Dashpot damping force: f_damp = -c * (v_j - v_b)
                    node_idx = self.tets[k][j]
                    v_rel = self.vel[node_idx] - v_b_t
                    f_j_damp = -c * v_rel
                    
                    # Total force
                    f_j_total = f_j_elastic + f_j_damp
                    
                    # Apply force to vertex
                    ti.atomic_add(self.force[node_idx], f_j_total)
        
        return delta_L, sum_current_lengths
    
    @ti.func  
    def _adapt_stiffness(self, k, delta_V, sum_current_lengths):
        """Adapt barycentric spring stiffness using LMS (Eq. 2.81)"""
        if (self.volume_preservation_enabled[k] == 1 or self.volume_control_enabled[k] == 1):
            # Adaptive update: k_s^{t+h} = k_s^t + mu * delta V * sum|xi_j^t|
            mu = self.barycentric_mu
            self.barycentric_ks[k] += mu * delta_V * sum_current_lengths
            
            # Clamp to reasonable bounds
            self.barycentric_ks[k] = ti.max(0.01, ti.min(1000.0, self.barycentric_ks[k]))

    @ti.kernel
    def _update_current_volumes(self):
        """Update current volume for each tetrahedron"""
        for k in range(self.M):
            if self.vol[k] > 1e-12:
                v0 = self.x[self.tets[k][0]]
                v1 = self.x[self.tets[k][1]]
                v2 = self.x[self.tets[k][2]]
                v3 = self.x[self.tets[k][3]]
                
                vol = ti.abs((v1 - v0).dot((v2 - v0).cross(v3 - v0))) / 6.0
                self.current_volume[k] = vol

    @ti.kernel
    def _compute_internal_forces(self):
        """Compute internal forces: axial springs (2.4.2.1) + barycentric volume springs (2.4.2.3/2.4.2.4)"""
        # Clear forces
        for i in range(self.N):
            self.force[i] = ti.Vector([0.0, 0.0, 0.0])
        
        # Compute forces for each tetrahedron
        for k in range(self.M):
            if self.vol[k] > EPS_ZERO:
                # —— 2.4.2.1: Axial spring forces (Eq. 2.35→2.29) ——
                self._apply_axial_springs(k)
                
                # —— 2.4.2.1: Torsion spring forces (Eq. 2.44/2.45) ——
                self._apply_torsion_springs(k)
                
                # —— 2.4.2.3/2.4.2.4: Barycentric volume springs (Eq. 2.76/2.77) ——
                _, sum_current_lengths = self._apply_barycentric_volume_springs(k)
                
                # —— Adaptive LMS update for volume spring stiffness (Eq. 2.81) ——
                # Use absolute volume error: delta V = V - V₀ (in m³)
                V = self.current_volume[k]
                V0 = self.initial_volume[k]
                delta_V = V - V0  # Absolute volume error as per Eq. 2.81
                
                self._adapt_stiffness(k, delta_V, sum_current_lengths)

    @ti.kernel
    def _add_gravity(self):
        """Add gravity forces"""
        gravity = ti.Vector([0.0, -9, 0.0])
        for i in range(self.N):
            self.force[i] += self.mass[i] * gravity



    def compute_forces(self):
        """Compute all forces"""
        self._update_intersections_coefficients()
        self._update_current_volumes()
        self._compute_internal_forces()
        self._add_gravity()


    def enable_volume_preservation(self, bulk_modulus=1e4):
        """Enable volume preservation for all tetrahedra"""
        @ti.kernel
        def set_volume_preservation(p: float):
            for k in range(self.M):
                self.volume_preservation_enabled[k] = 1
                self.volume_control_enabled[k] = 0
                self.bulk_modulus[k] = p
        
        set_volume_preservation(bulk_modulus)

    def enable_volume_control(self, target_ratio=1.15, bulk_modulus=1e4):
        """Enable volume control for all tetrahedra"""
        @ti.kernel
        def set_volume_control(r: float, p: float):
            for k in range(self.M):
                self.volume_preservation_enabled[k] = 0
                self.volume_control_enabled[k] = 1
                self.target_volume_ratio[k] = r
                self.bulk_modulus[k] = p
        
        set_volume_control(target_ratio, bulk_modulus)

    def get_volume_change_stats(self):
        """Get statistics about volume changes"""
        current_vols = self.current_volume.to_numpy()
        initial_vols = self.initial_volume.to_numpy()
        
        rel_changes = (current_vols - initial_vols) / initial_vols
        
        return {
            'mean_relative_change': np.mean(rel_changes),
            'max_relative_change': np.max(np.abs(rel_changes)),
            'total_volume_change': np.sum(current_vols) - np.sum(initial_vols)
        }


    def get_intersection_stats(self):
        """Get intersection statistics"""
        valid_data = self.intersection_valid.to_numpy()
        
        total_possible = self.M * 6
        total_valid = np.sum(valid_data)
        success_rate = total_valid / total_possible if total_possible > 0 else 0.0
        
        per_tet_valid = np.sum(valid_data, axis=1)
        per_tet_average = np.mean(per_tet_valid)
        
        return {
            'success_rate': success_rate,
            'per_tet_average': per_tet_average,
            'total_valid': total_valid,
            'total_possible': total_possible
        }

    def summary(self):
        """Print simulation summary"""
        total_mass = self.mass.to_numpy().sum()
        print(f"#nodes={self.N}, #tets={self.M}, total mass={total_mass:.3f} kg")
        
        stats = self.get_intersection_stats()
        print(f"Intersection success rate: {stats['success_rate']:.2%}")
        print(f"Average intersections per tet: {stats['per_tet_average']:.1f}/6")
        
        print(f"\nPure Mass-Spring mechanics:")
        print(f"  Axial spring stiffness: {self.stiffness.to_numpy().mean():.3f} ± {self.stiffness.to_numpy().std():.3f} N/m")
        print(f"  Torsion spring stiffness: {self.torsion_stiffness.to_numpy().mean():.3f} ± {self.torsion_stiffness.to_numpy().std():.3f} N⋅m/rad")
        print(f"  Barycentric spring stiffness: {self.barycentric_ks.to_numpy().mean():.3f} ± {self.barycentric_ks.to_numpy().std():.3f} N/m")
        print(f"  Volume preservation enabled: {self.volume_preservation_enabled.to_numpy().sum()}/{self.M} tets")


def load_mesh_from_file(mesh_path):
    """Load mesh from file"""
    if not os.path.exists(mesh_path):
        raise FileNotFoundError(f"Mesh file not found: {mesh_path}")
    
    try:
        mesh = meshio.read(mesh_path)
        print(f"Successfully loaded mesh from: {mesh_path}")
        
        pts = mesh.points
        
        tid = next((i for i, c in enumerate(mesh.cells) if c.type == "tetra"), None)
        if tid is None:
            raise ValueError("No tetrahedra found in mesh file")
        
        tets = mesh.cells[tid].data
        
        if "c_labels" not in mesh.cell_data:
            print("Warning: No cell labels found, using default label 0")
            lbls = np.zeros(tets.shape[0], dtype=np.int32)
        else:
            lbls = mesh.cell_data["c_labels"][tid]
        
        print(f"Mesh data: {pts.shape[0]} points, {tets.shape[0]} tetrahedra")
        return pts, tets, lbls
        
    except Exception as e:
        raise RuntimeError(f"Error loading mesh file: {str(e)}")


def load_mesh_with_displacement(mesh_path, displacement_path=None):
    """
    Load mesh and optionally apply displacement field
    
    Args:
        mesh_path: Path to mesh file
        displacement_path: Optional path to NIfTI displacement field
        
    Returns:
        tuple: (pts, tets, lbls, displacement_vectors)
    """
    from displacement_loader import load_and_interpolate_displacement
    
    if displacement_path is not None and os.path.exists(displacement_path):
        print(f"Loading mesh with displacement field from: {displacement_path}")
        pts, tets, lbls, displacement_vectors = load_and_interpolate_displacement(
            displacement_path, mesh_path
        )
        return pts, tets, lbls, displacement_vectors
    else:
        # Load mesh without displacement
        pts, tets, lbls = load_mesh_from_file(mesh_path)
        displacement_vectors = np.zeros((pts.shape[0], 3))  # Zero displacement
        return pts, tets, lbls, displacement_vectors


def main():
    """Main function with 4DCT displacement field integration"""
    import os
    
    # Update paths to use the proper Case1Pack data
    path = os.path.dirname(os.path.abspath(__file__))
    mesh_path = os.path.join(path, "./data/Case1Pack/pygalmesh/case1_T00_lung_regions_11.xdmf")
    displacement_path = os.path.join(path, "./data/Case1Pack/CorrField/case1_T00_T50.nii.gz")
    
    # Load mesh with displacement field
    print("=== Loading Mesh and 4DCT Displacement Field ===")
    pts, tets, lbls, displacement_vectors = load_mesh_with_displacement(mesh_path, displacement_path)
    
    # Create simulation
    sim = Adamss(pts, tets, lbls)
    
    # Set the displacement field to compute anisotropy axes from 4DCT data
    print("=== Integrating 4DCT Displacement Data ===")
    sim.set_displacement_field(displacement_vectors)
    
    # Apply FEniCS-style automatic boundary conditions
    print("=== Applying FEniCS-style Dirichlet BC on Boundary ===")
    sim.apply_dirichlet_bc_on_boundary()
    
    # Enable volume preservation
    sim.enable_volume_preservation(bulk_modulus=1e2)  # Reduced for stability
    
    sim.summary()
    
    # Print displacement field statistics
    displ_magnitudes = np.linalg.norm(displacement_vectors, axis=1)
    print(f"\n4DCT Displacement Field Statistics:")
    print(f"  Max displacement: {displ_magnitudes.max():.6f} m")
    print(f"  Mean displacement: {displ_magnitudes.mean():.6f} m")
    print(f"  Nodes with displacement > 1mm: {np.sum(displ_magnitudes > 1e-3)}/{len(displ_magnitudes)}")
    
    # Solve static equilibrium instead of dynamic simulation
    print("\n=== Solving Static Equilibrium with 4DCT Boundary Conditions ===")
    sim.solve_static_equilibrium(max_iterations=50, tolerance=1e-4)

    # Final statistics
    final_vol_stats = sim.get_volume_change_stats()
    print(f"\nFinal volume preservation statistics:")
    print(f"  Mean relative volume change: {final_vol_stats['mean_relative_change']:.4%}")
    print(f"  Max relative volume change: {final_vol_stats['max_relative_change']:.4%}")
    print(f"  Total volume change: {final_vol_stats['total_volume_change']:.6f} m³")


if __name__ == "__main__":
    main()