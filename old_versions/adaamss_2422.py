"""
Continuum mechanics approach (2.4.2.2) - Virtual Hexahedron Method with Volume Preservation/Control

This implementation follows Section 2.4.2 of the ADAMSS documentation, including:
- Section 2.4.2.2: Deformation forces using continuum mechanics
- Section 2.4.2.3: Volume preservation (Eq. 2.86-2.100)
- Section 2.4.2.4: Volume control (Eq. 2.101-2.106)

Key features:
- Deformation tensor F using shape functions method (Eq. 2.68-2.72)
- Green strain tensor E = 0.5 * (F^T * F - I)
- Neo-Hookean stress for lung tissue
- Volume preservation/control stress using Ω = det(2E + I)
- Force calculation via virtual hexahedron method (Eq. 2.74)
"""

import taichi as ti
import meshio
import numpy as np
import sys
import os

ti.init(arch=ti.cuda, debug=True)

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

        # Core Taichi fields -------------------------------------------------
        self.x     = ti.Vector.field(3, ti.f64, shape=self.N)     # positions
        self.tets  = ti.Vector.field(4, ti.i32, shape=self.M)     # indices
        self.label = ti.field(ti.i32, shape=self.M)               # region id
        self.rho   = ti.field(ti.f64, shape=self.M)               # density
        self.vol   = ti.field(ti.f64, shape=self.M)               # volume
        self.mass  = ti.field(ti.f64, shape=self.N)               # lumped mass
        
        # Anisotropy and intersection fields
        self.anisotropy_axes = ti.Vector.field(3, ti.f64, shape=(self.M, 3))      # 3 axes per tet
        self.intersection_points = ti.Vector.field(3, ti.f64, shape=(self.M, 6))  # 6 points per tet
        self.intersection_valid = ti.field(ti.i32, shape=(self.M, 6))             # validity flag
        self.intersection_face = ti.field(ti.i32, shape=(self.M, 6))              # face index
        self.C_k = ti.field(ti.f64, shape=(self.M, 4, 6))                        # Coefficient matrix
        
        # Force and dynamics fields
        self.force = ti.Vector.field(3, ti.f64, shape=self.N)    # forces on nodes
        self.vel = ti.Vector.field(3, ti.f64, shape=self.N)      # velocities
        self.rest_lengths = ti.field(ti.f64, shape=(self.M, 3))  # rest lengths for each axis
        self.damping = 0.95                                      # Increase damping from 0.98 to 0.95
        
        # Continuum mechanics fields
        self.deformation_tensor = ti.Matrix.field(3, 3, ti.f64, shape=self.M)  # F tensor
        self.strain_tensor = ti.Matrix.field(3, 3, ti.f64, shape=self.M)       # E tensor
        self.stress_tensor = ti.Matrix.field(3, 3, ti.f64, shape=self.M)       # S tensor
        self.material_params = ti.Vector.field(2, ti.f64, shape=self.M)        # [mu, lambda]
        
        # Volume preservation/control fields (Section 2.4.2.3 & 2.4.2.4)
        self.volume_preservation_enabled = ti.field(ti.i32, shape=self.M)  # Enable flag
        self.volume_control_enabled = ti.field(ti.i32, shape=self.M)       # Volume control flag
        self.bulk_modulus = ti.field(ti.f64, shape=self.M)                # p parameter (Pa)
        self.target_volume_ratio = ti.field(ti.f64, shape=self.M)         # r = V^∞/V^0
        self.current_volume = ti.field(ti.f64, shape=self.M)              # Current volume
        
        # Initial configuration storage
        self.initial_positions = ti.Vector.field(3, ti.f64, shape=self.N)         # Initial vertices
        self.initial_volume = ti.field(ti.f64, shape=self.M)                      # Initial volumes
        self.initial_intersection_points = ti.Vector.field(3, ti.f64, shape=(self.M, 6))
        self.initial_axis_vectors = ti.Vector.field(3, ti.f64, shape=(self.M, 3))

        # Copy data from NumPy
        self.x.from_numpy(pts_np.astype(np.float64) * 1e-3)  # mm → m
        self.tets.from_numpy(tets_np.astype(np.int32))
        self.label.from_numpy(labels_np.astype(np.int32))

        # Validate mesh
        self._validate_mesh()

        # Pre-compute density
        rho_np = np.vectorize(lambda lbl: DEFAULT_RHO.get(int(lbl), DEFAULT_RHO[0]),
                              otypes=[np.float64])(labels_np)
        self.rho.from_numpy(rho_np)

        # Initialize simulation
        self._compute_tet_volume()
        self._check_degenerate_tets()
        self._compute_node_mass()
        self._initialize_anisotropy()
        self._initialize_arrays()
        self._compute_intersections()
        self._compute_coefficients()
        self._initialize_material_params()
        self._initialize_volume_preservation()
        self._initialize_dynamics()
        self._store_initial_configuration()
        self._store_rest_lengths()

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
        """Initialize anisotropy axes"""
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
    def _initialize_material_params(self):
        """Initialize Neo-Hookean material parameters for lung tissue"""
        for k in range(self.M):
            # Neo-Hookean parameters for lung tissue
            mu = 2000.0           # Shear modulus (Pa)
            lambda_param = 3000.0 # Lame's first parameter (Pa)
            
            # Adjust based on tissue type
            if self.label[k] == 1:  # Air-rich tissue
                mu *= 0.2
                lambda_param *= 0.2
            elif self.label[k] == 2:  # Fibrotic tissue
                mu *= 3.0
                lambda_param *= 3.0
            
            self.material_params[k] = ti.Vector([mu, lambda_param])

    @ti.kernel
    def _initialize_volume_preservation(self):
        """Initialize volume preservation parameters"""
        for k in range(self.M):
            self.volume_preservation_enabled[k] = 1  # Enable by default
            self.volume_control_enabled[k] = 0       # Disabled by default
            self.bulk_modulus[k] = 1e4               # 10 kPa default
            self.target_volume_ratio[k] = 1.0        # Preserve initial volume
            self.current_volume[k] = self.vol[k]

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
                            denom = ray_dir.dot(face_normal)
                            if ti.abs(denom) > EPS_PARALLEL:
                                t = (va - barycenter).dot(face_normal) / denom
                                p = barycenter + t * ray_dir
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
    def _update_intersections_coefficients(self):
        """Update intersections and coefficients combined"""
        # Clear previous data
        for k, i in ti.ndrange(self.M, 6):
            self.intersection_valid[k, i] = 0
            self.intersection_face[k, i] = -1
            
        for k, i, j in ti.ndrange(self.M, 4, 6):
            self.C_k[k, i, j] = 0.0
        
        # Recompute intersections
        for k in range(self.M):
            if self.vol[k] > 1e-12:
                # Get current vertices
                v0 = self.x[self.tets[k][0]]
                v1 = self.x[self.tets[k][1]]
                v2 = self.x[self.tets[k][2]]
                v3 = self.x[self.tets[k][3]]
                
                # Calculate current barycenter
                barycenter = (v0 + v1 + v2 + v3) / 4.0
                
                # Process each axis
                for axis_idx in ti.static(range(3)):
                    ray_dir = self.anisotropy_axes[k, axis_idx]
                    intersections_found = 0
                    
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
                            denom = ray_dir.dot(face_normal)
                            if ti.abs(denom) > EPS_PARALLEL:
                                t = (va - barycenter).dot(face_normal) / denom
                                p = barycenter + t * ray_dir
                                if self.point_in_triangle(p, va, vb, vc):
                                    if intersections_found < 2:
                                        idx = axis_idx * 2 + intersections_found
                                        self.intersection_points[k, idx] = p
                                        self.intersection_valid[k, idx] = 1
                                        self.intersection_face[k, idx] = face_idx
                                        intersections_found += 1

                # Update coefficients for new intersections
                for pt_idx in range(6):
                    if self.intersection_valid[k, pt_idx] == 1:
                        p = self.intersection_points[k, pt_idx]
                        face_idx = self.intersection_face[k, pt_idx]
                        
                        if face_idx == 0:
                            u, v, w = self.compute_barycentric(p, v0, v1, v2)
                            self.C_k[k, 0, pt_idx] = w
                            self.C_k[k, 1, pt_idx] = v
                            self.C_k[k, 2, pt_idx] = u
                            self.C_k[k, 3, pt_idx] = 0.0
                        elif face_idx == 1:
                            u, v, w = self.compute_barycentric(p, v0, v2, v3)
                            self.C_k[k, 0, pt_idx] = w
                            self.C_k[k, 1, pt_idx] = 0.0
                            self.C_k[k, 2, pt_idx] = v
                            self.C_k[k, 3, pt_idx] = u
                        elif face_idx == 2:
                            u, v, w = self.compute_barycentric(p, v0, v3, v1)
                            self.C_k[k, 0, pt_idx] = w
                            self.C_k[k, 1, pt_idx] = u
                            self.C_k[k, 2, pt_idx] = 0.0
                            self.C_k[k, 3, pt_idx] = v
                        else:
                            u, v, w = self.compute_barycentric(p, v1, v3, v2)
                            self.C_k[k, 0, pt_idx] = 0.0
                            self.C_k[k, 1, pt_idx] = w
                            self.C_k[k, 2, pt_idx] = u
                            self.C_k[k, 3, pt_idx] = v

    @ti.func
    def _compute_deformation_tensor(self, k):
        """Compute deformation tensor F using shape functions (Eq. 2.68-2.72)"""
        # Get current vertices
        x0 = self.x[self.tets[k][0]]
        x1 = self.x[self.tets[k][1]]
        x2 = self.x[self.tets[k][2]]
        x3 = self.x[self.tets[k][3]]
        
        # Get initial vertices
        X0 = self.initial_positions[self.tets[k][0]]
        X1 = self.initial_positions[self.tets[k][1]]
        X2 = self.initial_positions[self.tets[k][2]]
        X3 = self.initial_positions[self.tets[k][3]]
        
        # Current configuration matrix
        x_mat = ti.Matrix.zero(ti.f64, 3, 3)
        x_mat[0, 0] = x1[0] - x0[0]
        x_mat[0, 1] = x2[0] - x0[0]
        x_mat[0, 2] = x3[0] - x0[0]
        x_mat[1, 0] = x1[1] - x0[1]
        x_mat[1, 1] = x2[1] - x0[1]
        x_mat[1, 2] = x3[1] - x0[1]
        x_mat[2, 0] = x1[2] - x0[2]
        x_mat[2, 1] = x2[2] - x0[2]
        x_mat[2, 2] = x3[2] - x0[2]
        
        # Initial configuration matrix
        X_mat = ti.Matrix.zero(ti.f64, 3, 3)
        X_mat[0, 0] = X1[0] - X0[0]
        X_mat[0, 1] = X2[0] - X0[0]
        X_mat[0, 2] = X3[0] - X0[0]
        X_mat[1, 0] = X1[1] - X0[1]
        X_mat[1, 1] = X2[1] - X0[1]
        X_mat[1, 2] = X3[1] - X0[1]
        X_mat[2, 0] = X1[2] - X0[2]
        X_mat[2, 1] = X2[2] - X0[2]
        X_mat[2, 2] = X3[2] - X0[2]
        
        # Compute F = x_mat * X_mat^(-1)
        X_mat_det = X_mat.determinant()
        F = ti.Matrix.identity(ti.f64, 3)
        
        if ti.abs(X_mat_det) > EPS_ZERO:
            X_mat_inv = X_mat.inverse()
            F = x_mat @ X_mat_inv
        
        return F

    @ti.func
    def _compute_green_strain(self, F):
        """Compute Green strain tensor E = 0.5 * (F^T * F - I)"""
        FT = F.transpose()
        FTF = FT @ F
        I = ti.Matrix.identity(ti.f64, 3)
        E = 0.5 * (FTF - I)
        return E

    @ti.func
    def _compute_stress_tensor_lung(self, E, k):
        """Neo-Hookean model for lung tissue"""
        mu = self.material_params[k][0]
        lambda_param = self.material_params[k][1]
        
        trace_E = E.trace()
        I = ti.Matrix.identity(ti.f64, 3)
        
        # Strain limiting for stability
        max_strain = 0.3
        E_limited = ti.Matrix.zero(ti.f64, 3, 3)
        
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                E_limited[i,j] = ti.max(-max_strain, ti.min(max_strain, E[i,j]))
        trace_E = E_limited.trace()
        S = 2.0 * mu * E_limited + lambda_param * trace_E * I
        return S
        
    @ti.func
    def _compute_omega(self, E):
        """Compute Ω = det(2E + I) according to Eq. 2.99"""
        E11 = E[0, 0]
        E22 = E[1, 1]
        E33 = E[2, 2]
        E12 = E[0, 1]
        E13 = E[0, 2]
        E23 = E[1, 2]
        E21 = E[1, 0]
        E31 = E[2, 0]
        E32 = E[2, 1]
        
        omega = ((2.0 * E11 + 1.0) * (2.0 * E22 + 1.0) * (2.0 * E33 + 1.0) +
                 8.0 * (E12 * E23 * E31) +
                 8.0 * (E21 * E32 * E13) -
                 4.0 * (E23 * E32 * (2.0 * E11 + 1.0)) -
                 4.0 * (E13 * E31 * (2.0 * E22 + 1.0)) -
                 4.0 * (E12 * E21 * (2.0 * E33 + 1.0)))
        
        return omega

    @ti.func
    def _compute_d_omega_d_E(self, E, i, j):
        """Compute ∂Ω/∂E_ij according to Eq. 2.101"""
        E11 = E[0, 0]
        E22 = E[1, 1]
        E33 = E[2, 2]
        E12 = E[0, 1]
        E13 = E[0, 2]
        E23 = E[1, 2]
        E21 = E[1, 0]
        E31 = E[2, 0]
        E32 = E[2, 1]
        
        d_omega = 0.0
        
        if i == j:  # Diagonal terms
            if i == 0:
                d_omega = 2.0 * (2.0 * E22 + 1.0) * (2.0 * E33 + 1.0) - 8.0 * E23 * E32
            elif i == 1:
                d_omega = 2.0 * (2.0 * E11 + 1.0) * (2.0 * E33 + 1.0) - 8.0 * E13 * E31
            else:  # i == 2
                d_omega = 2.0 * (2.0 * E11 + 1.0) * (2.0 * E22 + 1.0) - 8.0 * E12 * E21
        else:  # Off-diagonal terms
            if (i == 0 and j == 1) or (i == 1 and j == 0):
                if i == 0:  # E12
                    d_omega = 8.0 * (E23 * E31) - 4.0 * E21 * (2.0 * E33 + 1.0)  
                else:  # E21
                    d_omega = 8.0 * (E32 * E13) - 4.0 * E12 * (2.0 * E33 + 1.0)  
            elif (i == 0 and j == 2) or (i == 2 and j == 0):
                if i == 0:  # E13
                    d_omega = 8.0 * (E21 * E32) - 4.0 * E31 * (2.0 * E22 + 1.0)  
                else:  # E31
                    d_omega = 8.0 * (E12 * E23) - 4.0 * E13 * (2.0 * E22 + 1.0)  
            else:  # (i == 1 and j == 2) or (i == 2 and j == 1)
                if i == 1:  # E23
                    d_omega = 8.0 * (E31 * E12) - 4.0 * E32 * (2.0 * E11 + 1.0)  
                else:  # E32
                    d_omega = 8.0 * (E13 * E21) - 4.0 * E23 * (2.0 * E11 + 1.0)  

        return d_omega

    @ti.func
    def _compute_volume_preservation_stress(self, E, k):
        """Compute volume preservation stress (Eq. 2.100)"""
        p = self.bulk_modulus[k]
        omega = self._compute_omega(E)
        
        S_vol = ti.Matrix.zero(ti.f64, 3, 3)
        
        if ti.abs(omega) > EPS_ZERO:
            sqrt_omega = ti.sqrt(ti.abs(omega))
            # Clamp sqrt_omega to avoid extreme values
            sqrt_omega = ti.max(0.1, ti.min(10.0, sqrt_omega))
            factor = 1.0 - 1.0 / sqrt_omega
            
            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    d_omega_d_Eij = self._compute_d_omega_d_E(E, i, j)
                    S_vol[i, j] = p * d_omega_d_Eij * factor
        
        return S_vol

    @ti.func
    def _compute_volume_control_stress(self, E, k):
        """Compute volume control stress (Eq. 2.106)"""
        p = self.bulk_modulus[k]
        r = self.target_volume_ratio[k]
        omega = self._compute_omega(E)
        
        S_vol = ti.Matrix.zero(ti.f64, 3, 3)
        
        if ti.abs(omega) > EPS_ZERO:
            sqrt_omega = ti.sqrt(ti.abs(omega))
            # Clamp sqrt_omega to avoid extreme values
            sqrt_omega = ti.max(0.1, ti.min(10.0, sqrt_omega))
            factor = 1.0 - r / sqrt_omega
            
            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    d_omega_d_Eij = self._compute_d_omega_d_E(E, i, j)
                    S_vol[i, j] = p * d_omega_d_Eij * factor
        
        return S_vol

    @ti.func
    def _compute_transformation_matrix(self, k):
        """Compute transformation matrix B from local to global coordinates"""
        B = ti.Matrix.zero(ti.f64, 3, 3)
        for i in ti.static(range(3)):
            axis_normalized = self.anisotropy_axes[k, i].normalized()
            B[i, 0] = axis_normalized[0]
            B[i, 1] = axis_normalized[1]
            B[i, 2] = axis_normalized[2]
        return B.transpose()

    @ti.func
    def _compute_volume_ratio(self, k):
        """Compute R_v = V_tet / V_hex (Eq. 2.58)"""
        V_tet = self.initial_volume[k]
        
        # Virtual hexahedron volume = |ζ1 × ζ2 · ζ3|
        zeta1 = self.initial_axis_vectors[k, 0] * self.rest_lengths[k, 0]
        zeta2 = self.initial_axis_vectors[k, 1] * self.rest_lengths[k, 1]
        zeta3 = self.initial_axis_vectors[k, 2] * self.rest_lengths[k, 2]
        
        V_hex = ti.abs(zeta1.cross(zeta2).dot(zeta3))
        ratio = 1.0
        if V_hex > EPS_ZERO:
            ratio = V_tet / V_hex
            # Clamp to reasonable range
            ratio = ti.max(0.1, ti.min(10.0, ratio))
        
        return ratio
        

    @ti.func
    def _get_current_axis_direction(self, k, axis_idx):
        """Get current axis direction after deformation"""
        pt1_idx = axis_idx * 2
        pt2_idx = axis_idx * 2 + 1
        
        # Default to initial axis direction
        result = self.anisotropy_axes[k, axis_idx].normalized()
        
        # Try to compute from intersection points if valid
        if (self.intersection_valid[k, pt1_idx] == 1 and 
            self.intersection_valid[k, pt2_idx] == 1):
            p1 = self.intersection_points[k, pt1_idx]
            p2 = self.intersection_points[k, pt2_idx]
            axis_vector = p2 - p1
            if axis_vector.norm() > EPS_ZERO:
                result = axis_vector.normalized()
        
        return result

    @ti.func
    def _compute_intersection_force(self, k, pt_idx, F, S, R_v):
        """Compute force on intersection point (Eq. 2.74)"""
        axis_idx = pt_idx // 2  # This is 'l' in the paper
        is_positive = (pt_idx % 2) == 0
        
        # Calculate A_l^0 using Eq. 2.57
        A_0 = 0.0
        if axis_idx == 0:
            zeta_m = self.initial_axis_vectors[k, 1] * self.rest_lengths[k, 1]
            zeta_n = self.initial_axis_vectors[k, 2] * self.rest_lengths[k, 2]
            A_0 = zeta_m.cross(zeta_n).norm()
        elif axis_idx == 1:
            zeta_m = self.initial_axis_vectors[k, 0] * self.rest_lengths[k, 0]
            zeta_n = self.initial_axis_vectors[k, 2] * self.rest_lengths[k, 2]
            A_0 = zeta_m.cross(zeta_n).norm()
        else:  # axis_idx == 2
            zeta_m = self.initial_axis_vectors[k, 0] * self.rest_lengths[k, 0]
            zeta_n = self.initial_axis_vectors[k, 1] * self.rest_lengths[k, 1]
            A_0 = zeta_m.cross(zeta_n).norm()
        
        # Force according to Eq. 2.74: f_2l = R_v * A_l^0 * (S_1λ * ζ_1 + S_2λ * ζ_2 + S_3λ * ζ_3)
        force = ti.Vector.zero(ti.f64, 3)
        
        # λ = l + 1 (but we need to handle modulo 3 for stress tensor indices)
        lambda_idx = (axis_idx + 1) % 3
        
        # Sum contributions: S_1λ * ζ_1 + S_2λ * ζ_2 + S_3λ * ζ_3
        for i in ti.static(range(3)):
            axis_current = self._get_current_axis_direction(k, i)
            # Get S_iλ component (S is symmetric)
            S_component = S[i, lambda_idx]
            force += S_component * axis_current
        
        # Apply scaling factors
        force = R_v * A_0 * force
        
        # For second intersection point (odd indices), force is negative
        if not is_positive:
            force = -force
        
        return force

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
        """Compute internal forces including volume preservation/control"""
        # Clear forces
        for i in range(self.N):
            self.force[i] = ti.Vector([0.0, 0.0, 0.0])
        
        # Compute forces for each tetrahedron
        for k in range(self.M):
            if self.vol[k] > 1e-12:
                # Calculate deformation tensor F
                F = self._compute_deformation_tensor(k)
                
                # Calculate Green strain tensor E
                E = self._compute_green_strain(F)
                
                # Calculate material stress tensor S
                S_material = self._compute_stress_tensor_lung(E, k)
                
                # Add volume preservation or control stress
                S_total = S_material
                if self.volume_control_enabled[k] == 1:
                    S_vol = self._compute_volume_control_stress(E, k)
                    S_total = S_material + S_vol
                elif self.volume_preservation_enabled[k] == 1:
                    S_vol = self._compute_volume_preservation_stress(E, k)
                    S_total = S_material + S_vol
                    
                    # Additional direct volume constraint for stability
                    current_vol = self.current_volume[k]
                    initial_vol = self.initial_volume[k]
                    if initial_vol > EPS_ZERO:
                        vol_ratio = current_vol / initial_vol
                        # If volume change is too large, add corrective stress
                        if ti.abs(vol_ratio - 1.0) > 0.1:  # More than 10% change
                            correction_factor = self.bulk_modulus[k] * (1.0 - vol_ratio) * 0.1
                            I = ti.Matrix.identity(ti.f64, 3)
                            S_correction = correction_factor * I
                            S_total = S_total + S_correction
                
                # Calculate volume ratio
                R_v = self._compute_volume_ratio(k)
                
                # Apply forces to intersection points
                for axis_idx in ti.static(range(3)):
                    pt1_idx = axis_idx * 2
                    pt2_idx = axis_idx * 2 + 1
                    
                    if (self.intersection_valid[k, pt1_idx] == 1 and 
                        self.intersection_valid[k, pt2_idx] == 1):
                        
                        # Calculate forces
                        f1 = self._compute_intersection_force(k, pt1_idx, F, S_total, R_v)
                        f2 = self._compute_intersection_force(k, pt2_idx, F, S_total, R_v)
                        
                        # Distribute to vertices using coefficient matrix
                        for vertex_idx in ti.static(range(4)):
                            c1 = self.C_k[k, vertex_idx, pt1_idx]
                            c2 = self.C_k[k, vertex_idx, pt2_idx]
                            
                            vertex_force = c1 * f1 + c2 * f2
                            
                            node_idx = self.tets[k][vertex_idx]
                            ti.atomic_add(self.force[node_idx], vertex_force)

    @ti.kernel
    def _add_gravity(self):
        """Add gravity forces"""
        gravity = ti.Vector([0.0, -9, 0.0])
        for i in range(self.N):
            self.force[i] += self.mass[i] * gravity

    @ti.kernel
    def _update_dynamics(self, dt: float):
        """Update velocities and positions with stability controls"""
        max_velocity = 0.1      # Reduce from 1.0 to 0.1
        max_acceleration = 100.0  # Reduce from 1000.0 to 100.0
        max_displacement = 0.0001  # Reduce from 0.001 to 0.0001
        
        for i in range(self.N):
            if self.mass[i] > EPS_ZERO:
                # Calculate acceleration
                acceleration = self.force[i] / self.mass[i]
                
                # Clamp acceleration
                acc_mag = acceleration.norm()
                if acc_mag > max_acceleration:
                    acceleration = acceleration * (max_acceleration / acc_mag)
                
                # Update velocity
                self.vel[i] += acceleration * dt
                
                # Clamp velocity
                vel_mag = self.vel[i].norm()
                if vel_mag > max_velocity:
                    self.vel[i] = self.vel[i] * (max_velocity / vel_mag)
                
                # Apply damping
                self.vel[i] *= self.damping
                
                # Calculate displacement
                displacement = self.vel[i] * dt
                disp_mag = displacement.norm()
                if disp_mag > max_displacement:
                    displacement = displacement * (max_displacement / disp_mag)
                
                # Update position
                self.x[i] += displacement

    @ti.kernel
    def _store_tensors(self):
        """Store computed tensors for analysis"""
        for k in range(self.M):
            if self.vol[k] > 1e-12:
                F = self._compute_deformation_tensor(k)
                self.deformation_tensor[k] = F
                
                E = self._compute_green_strain(F)
                self.strain_tensor[k] = E
                
                S = self._compute_stress_tensor_lung(E, k)
                self.stress_tensor[k] = S

    def compute_forces(self):
        """Compute all forces"""
        self._update_intersections_coefficients()
        self._store_tensors()
        self._update_current_volumes()
        self._compute_internal_forces()
        self._add_gravity()

    def time_step(self, dt):
        """Perform one time step"""
        self.compute_forces()
        self._update_dynamics(dt)

    def enable_volume_preservation(self, bulk_modulus=1e4):
        """Enable volume preservation for all tetrahedra"""
        @ti.kernel
        def set_volume_preservation(p: float):
            for k in range(self.M):
                self.volume_preservation_enabled[k] = 1
                self.volume_control_enabled[k] = 0
                self.bulk_modulus[k] = p
        
        set_volume_preservation(bulk_modulus)

    def enable_volume_control(self, target_ratio=1.0, bulk_modulus=1e4):
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

    def get_tensor_stats(self):
        """Get tensor statistics"""
        F_data = self.deformation_tensor.to_numpy()
        E_data = self.strain_tensor.to_numpy()
        S_data = self.stress_tensor.to_numpy()
        
        F_det = np.linalg.det(F_data)
        E_trace = np.trace(E_data, axis1=1, axis2=2)
        S_frobenius = np.linalg.norm(S_data, axis=(1, 2))
        
        return {
            'deformation_det_mean': np.mean(F_det),
            'deformation_det_std': np.std(F_det),
            'strain_trace_mean': np.mean(E_trace),
            'strain_trace_std': np.std(E_trace),
            'stress_norm_mean': np.mean(S_frobenius),
            'stress_norm_std': np.std(S_frobenius)
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
        
        self._store_tensors()
        tensor_stats = self.get_tensor_stats()
        print(f"\nContinuum mechanics tensor statistics:")
        print(f"  Deformation tensor det: {tensor_stats['deformation_det_mean']:.3f} ± {tensor_stats['deformation_det_std']:.3f}")
        print(f"  Strain tensor trace: {tensor_stats['strain_trace_mean']:.6f} ± {tensor_stats['strain_trace_std']:.6f}")
        print(f"  Stress tensor norm: {tensor_stats['stress_norm_mean']:.1f} ± {tensor_stats['stress_norm_std']:.1f}")


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


def main():
    """Main function"""
    mesh_path = "/home/haozhe/lung-project/HeterogeneousSegmentGNN/LungSimulation/mesh_files/case1_T00_lung_regions_11.xdmf"
    
    # Load mesh
    pts, tets, lbls = load_mesh_from_file(mesh_path)
    
    # Create simulation
    sim = Adamss(pts, tets, lbls)
    
    # Enable volume preservation
    sim.enable_volume_preservation(bulk_modulus=1e6)  # Increase from 1e4 to 1e6
    
    sim.summary()
    
    # Run simulation
    print("\nRunning simulation with volume preservation...")
    dt = 0.0001  # Reduce from 0.0005 to 0.0001 for better stability
    num_steps = 50
    
    for step in range(num_steps):
        sim.time_step(dt)
        
        # Get statistics
        total_force = sim.force.to_numpy()
        mean_force = np.mean(np.linalg.norm(total_force, axis=1))
        total_velocity = sim.vel.to_numpy()
        mean_velocity = np.mean(np.linalg.norm(total_velocity, axis=1))
        
        tensor_stats = sim.get_tensor_stats()
        
        # Check displacement
        total_positions = sim.x.to_numpy()
        initial_positions = sim.initial_positions.to_numpy()
        max_displacement = np.max(np.linalg.norm(total_positions - initial_positions, axis=1))
        
        # Volume change statistics
        vol_stats = sim.get_volume_change_stats()
        
        print(f"Step {step}: Force={mean_force:.6f}N, Vel={mean_velocity:.6f}m/s, "
              f"MaxDisp={max_displacement:.6f}m, VolChange={vol_stats['mean_relative_change']:.4%}")
        
        # Stability check - more aggressive termination
        if (mean_force > 100.0 or mean_velocity > 1.0 or max_displacement > 0.01 or 
            abs(tensor_stats['deformation_det_mean']) > 10.0 or 
            abs(tensor_stats['strain_trace_mean']) > 0.5 or
            abs(vol_stats['mean_relative_change']) > 0.5):  # More than 50% volume change
            print(f"*** Simulation became unstable at step {step}. Terminating. ***")
            break

    print("\nSimulation completed.")
    
    # Final statistics
    final_vol_stats = sim.get_volume_change_stats()
    print(f"\nFinal volume preservation statistics:")
    print(f"  Mean relative volume change: {final_vol_stats['mean_relative_change']:.4%}")
    print(f"  Max relative volume change: {final_vol_stats['max_relative_change']:.4%}")
    print(f"  Total volume change: {final_vol_stats['total_volume_change']:.6f} m³")


if __name__ == "__main__":
    main()