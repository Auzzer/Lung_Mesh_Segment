"""
Static Equilibrium (Simplified, SMS-rows-based)

This file only assembles the linear system using the precomputed SMS
measurement rows and solves for different (alpha, beta, kappa). It stops
computing spring internal forces and does not use the trick of adding an
identity on constrained DOFs; instead it directly constructs the free-DOF
compressed system K_FF · Δq_F = b_F with external loads on free DOFs.

Notes on Taichi usage:
- ti.kernel: executed from Python scope (launches kernels)
- ti.func: callable only within other Taichi kernels/functions

All required per-tetrahedron quantities are precomputed in
deformation_processor_v2.py and loaded here.
"""

import taichi as ti
import numpy as np
import time
import torch

ti.init(arch=ti.cuda, device_memory_fraction=0.9, debug=False, kernel_profiler = False)

# --------------------------------------------
# CONSTANTS
# --------------------------------------------
DEFAULT_RHO = {
    0: 1050.0,   # normal tissue (kg/m³)
    1:   250.0,   # air-rich
    2: 1100.0,   # fibrotic
}

# Tolerance constants
EPS_PARALLEL = 1e-10
EPS_BARYCENTRIC = 1e-8
EPS_DISTANCE = 1e-6
EPS_ZERO = 1e-10


# --------------------------------------------
@ti.data_oriented
class Sim:
    def __init__(self, preprocessed_data_path):
        """Initialize solver with preprocessed data from deformation_processor.py"""
        # Load preprocessed data
        data = np.load(preprocessed_data_path, allow_pickle=True)
        
        self.N = data['mesh_points'].shape[0]  # #nodes
        self.M = data['tetrahedra'].shape[0]   # #tets

        # -------------------------------Core Taichi  data fields ------------------------------------
        self.x = ti.Vector.field(3, ti.f32, shape=self.N)      # current positions
        self.tets = ti.Vector.field(4, ti.i32, shape=self.M)   # tetrahedron indices
        self.vol = ti.field(ti.f32, shape=self.M)              # volumes
        self.mass = ti.field(ti.f32, shape=self.N)             
        
        #-------------------------------------- Spring System Fields --------------------------------------
        self.intersection_points = ti.Vector.field(3, ti.f32, shape=(self.M, 6))  # 6 points per tet
        self.intersection_valid = ti.field(ti.i32, shape=(self.M, 6))             # validity indicator
        self.intersection_face = ti.field(ti.i32, shape=(self.M, 6))              # intersected face index
        self.C_k = ti.field(ti.f32, shape=(self.M, 4, 6))                        # Coefficient matrix
        
        ## Spring parameters
        self.rest_lengths = ti.field(ti.f32, shape=(self.M, 3))         # rest lengths
        self.axial_stiffness = ti.field(ti.f32, shape=(self.M, 3))            # axial stiffness
        self.torsion_stiffness = ti.field(ti.f32, shape=(self.M, 3))    # torsion stiffness
        self.volume_kappa = ti.field(ti.f32, shape=self.M)              # bulk modulus per tet
        self.rest_cos_angles = ti.field(ti.f32, shape=(self.M, 3))      # rest angles
        
        
        ## Boundary conditions
        self.boundary_nodes = ti.field(ti.i32, shape=self.N)                      # boundary indicator
        self.boundary_displacement = ti.Vector.field(3, ti.f32, shape=self.N)     # prescribed displacements
        self.is_boundary_constrained = ti.field(ti.i32, shape=self.N)             # constraint indicator
        # SMS measurement rows from preprocessing (reference geometry)
        self.r_axis  = ti.Vector.field(12, ti.f32, shape=(self.M, 3))
        self.r_shear = ti.Vector.field(12, ti.f32, shape=(self.M, 3))
        self.r_vol   = ti.Vector.field(12, ti.f32, shape=self.M)
        #------------------------------ Static equilibrium solver fields ----------------------------------------
        self.residual = ti.Vector.field(3, ti.f32, shape=self.N)              # Residual vector
        self.solution_increment = ti.Vector.field(3, ti.f32, shape=self.N)    # Solution increment
        self.temp_b = ti.field(ti.f32, shape=3*self.N)                        # Flattened RHS
        self.temp_x = ti.field(ti.f32, shape=3*self.N)                        # Flattened solution
        # Current element index for helper funcs during assembly
        self._cur_k = ti.field(dtype=ti.i32, shape=())
        
        self.initial_positions = ti.Vector.field(3, ti.f32, shape=self.N)         # Initial positions
        self.displacement_field = ti.Vector.field(3, ti.f32, shape=self.N)


        # Load all preprocessed data into Taichi fields
        self._load_preprocessed_data(data)
        self._initialize_solver_fields()

    def _load_preprocessed_data(self, data):
        """Load all preprocessed data from deformation_processor.py into Taichi fields"""
        # Basic mesh data - keep original (mm)
        self.x.from_numpy(data['mesh_points'].astype(np.float64))
        self.tets.from_numpy(data['tetrahedra'].astype(np.int32))
        self.vol.from_numpy(data['volume'].astype(np.float64))
        self.mass.from_numpy(data['mass'].astype(np.float64)*1e-9)
        
        # Spring system data
        self.intersection_points.from_numpy(data['intersection_points'].astype(np.float64))
        self.intersection_valid.from_numpy(data['intersection_valid'].astype(np.int32))
        self.C_k.from_numpy(data['coefficient_matrix'].astype(np.float64))
        
        # Spring parameters
        self.rest_lengths.from_numpy(data['rest_lengths'].astype(np.float64))
        self.volume_kappa.from_numpy(data['volume_kappa'].astype(np.float64))
        self.axial_stiffness.from_numpy(data['stiffness'].astype(np.float64))
        self.torsion_stiffness.from_numpy(data['torsion_stiffness'].astype(np.float64))
        self.rest_cos_angles.from_numpy(data['rest_cos_angles'].astype(np.float64))
        
        self.r_axis.from_numpy(data['r_axis'].astype(np.float64))
        self.r_shear.from_numpy(data['r_shear'].astype(np.float64))
        self.r_vol.from_numpy(data['r_vol'].astype(np.float64))

        # Boundary conditions
        self.boundary_nodes.from_numpy(data['boundary_nodes'].astype(np.int32))
        self.initial_positions.from_numpy(data['initial_positions'].astype(np.float64))#keep original (mm)
        self.displacement_field.from_numpy(data['displacement_field'].astype(np.float64))# keep original (mm)
        
        print(f"Loaded preprocessed data: {self.N} nodes, {self.M} tetrahedra")
        print(f"Boundary nodes: {np.sum(data['boundary_nodes'])}")
        print(f"Valid intersections: {np.sum(data['intersection_valid'])}/{data['intersection_valid'].size}")

    def _count_constrained_nodes(self):
        """Count number of nodes with Dirichlet constraints applied"""
        return int(self.is_boundary_constrained.to_numpy().sum())

    def _compute_residual_norm(self):
        """Compute L2 norm of residual vector for free nodes"""
        residual_np = self.residual.to_numpy()
        constrained_np = self.is_boundary_constrained.to_numpy()
        
        # Only compute norm for free (unconstrained) nodes
        free_residual = residual_np[constrained_np == 0]
        return np.linalg.norm(free_residual.flatten())
    
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

    # ---------------------- residual assembly (each tetra iteration) ----------------------
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
                    k_spring = self.axial_stiffness[k, axis_idx]
                    
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
        """Add torsion spring force contributions to residual"""
        # approximation with -beta * (x_j - x_i)
        beta = self.torsion_stiffness[k, 0]
        for i in ti.static(range(4)):
            node_i = self.tets[k][i]
            force_sum = ti.Vector([0.0, 0.0, 0.0])
            
            for j in ti.static(range(4)):
                if i != j:
                    node_j = self.tets[k][j]
                     
                    pos_diff = self.x[node_j] - self.x[node_i]
                    force_sum += -beta * pos_diff

            ti.atomic_add(self.residual[node_i], force_sum)
    
    @ti.func
    def _add_residual_from_volume_kappa(self, k):
        """Add volume reservation force contributions to residual"""
        i0, i1, i2, i3 = self.tets[k][0], self.tets[k][1], self.tets[k][2], self.tets[k][3]
        x0, x1, x2, x3 = self.x[i0], self.x[i1], self.x[i2], self.x[i3]

        V =  ((x1 - x0).cross(x2 - x0)).dot(x3 - x0)/6  # current volume
        V0 = self.vol[k]  # rest volume
        kappa = self.volume_kappa[k]  # bulk modulus

        if V0 > EPS_ZERO and kappa > 0:
            g0 = (x1 - x2).cross(x3 - x2) / 6.0
            g1 = (x2 - x0).cross(x3 - x0) / 6.0
            g2 = (x0 - x1).cross(x3 - x1) / 6.0
            g3 = (x0 - x2).cross(x1 - x2) / 6.0
            volume_strain = (V - V0) / V0
            pressure = kappa * volume_strain

            if self.is_boundary_constrained[i0] == 0: ti.atomic_add(self.residual[i0], -pressure * g0)
            if self.is_boundary_constrained[i1] == 0: ti.atomic_add(self.residual[i1], -pressure * g1)
            if self.is_boundary_constrained[i2] == 0: ti.atomic_add(self.residual[i2], -pressure * g2)
            if self.is_boundary_constrained[i3] == 0: ti.atomic_add(self.residual[i3], -pressure * g3)

    @ti.kernel
    def _compute_residual(self):
        """Compute residual vector r = g_int + F_ext"""
        # Initialize residual to zero
        for i in range(self.N):
            self.residual[i] = ti.Vector([0.0, 0.0, 0.0])
        
        # Add gravity forces (external forces)
        # mass is in kg, positions in mm → gravity acceleration is 9810 mm/s²
        gravity = ti.Vector([0.0, -9810.0, 0.0])  # mm/s² (consistent with mm units)
        for i in range(self.N):
            self.residual[i] += self.mass[i] * gravity
        
        # Compute internal forces 
        for k in range(self.M):
            if self.vol[k] > 1e-10:
                self._add_residual_from_axial_springs(k)
                self._add_residual_from_torsion_springs(k)
                self._add_residual_from_volume_kappa(k)

    @ti.kernel
    def _initialize_solver_fields(self):
        """Initialize solver-specific fields"""
        for i in range(self.N):
            self.residual[i] = ti.Vector([0.0, 0.0, 0.0])
            self.solution_increment[i] = ti.Vector([0.0, 0.0, 0.0])
            self.boundary_displacement[i] = ti.Vector([0.0, 0.0, 0.0])
            self.is_boundary_constrained[i] = 0
        
        for i in range(3 * self.N):
            self.temp_b[i] = 0.0
            self.temp_x[i] = 0.0
    
    def apply_dirichlet_bc_on_boundary(self):
        """
        Apply Dirichlet boundary conditions using preprocessed boundary nodes 
        and displacement field
        """
        self._apply_boundary_displacements()
        constrained_count = self._count_constrained_nodes()
        print(f"Applied Dirichlet BC to {constrained_count} boundary nodes")
        # Build free-DOF map after BCs are applied
        self._build_dof_map()

    @ti.kernel
    def _apply_boundary_displacements(self):
        """Apply displacement boundary conditions to detected boundary nodes"""
        for i in range(self.N):
            if self.boundary_nodes[i] == 1:  # This is a boundary node
                # Set prescribed displacement from preprocessed 4DCT data
                self.boundary_displacement[i] = self.displacement_field[i]
                # Mark as constrained
                self.is_boundary_constrained[i] = 1
                # Update position to satisfy Dirichlet condition: q = X + u
                self.x[i] = self.initial_positions[i] + self.displacement_field[i]



    

    @ti.kernel
    def _flatten_residual_to_vector(self, b: ti.template()):
        """Flatten residual vector to 1D for solver
        free DOFs are set to negative residual, limited DOFs are set to zero.
        """
        for i in range(self.N):
            if self.is_boundary_constrained[i] == 1:        
                b[3*i + 0] = 0
                b[3*i + 1] = 0
                b[3*i + 2] = 0
            else:
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
        """Update nodal positions: x = x + delta x (for free nodes only)"""
        for i in range(self.N):
            if self.is_boundary_constrained[i] == 0:  # Free node
                self.x[i] += self.solution_increment[i]
            # Constrained nodes remain at their prescribed positions

    # ----------free-DOF compressed assembly and solver ----------
    @ti.kernel
    def _build_rhs_from_bc_free(self, b: ti.template(), alpha: ti.f32, beta: ti.f32, kappa: ti.f32):
        """
        Computes the RHS contribution from boundary conditions: -K_FD * Δq_D
        This is done by iterating through all elements and computing the force
        contribution from displaced boundary nodes onto free nodes.
        """
        for k in range(self.M):
            if self.vol[k] > 1e-12:
                V = self.vol[k]
                
                # Collect displacements for the 4 nodes of the current tet
                disp_tet = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                for i in ti.static(range(4)):
                    node_idx = self.tets[k][i]
                    # We only care about the prescribed displacement of boundary nodes
                    if self.is_boundary_constrained[node_idx] == 1:
                        disp = self.boundary_displacement[node_idx]
                        disp_tet[3*i+0] = disp[0]
                        disp_tet[3*i+1] = disp[1]
                        disp_tet[3*i+2] = disp[2]

                # Calculate force contribution from this element's BCs
                # f_k = -K_k * Δq_k
                
                # 1. Axial contribution
                coeff_alpha = 4.0 * alpha * V
                for ell in ti.static(range(3)):
                    r_ax = self.r_axis[k, ell]
                    force_on_free_nodes = -coeff_alpha * r_ax.dot(disp_tet)
                    # Add this force to the free DOFs connected to this element
                    for i in ti.static(range(4)):
                        node_idx = self.tets[k][i]
                        if self.is_boundary_constrained[node_idx] == 0: # This is a free node
                            for d in ti.static(range(3)):
                                row = self.dof_map[3*node_idx + d]
                                if row != -1:
                                    ti.atomic_add(b[row], force_on_free_nodes * r_ax[3*i+d])

                # 2. Shear contribution
                coeff_beta = 4.0 * beta * V
                for s in ti.static(range(3)):
                    r_sh = self.r_shear[k, s]
                    force_on_free_nodes = -coeff_beta * r_sh.dot(disp_tet)
                    for i in ti.static(range(4)):
                        node_idx = self.tets[k][i]
                        if self.is_boundary_constrained[node_idx] == 0:
                            for d in ti.static(range(3)):
                                row = self.dof_map[3*node_idx + d]
                                if row != -1:
                                    ti.atomic_add(b[row], force_on_free_nodes * r_sh[3*i+d])

                # 3. Volume contribution
                coeff_kappa = kappa * V
                r_v = self.r_vol[k]
                force_on_free_nodes = -coeff_kappa * r_v.dot(disp_tet)
                for i in ti.static(range(4)):
                    node_idx = self.tets[k][i]
                    if self.is_boundary_constrained[node_idx] == 0:
                        for d in ti.static(range(3)):
                            row = self.dof_map[3*node_idx + d]
                            if row != -1:
                                ti.atomic_add(b[row], force_on_free_nodes * r_v[3*i+d])

    @ti.kernel
    def _build_rhs_free(self, b: ti.template()):
        """Assemble external loads (e.g., gravity) on free DOFs only."""
        g = ti.Vector([0.0, -9810.0, 0.0])  # mm/s^2
        for i in range(self.N):
            if self.is_boundary_constrained[i] == 0:
                rowx = self.dof_map[3*i + 0]
                rowy = self.dof_map[3*i + 1]
                rowz = self.dof_map[3*i + 2]
                if rowx != -1:
                    # Use += to add to existing values in b
                    b[rowx] += self.mass[i] * g[0]
                    b[rowy] += self.mass[i] * g[1]
                    b[rowz] += self.mass[i] * g[2]

    # ---------------------- Optimized free-DOF stiffness assembly ----------------------
    @ti.func
    def _add_outer_free(self, K: ti.template(),
                        r: ti.template(), coeff: ti.f32):
        # r: length-12 row vector (4 nodes × 3 DOFs)
        k = self._cur_k[None]
        for i in ti.static(range(4)):
            ni = self.tets[k][i]
            for d1 in ti.static(range(3)):
                row = self.dof_map[3*ni + d1]
                vi = r[3*i + d1]
                if row != -1 and ti.abs(vi) >= 1e-20:
                    for j in ti.static(range(4)):
                        nj = self.tets[k][j]
                        for d2 in ti.static(range(3)):
                            col = self.dof_map[3*nj + d2]
                            vj = r[3*j + d2]
                            if col != -1 and ti.abs(vj) > 1e-20:
                                K[row, col] += coeff * vi * vj

    @ti.kernel
    def _assemble_K_axis_free(self, K: ti.types.sparse_matrix_builder(),
                              alpha: ti.f32):
        for k in range(self.M):
            if self.vol[k] > 1e-12:
                self._cur_k[None] = k
                V = self.vol[k]
                coeff = 4.0 * alpha * V
                for ell in ti.static(range(3)):
                    r = self.r_axis[k, ell]
                    self._add_outer_free(K, r, coeff)

    @ti.kernel
    def _assemble_K_shear_free(self, K: ti.types.sparse_matrix_builder(),
                               beta: ti.f32):
        for k in range(self.M):
            if self.vol[k] > 1e-12:
                self._cur_k[None] = k
                V = self.vol[k]
                coeff = 4.0 * beta * V
                for s in ti.static(range(3)):
                    r = self.r_shear[k, s]
                    self._add_outer_free(K, r, coeff)

    @ti.kernel
    def _assemble_K_vol_free(self, K: ti.types.sparse_matrix_builder(),
                             kappa: ti.f32):
        for k in range(self.M):
            if self.vol[k] > 1e-12:
                self._cur_k[None] = k
                V = self.vol[k]
                coeff = kappa * V
                r = self.r_vol[k]
                self._add_outer_free(K, r, coeff)


    def _build_dof_map(self):
        dof_map_np = -np.ones(3*self.N, dtype=np.int32)
        cnt = 0
        is_bc = self.is_boundary_constrained.to_numpy()
        for i in range(self.N):
            if is_bc[i] == 0:
                for d in range(3):
                    dof_map_np[3*i + d] = cnt
                    cnt += 1
        self.n_free_dof = int(cnt)
        self.dof_map = ti.field(ti.i32, shape=3*self.N)
        self.dof_map.from_numpy(dof_map_np)

    def _alloc_free_buffers(self):
        n = int(self.n_free_dof)
        self.b_free = ti.field(ti.f32, shape=n)
        self.x_free = ti.field(ti.f32, shape=n)

    @ti.kernel
    def _scatter_solution_free(self, x_free: ti.template()):
        for i in range(self.N):
            if self.is_boundary_constrained[i] == 0:
                self.solution_increment[i] = ti.Vector([
                    x_free[self.dof_map[3*i + 0]],
                    x_free[self.dof_map[3*i + 1]],
                    x_free[self.dof_map[3*i + 2]],
                ])
            else:
                self.solution_increment[i] = ti.Vector([0.0, 0.0, 0.0])

    def solve_static_equilibrium(self, alpha: float, beta: float, kappa: float):
        print("=== Static Equilibrium (SMS quadratic) ===")
        if not hasattr(self, 'dof_map'):
            self._build_dof_map()
        self._alloc_free_buffers()
        # Assemble K_FF
        n = self.b_free.shape[0]
        max_triplets = 400 * self.M + 40 * self.N  # initial estimate
        Kb = ti.linalg.SparseMatrixBuilder(n, n, max_num_triplets=max_triplets)
        ## Three-stage assembly for performance
        self._assemble_K_axis_free(Kb, float(alpha))
        self._assemble_K_shear_free(Kb, float(beta))
        self._assemble_K_vol_free(Kb, float(kappa))
        A = Kb.build()
        print("K is build")
        
        # RHS b_free = F_ext_F - K_FD * Δq_D
        self.b_free.fill(0.0)
        # First, compute the boundary condition part: -K_FD * Δq_D
        self._build_rhs_from_bc_free(self.b_free, float(alpha), float(beta), float(kappa))
        # Then, add the external forces part: F_ext_F (gravity)
        self._build_rhs_free(self.b_free)
        
        self.x_free.fill(0.0)
        # Solve
        print(f"Matrix shape: {A.n} x {A.m}")
        
        try:
            from scipy.sparse.linalg import spsolve
            from scipy.sparse import csr_matrix
            import numpy as np
            
            # Use scipy solver as primary method due to Taichi CUDA solver issues
            print("Using scipy sparse solver...")
            
            # Export matrix to file and import with scipy
            import tempfile
            import os
            with tempfile.NamedTemporaryFile(suffix='.mtx', delete=False) as tmp:
                tmp_path = tmp.name
            
            try:
                # Export matrix in MatrixMarket format
                A.mmwrite(tmp_path)
                
                # Read back with scipy
                from scipy.io import mmread
                A_scipy = mmread(tmp_path).tocsr()
                
                # Solve the system
                b_np = self.b_free.to_numpy()
                x_np = spsolve(A_scipy, b_np)
                
                self.x_free.from_numpy(x_np)
                print("Scipy solve successful!")
                
            finally:
                # Clean up temporary file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            
        except Exception as e:
            print(f"Solver failed: {e}")
            # Create zero solution as fallback
            self.x_free.fill(0.0)
            print("Using zero solution as fallback")
            return False
        # Scatter and update
        self._scatter_solution_free(self.x_free)
        self._update_positions_from_solution()
        return True
        
    def summary(self):
        """Print simulation summary"""
        total_mass = self.mass.to_numpy().sum()
        print(f"#nodes={self.N}, #tets={self.M}, total mass={total_mass:.3f} kg")
        
        stats = self.get_intersection_stats()
        print(f"Intersection success rate: {stats['success_rate']:.2%}")
        print(f"Average intersections per tet: {stats['per_tet_average']:.1f}/6")
        
        print(f"\nMass-Spring System (from preprocessed data):")
        print(f"  Axial spring stiffness: {self.axial_stiffness.to_numpy().mean():.3f} ± {self.axial_stiffness.to_numpy().std():.3f} N/m")
        print(f"  Torsion spring stiffness: {self.torsion_stiffness.to_numpy().mean():.3f} ± {self.torsion_stiffness.to_numpy().std():.3f} N⋅m/rad")
        print(f"  Rest lengths: {self.rest_lengths.to_numpy().mean():.6f} ± {self.rest_lengths.to_numpy().std():.6f} m")



######use the loaded data for the previous taichi scope simulation

def sim_main():    
    preprocessed_data_path = "data_processed_deformation/Case1Pack/Case1Pack_T00_to_T10_deformation.npz"
    print(f"=== Loading Preprocessed Data ===")
    print(f"Data file: {preprocessed_data_path}")
    sim = Sim(preprocessed_data_path)
    
    
    # Apply boundary conditions (already preprocessed)
    print("=== Applying Preprocessed Boundary Conditions ===")
    sim.apply_dirichlet_bc_on_boundary()
    
    # Print simulation summary
    sim.summary()
    
    # Print displacement field statistics
    displacement_np = sim.displacement_field.to_numpy()
    displ_magnitudes = np.linalg.norm(displacement_np, axis=1)
    print(f"\n4DCT Displacement Field Statistics:")
    print(f"  Max displacement: {displ_magnitudes.max():.6f} mm")
    print(f"  Mean displacement: {displ_magnitudes.mean():.6f} mm")
    print(f"  Nodes with displacement > 1m: {np.sum(displ_magnitudes > 1)}/{len(displ_magnitudes)}")
    
    # Solve static equilibrium and update positions
    print("\n=== Solving Static Equilibrium (SMS rows) ===")
    alpha, beta, kappa = 0.5, 0.5, 15.0  # example values, consistent with preprocessing scales
    success = sim.solve_static_equilibrium(alpha, beta, kappa)
    

    if success:
        print("Static equilibrium solved successfully!")
        
        # Print final position statistics
        final_positions = sim.x.to_numpy()
        position_changes = final_positions - sim.initial_positions.to_numpy()
        change_magnitudes = np.linalg.norm(position_changes, axis=1)
        
        print(f"\nFinal Position Update Statistics:")
        print(f"  Max position change: {change_magnitudes.max():.6f} mm")
        print(f"  Mean position change: {change_magnitudes.mean():.6f} mm")
        print(f"  RMS position change: {np.sqrt(np.mean(change_magnitudes**2)):.6f} mm")
    else:
        print("Failed to solve static equilibrium!")
    

if __name__ == "__main__":
    sim_main()
    
