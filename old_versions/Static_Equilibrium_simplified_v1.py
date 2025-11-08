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

simplified version of Static_Equilibrium.py
All the tetra-related data is pre-computed in deformation_processor.py.
This script only implements the static equilibrium solver for the mass-spring system 
to reduce compiling time and cuda kernel size.
"""

import taichi as ti
import numpy as np
import time
import torch

ti.init(arch=ti.cuda, device_memory_fraction=0.9, debug=True, kernel_profiler = True)

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
        self.x = ti.Vector.field(3, ti.f64, shape=self.N)      # current positions
        self.tets = ti.Vector.field(4, ti.i32, shape=self.M)   # tetrahedron indices
        self.vol = ti.field(ti.f64, shape=self.M)              # volumes
        self.mass = ti.field(ti.f64, shape=self.N)             
        
        #-------------------------------------- Spring System Fields --------------------------------------
        self.intersection_points = ti.Vector.field(3, ti.f64, shape=(self.M, 6))  # 6 points per tet
        self.intersection_valid = ti.field(ti.i32, shape=(self.M, 6))             # validity indicator
        self.intersection_face = ti.field(ti.i32, shape=(self.M, 6))              # intersected face index
        self.C_k = ti.field(ti.f64, shape=(self.M, 4, 6))                        # Coefficient matrix
        
        ## Spring parameters
        self.rest_lengths = ti.field(ti.f64, shape=(self.M, 3))         # rest lengths
        self.axial_stiffness = ti.field(ti.f64, shape=(self.M, 3))            # axial stiffness
        self.torsion_stiffness = ti.field(ti.f64, shape=(self.M, 3))    # torsion stiffness
        self.volume_kappa = ti.field(ti.f64, shape=self.M)              # bulk modulus per tet
        self.rest_cos_angles = ti.field(ti.f64, shape=(self.M, 3))      # rest angles
        
        
        ## Boundary conditions
        self.boundary_nodes = ti.field(ti.i32, shape=self.N)                      # boundary indicator
        self.boundary_displacement = ti.Vector.field(3, ti.f64, shape=self.N)     # prescribed displacements
        self.is_boundary_constrained = ti.field(ti.i32, shape=self.N)             # constraint indicator
        
        #------------------------------ Static equilibrium solver fields ----------------------------------------
        self.residual = ti.Vector.field(3, ti.f64, shape=self.N)              # Residual vector
        self.solution_increment = ti.Vector.field(3, ti.f64, shape=self.N)    # Solution increment
        self.temp_b = ti.field(ti.f64, shape=3*self.N)                        # Flattened RHS
        self.temp_x = ti.field(ti.f64, shape=3*self.N)                        # Flattened solution
        
        self.initial_positions = ti.Vector.field(3, ti.f64, shape=self.N)         # Initial positions
        self.displacement_field = ti.Vector.field(3, ti.f64, shape=self.N)


        # Load all preprocessed data into Taichi fields
        self._load_preprocessed_data(data)
        self._initialize_solver_fields()

    def _load_preprocessed_data(self, data):
        """Load all preprocessed data from deformation_processor.py into Taichi fields"""
        # Basic mesh data - keep original (mm)
        self.x.from_numpy(data['mesh_points'].astype(np.float64))
        self.tets.from_numpy(data['tetrahedra'].astype(np.int32))
        self.vol.from_numpy(data['volume'].astype(np.float64))
        self.mass.from_numpy(data['mass'].astype(np.float64))
        
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
    def _build_sparse_matrix(self, K: ti.types.sparse_matrix_builder()):
        """
        In an element-based assembly you always go from a local stiffness matrix of size (number of local DOFs)^2
        to global blocks of size (DOFs per node)^2. 

        Ke is 12*12: Each tetrahedron has 4 nodes. Each node in 3D has 3 translational degrees of freedom (x, y, z).
        Total local DOFs per tet = 4 nodes * 3 DOF/node = 12, so its tangent stiffness matrix Ke is 12*12.
        When we generate Ke into the global stiffness K, it's node-pair by node-pair.
        
        For a given pair of nodes i and j (each with 3 DOFs), their mutual coupling is a 3*3 submatrix of Ke.
        we map that 3*3 block into the global matrix at rows (3*node_i + 0..2) and columns (3*node_j + 0..2).

        So the assembly loop over i,j and over d1,d2 (0…2) is simply 
        extract each 3*3 block from Ke and add it to the big K at the proper global indices.

        The boundary condition is :
        [K_FF, 0] [delta q_F]  = [b_F]
        [   0, I] [delta q_D]    [0]
        """
        for k in range(self.M):
            if self.vol[k] > 1e-10:
                # --- Axial springs  ---
                for axis_idx in ti.static(range(3)):
                    pt1_idx = axis_idx * 2
                    pt2_idx = axis_idx * 2 + 1
                    if (self.intersection_valid[k, pt1_idx] == 1 and
                        self.intersection_valid[k, pt2_idx] == 1):

                        kspring = self.axial_stiffness[k, axis_idx]
                        p1 = self.intersection_points[k, pt1_idx]
                        p2 = self.intersection_points[k, pt2_idx]
                        v = p2 - p1
                        L = v.norm()
                        if L > 1e-10:
                            n = v / L
                            # s_u = c(p1) - c(p2)
                            s = ti.Vector([0.0, 0.0, 0.0, 0.0])
                            for i in ti.static(range(4)):
                                s[i] = self.C_k[k, i, pt1_idx] - self.C_k[k, i, pt2_idx]

                            for i in ti.static(range(4)):
                                ni = self.tets[k][i]
                                if self.is_boundary_constrained[ni] == 0:  # assembly only free rows
                                    for j in ti.static(range(4)):
                                        nj = self.tets[k][j]
                                        if self.is_boundary_constrained[nj] == 0:  # assembly only free columns
                                            coeff = kspring * s[i] * s[j]
                                            if ti.abs(coeff) > 1e-12:
                                                for d1 in ti.static(range(3)):
                                                    row = 3 * ni + d1
                                                    for d2 in ti.static(range(3)):
                                                        col = 3 * nj + d2
                                                        K[row, col] += coeff * n[d1] * n[d2]

                # --- Torsion-springs ---
                w = self.torsion_stiffness[k, 0]
                for i in ti.static(range(4)):
                    ni = self.tets[k][i]
                    if self.is_boundary_constrained[ni] == 0:
                        deg_free = 0
                        for j in ti.static(range(4)):
                            if i != j:
                                nj = self.tets[k][j]
                                if self.is_boundary_constrained[nj] == 0:
                                    deg_free += 1
                                    for d in ti.static(range(3)):
                                        row = 3 * ni + d
                                        col = 3 * nj + d
                                        K[row, col] += -w
                        for d in ti.static(range(3)):
                            row = 3 * ni + d
                            K[row, row] += w * deg_free
                
                # --- Volume preservation ---
                kappa = self.volume_kappa[k]
                i0, i1, i2, i3 = self.tets[k][0], self.tets[k][1], self.tets[k][2], self.tets[k][3]
                x0, x1, x2, x3 = self.x[i0], self.x[i1], self.x[i2], self.x[i3]
                V0 = self.vol[k]
                
                # keep the same volume formula as in residual
                g0 = (x1 - x2).cross(x3 - x2) / 6.0; g1 = (x2 - x0).cross(x3 - x0) / 6.0
                g2 = (x0 - x1).cross(x3 - x1) / 6.0; g3 = (x0 - x2).cross(x1 - x2) / 6.0

                coeff = kappa / V0 if V0 > 1e-10 else 0.0
                nodes = ti.Vector([i0, i1, i2, i3])
                gs = [g0, g1, g2, g3]

                for a in ti.static(range(4)):
                    na = nodes[a]
                    if self.is_boundary_constrained[na] == 0:
                        for b in ti.static(range(4)):
                            nb = nodes[b]
                            if self.is_boundary_constrained[nb] == 0:
                                for d1 in ti.static(range(3)):
                                    row = 3 * na + d1
                                    for d2 in ti.static(range(3)):
                                        col = 3 * nb + d2
                                        K[row, col] += coeff * gs[a][d1] * gs[b][d2]


        # --- Constrained DOFs -> identity diagonal (Δq_D = 0 ) ---
        for i in range(self.N):
            if self.is_boundary_constrained[i] == 1:
                for d in ti.static(range(3)):
                    dof = 3 * i + d
                    K[dof, dof] += 1.0


    

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


    def _solve_linear_system_with_taichi(self):
        """Solve K * delta x = -r using Taichi sparse solver"""
        try:
            print("Linear system timing breakdown:")
            total_start = time.time()
            
            # Convert dense matrix to sparse format
            sparse_start = time.time()
            # Estimate triplets: ~60 3x3 blocks per tet (axial + torsion) = ~540 triplets per tet
            max_triplets = 1000 * self.M + 9 * self.N  
            K_sparse = ti.linalg.SparseMatrixBuilder(3*self.N, 3*self.N, max_num_triplets=max_triplets)
            self._build_sparse_matrix(K_sparse)
            A = K_sparse.build()
            sparse_time = time.time() - sparse_start
            print(f"  Sparse matrix building: {sparse_time:.4f} seconds")

            # Flatten residual vector
            flatten_start = time.time()
            self._flatten_residual_to_vector(self.temp_b)
            self.temp_x.fill(0.0)
            flatten_time = time.time() - flatten_start
            print(f"  Vector preparation: {flatten_time:.4f} seconds")
            
            # Create sparse solver
            setup_start = time.time()
            solver = ti.linalg.SparseSolver(solver_type="LDLT")
            solver.analyze_pattern(A)
            solver.factorize(A)
            setup_time = time.time() - setup_start
            print(f"  Solver setup (analyze + factorize): {setup_time:.4f} seconds")
            
            # Solve the system  
            solve_start = time.time()
            solver.solve(self.temp_b, self.temp_x)
            solve_time = time.time() - solve_start
            print(f"  Linear system solve: {solve_time:.4f} seconds")
            

            total_time = time.time() - total_start
            print(f"  Total linear solver time: {total_time:.4f} seconds")
            print(f"  System size: {3*self.N} x {3*self.N} ({3*self.N} DOF)")
            
            return True
            
        except Exception as e:
            print(f"Taichi sparse solver error: {e}")
            return False
        
    def solve_static_equilibrium(self):
        """
        Solve static equilibrium: K * delta q = -r
        Single linear solve for small deformations
        """
        print(f"=== Static Equilibrium Solver ===")
        
        # Compute residual forces
        self._compute_residual()
        
        # Check initial residual
        residual_norm = self._compute_residual_norm()
        print(f"Initial residual norm = {residual_norm:.6e}")
        
        # Solve linear system with boundary constraints applied during sparse matrix assembly
        success = self._solve_linear_system_with_taichi()
        
        if success:
            print("Linear system solved successfully")
            return True
        else:
            print("Linear solver failed")
            return False
        
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
    preprocessed_data_path = "data_processed_deformation/Case4Pack/Case4Pack_T00_to_T10_deformation.npz"
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
    print("\n=== Solving Static Equilibrium ===")
    success = sim.solve_static_equilibrium()
    

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
    