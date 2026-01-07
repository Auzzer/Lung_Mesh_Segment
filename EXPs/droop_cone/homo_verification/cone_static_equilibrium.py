"""
Cone Static Equilibrium Solver for Homogenization Verification

MODIFIED FROM: Static_Equilibrium_Simplified_V2_scipy_version.py
PURPOSE: Cone-specific verification case for SMS homogenization validation

This file solves static equilibrium on the preprocessed cone data using the SMS
(Second Moment of Stiffness) approach. It's adapted from 
Static_Equilibrium_Simplified_V2_scipy_version.py specifically for cone verification.

This enables comparison between standard FEM linear elasticity and the SMS approach
to validate the homogenization methodology.

KEY MODIFICATIONS FROM ORIGINAL:
- Adapted for cone geometry instead of lung tissue
- Simplified boundary conditions (clamp top face instead of complex lung BCs)
- Saves deformed mesh files (XDMF) instead of NPZ data files
- Removed lung-specific preprocessing dependencies
- Added FEM comparison functionality
"""

import taichi as ti
import numpy as np
import time
from pathlib import Path
import meshio  # MODIFICATION: Added for mesh file output

ti.init(arch=ti.cuda, device_memory_fraction=0.9, debug=False, kernel_profiler=False)

# Tolerance constants
EPS_PARALLEL = 1e-10
EPS_BARYCENTRIC = 1e-8
EPS_DISTANCE = 1e-6
EPS_ZERO = 1e-10

@ti.data_oriented
class ConeStaticEquilibrium:
    def __init__(self, preprocessed_data_path):
        """Initialize solver with preprocessed cone data"""
        # Load preprocessed data
        data = np.load(preprocessed_data_path, allow_pickle=True)
        
        self.N = data['mesh_points'].shape[0]  # #nodes
        self.M = data['tetrahedra'].shape[0]   # #tets

        # Core Taichi data fields
        self.x = ti.Vector.field(3, ti.f32, shape=self.N)      # current positions
        self.tets = ti.Vector.field(4, ti.i32, shape=self.M)   # tetrahedron indices
        self.vol = ti.field(ti.f32, shape=self.M)              # volumes
        self.mass = ti.field(ti.f32, shape=self.N)             
        
        # Spring System Fields
        self.intersection_points = ti.Vector.field(3, ti.f32, shape=(self.M, 6))
        self.intersection_valid = ti.field(ti.i32, shape=(self.M, 6))
        self.intersection_face = ti.field(ti.i32, shape=(self.M, 6))
        self.C_k = ti.field(ti.f32, shape=(self.M, 4, 6))
        
        # Spring parameters
        self.rest_lengths = ti.field(ti.f32, shape=(self.M, 3))
        self.axial_stiffness = ti.field(ti.f32, shape=(self.M, 3))
        self.torsion_stiffness = ti.field(ti.f32, shape=(self.M, 3))
        self.volume_kappa = ti.field(ti.f32, shape=self.M)
        self.rest_cos_angles = ti.field(ti.f32, shape=(self.M, 3))
        
        # Boundary conditions
        self.boundary_nodes = ti.field(ti.i32, shape=self.N)
        self.boundary_displacement = ti.Vector.field(3, ti.f32, shape=self.N)
        self.is_boundary_constrained = ti.field(ti.i32, shape=self.N)
        
        # SMS measurement rows from preprocessing
        self.r_axis  = ti.Vector.field(12, ti.f32, shape=(self.M, 3))
        self.r_shear = ti.Vector.field(12, ti.f32, shape=(self.M, 3))
        self.r_vol   = ti.Vector.field(12, ti.f32, shape=self.M)
        
        # Static equilibrium solver fields
        self.residual = ti.Vector.field(3, ti.f32, shape=self.N)
        self.solution_increment = ti.Vector.field(3, ti.f32, shape=self.N)
        self.temp_b = ti.field(ti.f32, shape=3*self.N)
        self.temp_x = ti.field(ti.f32, shape=3*self.N)
        self._cur_k = ti.field(dtype=ti.i32, shape=())
        
        self.initial_positions = ti.Vector.field(3, ti.f32, shape=self.N)
        self.displacement_field = ti.Vector.field(3, ti.f32, shape=self.N)
        
        # MODIFICATION: Store mesh connectivity for output
        self.mesh_connectivity = None
        
        # Load preprocessed data and initialize
        self._load_preprocessed_data(data)
        self._initialize_solver_fields()

    def _load_preprocessed_data(self, data):
        """Load all preprocessed data from cone deformation processor"""
        # Basic mesh data
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
        
        # SMS measurement rows
        self.r_axis.from_numpy(data['r_axis'].astype(np.float64))
        self.r_shear.from_numpy(data['r_shear'].astype(np.float64))
        self.r_vol.from_numpy(data['r_vol'].astype(np.float64))

        # Boundary conditions
        self.boundary_nodes.from_numpy(data['boundary_nodes'].astype(np.int32))
        self.initial_positions.from_numpy(data['initial_positions'].astype(np.float64))
        self.displacement_field.from_numpy(data['displacement_field'].astype(np.float64))
        
        # MODIFICATION: Store mesh connectivity for mesh file output
        self.mesh_connectivity = data['tetrahedra'].astype(np.int32)
        
        print(f"Loaded cone preprocessed data: {self.N} nodes, {self.M} tetrahedra")
        print(f"Boundary nodes: {np.sum(data['boundary_nodes'])}")
        print(f"Valid intersections: {np.sum(data['intersection_valid'])}/{data['intersection_valid'].size}")

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

    def apply_cone_boundary_conditions(self):
        """Apply boundary conditions specific to cone geometry - constrain top face"""
        # MODIFICATION: Simplified BC application for cone (clamp top face)
        self._apply_cone_constraints()
        constrained_count = self._count_constrained_nodes()
        print(f"Applied cone boundary conditions to {constrained_count} nodes (top face)")
        self._build_dof_map()

    @ti.kernel
    def _apply_cone_constraints(self):
        """Apply constraints to cone top face - set displacement to zero (clamped)"""
        for i in range(self.N):
            if self.boundary_nodes[i] == 1:  # This is a boundary node (top face)
                # For cone verification, we clamp the top face (zero displacement)
                self.boundary_displacement[i] = ti.Vector([0.0, 0.0, 0.0])
                self.is_boundary_constrained[i] = 1
                # Keep position at initial value (no displacement)
                self.x[i] = self.initial_positions[i]

    def _count_constrained_nodes(self):
        """Count number of nodes with Dirichlet constraints applied"""
        return int(self.is_boundary_constrained.to_numpy().sum())

    def _build_dof_map(self):
        """Build mapping from global DOFs to free DOFs"""
        dof_map_np = -np.ones(3*self.N, dtype=np.int32)
        cnt = 0
        is_bc = self.is_boundary_constrained.to_numpy()
        for i in range(self.N):
            if is_bc[i] == 0:  # Free node
                for d in range(3):
                    dof_map_np[3*i + d] = cnt
                    cnt += 1
        self.n_free_dof = int(cnt)
        self.dof_map = ti.field(ti.i32, shape=3*self.N)
        self.dof_map.from_numpy(dof_map_np)

    def _alloc_free_buffers(self):
        """Allocate buffers for free DOF system"""
        n = int(self.n_free_dof)
        self.b_free = ti.field(ti.f32, shape=n)
        self.x_free = ti.field(ti.f32, shape=n)

    @ti.kernel
    def _build_rhs_free(self, b: ti.template()):
        """Assemble external loads (gravity) on free DOFs only"""
        g = ti.Vector([0.0, 0.0, -9.81])  # m/s^2 (gravity in -Z direction)
        for i in range(self.N):
            if self.is_boundary_constrained[i] == 0:
                rowx = self.dof_map[3*i + 0]
                rowy = self.dof_map[3*i + 1]
                rowz = self.dof_map[3*i + 2]
                if rowx != -1:
                    b[rowx] = self.mass[i] * g[0]
                    b[rowy] = self.mass[i] * g[1]
                    b[rowz] = self.mass[i] * g[2]

    # Optimized free-DOF stiffness assembly
    @ti.func
    def _add_outer_free(self, K: ti.template(), r: ti.template(), coeff: ti.f32):
        """Add outer product contribution to stiffness matrix for free DOFs"""
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
    def _assemble_K_axis_free(self, K: ti.types.sparse_matrix_builder(), alpha: ti.f32):
        """Assemble axial stiffness contribution"""
        for k in range(self.M):
            if self.vol[k] > 1e-12:
                self._cur_k[None] = k
                V = self.vol[k]
                coeff = 4.0 * alpha * V
                for ell in ti.static(range(3)):
                    r = self.r_axis[k, ell]
                    self._add_outer_free(K, r, coeff)

    @ti.kernel
    def _assemble_K_shear_free(self, K: ti.types.sparse_matrix_builder(), beta: ti.f32):
        """Assemble shear stiffness contribution"""
        for k in range(self.M):
            if self.vol[k] > 1e-12:
                self._cur_k[None] = k
                V = self.vol[k]
                coeff = 4.0 * beta * V
                for s in ti.static(range(3)):
                    r = self.r_shear[k, s]
                    self._add_outer_free(K, r, coeff)

    @ti.kernel
    def _assemble_K_vol_free(self, K: ti.types.sparse_matrix_builder(), kappa: ti.f32):
        """Assemble volumetric stiffness contribution"""
        for k in range(self.M):
            if self.vol[k] > 1e-12:
                self._cur_k[None] = k
                V = self.vol[k]
                coeff = kappa * V
                r = self.r_vol[k]
                self._add_outer_free(K, r, coeff)

    @ti.kernel
    def _scatter_solution_free(self, x_free: ti.template()):
        """Scatter solution from free DOFs back to all nodes"""
        for i in range(self.N):
            if self.is_boundary_constrained[i] == 0:
                self.solution_increment[i] = ti.Vector([
                    x_free[self.dof_map[3*i + 0]],
                    x_free[self.dof_map[3*i + 1]],
                    x_free[self.dof_map[3*i + 2]],
                ])
            else:
                self.solution_increment[i] = ti.Vector([0.0, 0.0, 0.0])

    @ti.kernel
    def _update_positions_from_solution(self):
        """Update nodal positions: x = x + delta x (for free nodes only)"""
        for i in range(self.N):
            if self.is_boundary_constrained[i] == 0:  # Free node
                self.x[i] += self.solution_increment[i]
            # Constrained nodes remain at their prescribed positions

    def solve_cone_static_equilibrium(self, alpha: float, beta: float, kappa: float):
        """Solve static equilibrium for cone using SMS approach"""
        print("=== Cone Static Equilibrium (SMS) ===")
        if not hasattr(self, 'dof_map'):
            self._build_dof_map()
        self._alloc_free_buffers()
        
        # Assemble stiffness matrix K_FF
        n = self.b_free.shape[0]
        max_triplets = 400 * self.M + 40 * self.N
        Kb = ti.linalg.SparseMatrixBuilder(n, n, max_num_triplets=max_triplets)
        
        # Three-stage assembly
        self._assemble_K_axis_free(Kb, float(alpha))
        self._assemble_K_shear_free(Kb, float(beta))
        self._assemble_K_vol_free(Kb, float(kappa))
        A = Kb.build()
        print("Stiffness matrix assembled")
        
        # RHS: only external forces (gravity) since boundary nodes are clamped (zero displacement)
        self.b_free.fill(0.0)
        self._build_rhs_free(self.b_free)
        
        self.x_free.fill(0.0)
        
        # Solve using scipy
        print(f"Matrix shape: {A.n} x {A.m}")
        
        try:
            from scipy.sparse.linalg import spsolve
            from scipy.io import mmread
            import tempfile
            import os
            
            print("Using scipy sparse solver...")
            
            # Export matrix and solve
            with tempfile.NamedTemporaryFile(suffix='.mtx', delete=False) as tmp:
                tmp_path = tmp.name
            
            try:
                A.mmwrite(tmp_path)
                A_scipy = mmread(tmp_path).tocsr()
                
                b_np = self.b_free.to_numpy()
                x_np = spsolve(A_scipy, b_np)
                
                self.x_free.from_numpy(x_np)
                print("Scipy solve successful!")
                
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            
        except Exception as e:
            print(f"Solver failed: {e}")
            self.x_free.fill(0.0)
            print("Using zero solution as fallback")
            return False
        
        # Update positions
        self._scatter_solution_free(self.x_free)
        self._update_positions_from_solution()
        return True

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

    def save_results(self, output_path):
        """Save SMS solution results as mesh files"""
        # MODIFICATION: Save mesh files instead of NPZ data files
        
        # Get final deformed positions
        final_positions = self.x.to_numpy()
        initial_positions = self.initial_positions.to_numpy()
        solution_increment = self.solution_increment.to_numpy()
        fem_displacement = self.displacement_field.to_numpy()
        
        # Create mesh data for output
        cells = [("tetra", self.mesh_connectivity)]
        
        # Save initial mesh (reference configuration)
        initial_mesh = meshio.Mesh(points=initial_positions, cells=cells)
        initial_path = f"{output_path}_initial.xdmf"
        meshio.write(initial_path, initial_mesh, data_format="XML")
        print(f"Initial mesh saved to {initial_path}")
        
        # Save FEM reference solution
        fem_final_positions = initial_positions + fem_displacement
        fem_mesh = meshio.Mesh(points=fem_final_positions, cells=cells)
        fem_path = f"{output_path}_fem_reference.xdmf"
        meshio.write(fem_path, fem_mesh, data_format="XML")
        print(f"FEM reference mesh saved to {fem_path}")
        
        # Save SMS solution
        sms_mesh = meshio.Mesh(points=final_positions, cells=cells)
        sms_path = f"{output_path}_sms_solution.xdmf"
        meshio.write(sms_path, sms_mesh, data_format="XML")
        print(f"SMS solution mesh saved to {sms_path}")
        
        # Save displacement comparison mesh with point data
        comparison_mesh = meshio.Mesh(
            points=final_positions,
            cells=cells,
            point_data={
                "SMS_displacement": solution_increment,
                "FEM_displacement": fem_displacement,
                "displacement_error": solution_increment - fem_displacement,
                "displacement_error_magnitude": np.linalg.norm(solution_increment - fem_displacement, axis=1)
            }
        )
        comparison_path = f"{output_path}_comparison.xdmf"
        meshio.write(comparison_path, comparison_mesh, data_format="XML")
        print(f"Comparison mesh with displacement data saved to {comparison_path}")
        
        # Also save summary data as NPZ for analysis
        summary_data = {
            'final_positions': final_positions,
            'initial_positions': initial_positions,
            'solution_increment': solution_increment,
            'boundary_nodes': self.boundary_nodes.to_numpy(),
            'is_constrained': self.is_boundary_constrained.to_numpy(),
            'fem_displacement': fem_displacement,
            'mesh_connectivity': self.mesh_connectivity,
            'mesh_info': {
                'n_nodes': self.N,
                'n_tetrahedra': self.M,
                'n_free_dof': self.n_free_dof if hasattr(self, 'n_free_dof') else 0
            }
        }
        summary_path = f"{output_path}_summary.npz"
        np.savez(summary_path, **summary_data)
        print(f"Summary data saved to {summary_path}")
        
        return {
            'initial_mesh': initial_path,
            'fem_reference': fem_path,
            'sms_solution': sms_path,
            'comparison': comparison_path,
            'summary': summary_path
        }

    def summary(self):
        """Print cone simulation summary"""
        total_mass = self.mass.to_numpy().sum()
        print(f"Cone: {self.N} nodes, {self.M} tets, total mass={total_mass:.6f} kg")
        
        stats = self.get_intersection_stats()
        print(f"Intersection success rate: {stats['success_rate']:.2%}")
        print(f"Average intersections per tet: {stats['per_tet_average']:.1f}/6")

    def compare_with_fem(self):
        """Compare SMS solution with FEM reference"""
        # MODIFICATION: Added quantitative comparison between SMS and FEM solutions
        sms_displacement = self.solution_increment.to_numpy()
        fem_displacement = self.displacement_field.to_numpy()
        
        # Only compare free nodes (non-constrained)
        is_free = self.is_boundary_constrained.to_numpy() == 0
        
        sms_free = sms_displacement[is_free]
        fem_free = fem_displacement[is_free]
        
        # Compute error metrics
        abs_error = np.linalg.norm(sms_free - fem_free, axis=1)
        rel_error = abs_error / (np.linalg.norm(fem_free, axis=1) + 1e-10)
        
        print(f"\n=== SMS vs FEM Comparison ===")
        print(f"Free nodes: {np.sum(is_free)}/{self.N}")
        print(f"Max absolute error: {abs_error.max():.6e} m")
        print(f"Mean absolute error: {abs_error.mean():.6e} m")
        print(f"RMS absolute error: {np.sqrt(np.mean(abs_error**2)):.6e} m")
        print(f"Max relative error: {rel_error.max():.6f}")
        print(f"Mean relative error: {rel_error.mean():.6f}")
        
        return {
            'max_abs_error': abs_error.max(),
            'mean_abs_error': abs_error.mean(),
            'rms_abs_error': np.sqrt(np.mean(abs_error**2)),
            'max_rel_error': rel_error.max(),
            'mean_rel_error': rel_error.mean()
        }


def run_cone_verification(alpha=1785.7143, beta=3571.4286, kappa=14285.7143):
    """Run complete cone verification"""
    preprocessed_data_path = "homo_verification/cone_verification_deformation.npz"
    output_path = "homo_verification/cone_sms_solution"
    
    # Check if preprocessed data exists
    if not Path(preprocessed_data_path).exists():
        print(f"Error: Preprocessed data not found at {preprocessed_data_path}")
        print("Please run cone_deformation_processor.py first")
        return None
    
    print(f"=== Loading Preprocessed Cone Data ===")
    sim = ConeStaticEquilibrium(preprocessed_data_path)
    
    # Apply boundary conditions (clamp top face)
    print("=== Applying Cone Boundary Conditions ===")
    sim.apply_cone_boundary_conditions()
    
    # Print simulation summary
    sim.summary()
    
    # Solve static equilibrium
    print(f"\n=== Solving Cone Static Equilibrium ===")
    print(f"Parameters: alpha={alpha}, beta={beta}, kappa={kappa}")
    success = sim.solve_cone_static_equilibrium(alpha, beta, kappa)
    
    if success:
        print("SMS solution successful!")
        
        # Compare with FEM reference
        comparison = sim.compare_with_fem()
        
        # MODIFICATION: Save mesh files instead of NPZ
        mesh_files = sim.save_results(output_path)
        
        # Print saved file information
        print(f"\n=== Output Files ===")
        for file_type, path in mesh_files.items():
            print(f"  {file_type}: {path}")
        
        return comparison
    else:
        print("SMS solution failed!")
        return None


if __name__ == "__main__":
    # MODIFICATION: Updated to handle mesh file outputs
    # Run cone verification with calibrated parameters
    comparison = run_cone_verification(alpha=1785.7143, beta=3571.4286, kappa=14285.7143)
    
    if comparison:
        print(f"\n=== Verification Complete ===")
        print(f"The SMS approach achieved:")
        print(f"  Max absolute error: {comparison['max_abs_error']:.6e} m")
        print(f"  Mean relative error: {comparison['mean_rel_error']:.4%}")
        print(f"\nMesh files saved for visualization:")
        print(f"  - Initial configuration: *_initial.xdmf")
        print(f"  - FEM reference solution: *_fem_reference.xdmf") 
        print(f"  - SMS solution: *_sms_solution.xdmf")
        print(f"  - Comparison with error data: *_comparison.xdmf")
