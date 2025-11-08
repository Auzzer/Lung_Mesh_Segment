# Two-region constant inverse: optimize only two alpha scalars (background & special)
# Based on inverse_alpha_2region_reg_log.py but with L-BFGS optimizer
#
# Parameterization:
#   Let theta = log(alpha)
#   Then alpha_bg = exp(theta_bg), alpha_sp = exp(theta_sp)
#
# Loss function:
#   L = L_data + lambda_F * R(u_sim)
#   where L_data is the data misfit term
#   and R(u_sim) = sum_k V_k * sqrt(||nabla u_sim,k||_F^2 + epsilon^2)
#   is the Charbonnier-smoothed TV regularization on displacement gradient
#
# Gradients computed via adjoint method:
#   The adjoint RHS includes both data misfit and regularization:
#   partial L/partial u_sim = (2/||u_obs||^2) * (u_sim - u_obs) + lambda_F * partial R/partial u_sim
#   
#   Then solve K(alpha) * lambda = partial L/partial u_sim for adjoint variable lambda
#   
#   Per-region gradients:
#   partial L/partial alpha_bg = sum_{k in background} lambda_k^T * partial K_k/partial alpha * u_k
#   partial L/partial alpha_sp = sum_{k in special} lambda_k^T * partial K_k/partial alpha * u_k
#
# Regularization uses Charbonnier-smoothed Total Variation on deformation gradient
# as described in doc/solver_torch.md section 7.5
#
# Optimizer: L-BFGS forcing strong Wolfe line search
# Results are saved to 2region_constant_inv_reg_LBFGS_{learning_rate} folder
#
# Usage:
#   Default (u_obs from ground truth):
#     python inverse_alpha_2region_reg_log_lbfgs.py
#   
#   Custom u_obs from alternative source (e.g., FEM results):
#     python inverse_alpha_2region_reg_log_lbfgs.py --u-obs-source /path/to/alternative_data.npz
#
#   The alternative source .npz file must contain: alpha_k, beta_k, kappa_k arrays.

import json
import sys
import argparse
import taichi as ti
import numpy as np
import torch
import meshio
from pathlib import Path
from typing import Optional, Callable

# Import FEM and preprocessing modules
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# Import after adding to path
import hetero_cone_fem
import hetero_cone_deformation_processor


GRAMS_PER_KG = 1000.0

# Poisson's ratio for lung tissue (constant)
POISSON_RATIO = 0.4

# Regularization hyperparameters 
REG_WEIGHT = 5e-2   # lambda_F: weight for Charbonnier TV regularization on displacement gradient
EPS_REG    = 1e-4   # epsilon: Charbonnier smoothing parameter
EPS_DIV    = 1e-12  # Stabilizer for data loss denominator

def compute_beta_from_alpha(alpha: torch.Tensor) -> torch.Tensor:
    """Compute beta from alpha using the relationship beta = mu = 2*alpha."""
    return 2.0 * alpha

def compute_kappa_from_alpha(alpha: torch.Tensor, nu: float = POISSON_RATIO) -> torch.Tensor:
    """Compute kappa from alpha using the relationship kappa = lambda = 4*alpha*nu/(1-2*nu)."""
    return 4.0 * alpha * nu / (1.0 - 2.0 * nu)


ti.init(arch=ti.cuda, debug=False, kernel_profiler=False)

def torch_solve_sparse(
    A_sp: torch.Tensor,
    b: torch.Tensor,
    tol: float = 1e-4,
    max_iter: int = 2000,
):
    """
    Solve the linear system A x = b using torch's sparse direct solver.
    Returns `(solution, iterations_used, residual_norm, converged_flag, spd_ok)`.
    """
    
    _ = max_iter  

    A = A_sp.to_dense().to(torch.float64)
    b = b.reshape(-1).to(torch.float64)
    if b.numel() == 0:
        return torch.zeros_like(b), 0, 0.0, True, True
    _, info = torch.linalg.cholesky_ex(A); spd_ok = (int(info)==0)
    x = torch.linalg.solve(A, b.unsqueeze(-1)).squeeze(-1)
    residual = A @ x - b
    res_norm = float(torch.linalg.norm(residual).item())
    converged = torch.allclose(residual, torch.zeros_like(residual), rtol=tol, atol=tol)
    
    return x, 1, res_norm, bool(converged), spd_ok

# --------------------------------
# Data-oriented solver class
# --------------------------------
@ti.data_oriented
class ConeStaticEquilibrium:
    def __init__(self, preprocessed_data_path: str):
        data = np.load(preprocessed_data_path, allow_pickle=True)

        # Mesh sizes
        self.N = int(data['mesh_points'].shape[0])   # nodes
        self.M = int(data['tetrahedra'].shape[0])    # tets

        # Core fields (Taichi f64 for numerical data, i32 for indices)
        self.x        = ti.Vector.field(3, ti.f64, shape=self.N)       # positions
        self.tets     = ti.Vector.field(4, ti.i32, shape=self.M)       # tet node ids (must be i32 for indexing)
        self.vol      = ti.field(ti.f64, shape=self.M)                 # per-tet volume
        self.mass     = ti.field(ti.f64, shape=self.N)                 # per-node mass
        self.labels   = ti.field(ti.i32, shape=self.M)                 # labels (i32 for indexing)

        # BCs
        self.boundary_nodes        = ti.field(ti.i32, shape=self.N)    # i32 for indexing
        self.boundary_displacement = ti.Vector.field(3, ti.f64, shape=self.N)
        self.is_boundary_constrained = ti.field(ti.i32, shape=self.N)  # i32 for indexing

        # SMS rows: r in R^{1x12} (3 axis, 3 shear, 1 volumetric)
        self.r_axis  = ti.Vector.field(12, ti.f64, shape=(self.M, 3))
        self.r_shear = ti.Vector.field(12, ti.f64, shape=(self.M, 3))
        self.r_vol   = ti.Vector.field(12, ti.f64, shape=self.M)

        # Misc fields kept from script
        self.initial_positions  = ti.Vector.field(3, ti.f64, shape=self.N)
        self.displacement_field = ti.Vector.field(3, ti.f64, shape=self.N)

        # Internal buffers / mapping
        self._cur_k = ti.field(ti.i32, shape=())          # i32 for loop counters
        self._nnz_counter = ti.field(ti.i32, shape=())    # i32 for counters

        # Load numpy blobs into Taichi fields
        self._load_preprocessed_data(data)

        # Build BCs and DOF map
        self.apply_cone_boundary_conditions()
        self._build_dof_map()
        self._alloc_free_buffers()

        # Cached assembled matrix and latest solution
        self._K_sparse = None
        self._K_sparse_shape = None
        self._last_u_star = None
        self._last_forward_status = None
        self._last_backward_status = None

    # ----------------------------
    # Data loading 
    # ----------------------------
    def _load_preprocessed_data(self, data):
        self.x.from_numpy(data['mesh_points'].astype(np.float64))
        self.tets.from_numpy(data['tetrahedra'].astype(np.int32))     # i32 for Taichi indexing
        self.vol.from_numpy(data['volume'].astype(np.float64))
        self.mass.from_numpy(data['mass'].astype(np.float64))

        self.boundary_nodes.from_numpy(data['boundary_nodes'].astype(np.int32))  # i32 for indexing
        self.initial_positions.from_numpy(data['initial_positions'].astype(np.float64))
        self.displacement_field.from_numpy(data['displacement_field'].astype(np.float64))

        self.r_axis.from_numpy(data['r_axis'].astype(np.float64))
        self.r_shear.from_numpy(data['r_shear'].astype(np.float64))
        self.r_vol.from_numpy(data['r_vol'].astype(np.float64))
        if 'labels' not in data:
            raise ValueError("Preprocessed data missing 'labels'")
        self.labels.from_numpy(data['labels'].astype(np.int32))        # i32 for indexing

    # ----------------------------
    # Boundary conditions
    # ----------------------------
    def apply_cone_boundary_conditions(self):
        self._apply_cone_constraints()
        self._initialize_aux_vectors()

    @ti.kernel
    def _apply_cone_constraints(self):
        for i in range(self.N):
            if self.boundary_nodes[i] == 1:
                self.boundary_displacement[i] = ti.Vector([0.0, 0.0, 0.0])
                self.is_boundary_constrained[i] = 1
                self.x[i] = self.initial_positions[i]
            else:
                self.is_boundary_constrained[i] = 0

    def _build_dof_map(self):
        """
        Build map: global_dof(3*N) -> [0..n_free-1] or -1 if constrained.
        """
        dof_map_np = -np.ones(3 * self.N, dtype=np.int32)  # i32 for Taichi indexing
        cnt = 0
        is_bc = self.is_boundary_constrained.to_numpy()
        for i in range(self.N):
            if is_bc[i] == 0:
                for d in range(3):
                    dof_map_np[3 * i + d] = cnt
                    cnt += 1
        self.n_free_dof = int(cnt)
        self.dof_map = ti.field(dtype=ti.i32, shape=3*self.N)  # i32 for indexing
        self.dof_map.from_numpy(dof_map_np)

    def _alloc_free_buffers(self):
        n = int(self.n_free_dof)
        self._b_free_ti = ti.field(dtype=ti.f64, shape=n)  # Taichi-side RHS if needed
        
        # Build gradient operator G for regularization
        self._build_gradient_operator()

    def _build_gradient_operator(self):
        """
        Build the gradient operator G that maps free DOFs to element-wise deformation gradients.
        G: R^{3N_f} -> R^{9M}
        For each element k, G_k = (B_k ⊗ I_3) where B_k contains shape function gradients.
        Uses Taichi for parallel assembly.
        """
        # Allocate fields for shape function gradients
        self.shape_grad = ti.Vector.field(3, ti.f64, shape=(self.M, 4))  # (M, 4) x 3D gradients
        
        # Precompute shape function gradients in parallel
        self._compute_shape_gradients()
        
        # Conservative capacity: 36 non-zeros per element (4 nodes * 3 components * 3 gradient dims)
        cap = int(36 * self.M)
        rows = torch.empty(cap, device='cuda', dtype=torch.int64)
        cols = torch.empty(cap, device='cuda', dtype=torch.int64)
        vals = torch.empty(cap, device='cuda', dtype=torch.float64)
        
        # Assemble triplets in parallel
        self._nnz_counter[None] = 0
        self._assemble_G_triplets(rows, cols, vals, cap)
        
        # Get actual number of non-zeros
        nnz = int(self._nnz_counter.to_numpy().item())
        if nnz == 0:
            raise RuntimeError("Gradient operator assembly produced zero non-zeros")
        
        # Trim and build sparse matrix
        rows = rows[:nnz]
        cols = cols[:nnz]
        vals = vals[:nnz]
        
        indices = torch.stack([rows, cols], dim=0)
        self._G_operator = torch.sparse_coo_tensor(indices, vals, (9 * self.M, self.n_free_dof), dtype=torch.float64, device='cuda')
        self._G_operator = self._G_operator.coalesce()

    @ti.kernel
    def _compute_shape_gradients(self):
        """Compute shape function gradients for all elements in parallel."""
        for k in range(self.M):
            # Get vertex positions
            v0 = self.x[self.tets[k][0]]
            v1 = self.x[self.tets[k][1]]
            v2 = self.x[self.tets[k][2]]
            v3 = self.x[self.tets[k][3]]
            
            # Build edge matrix D_m = [v1-v0, v2-v0, v3-v0]
            D_m = ti.Matrix.cols([v1 - v0, v2 - v0, v3 - v0])
            
            # Compute determinant
            det = D_m.determinant()
            
            # Skip degenerate elements
            if ti.abs(det) < 1e-12:
                for a in ti.static(range(4)):
                    self.shape_grad[k, a] = ti.Vector([0.0, 0.0, 0.0])
                continue
            
            # Compute inverse: D_m_inv
            D_m_inv = D_m.inverse()
            
            # Shape function gradients from columns of D_m_inv^T
            # nabla N_1, nabla N_2, nabla N_3 are columns of D_m_inv
            self.shape_grad[k, 1] = ti.Vector([D_m_inv[0, 0], D_m_inv[1, 0], D_m_inv[2, 0]])
            self.shape_grad[k, 2] = ti.Vector([D_m_inv[0, 1], D_m_inv[1, 1], D_m_inv[2, 1]])
            self.shape_grad[k, 3] = ti.Vector([D_m_inv[0, 2], D_m_inv[1, 2], D_m_inv[2, 2]])
            
            # nabla N_0 = -(nabla N_1 + nabla N_2 + nabla N_3)
            self.shape_grad[k, 0] = -(self.shape_grad[k, 1] + self.shape_grad[k, 2] + self.shape_grad[k, 3])

    @ti.kernel
    def _assemble_G_triplets(self,
                            rows: ti.types.ndarray(),
                            cols: ti.types.ndarray(),
                            vals: ti.types.ndarray(),
                            cap: ti.i32):
        """Assemble gradient operator G as triplets in parallel."""
        self._nnz_counter[None] = 0
        
        for k in range(self.M):
            # For each element, build G_k = (B_k ⊗ I_3)
            # B_k = [nabla N_0, nabla N_1, nabla N_2, nabla N_3]
            
            for a in ti.static(range(4)):  # node index
                node_id = self.tets[k][a]
                grad_N_a = self.shape_grad[k, a]
                
                for d in ti.static(range(3)):  # displacement component (x, y, z)
                    global_dof = 3 * node_id + d
                    free_dof = self.dof_map[global_dof]
                    
                    # Only process free DOFs (avoid continue in static loop)
                    if free_dof >= 0:
                        # This DOF contributes to gradient components
                        # nabla u has shape (3, 3): [partial u_i/partial X_j]
                        # Vectorized: [partial u_0/partial X_0, partial u_1/partial X_0, partial u_2/partial X_0, partial u_0/partial X_1, ...]
                        # Component d of displacement at node a contributes to row (d*3+i) of vec(nabla u)
                        # with weight grad_N_a[i]
                        
                        for i in ti.static(range(3)):  # gradient spatial index
                            grad_comp = d * 3 + i  # which component of vec(nabla u)
                            row_idx = k * 9 + grad_comp
                            col_idx = free_dof
                            value = grad_N_a[i]
                            
                            idx = ti.atomic_add(self._nnz_counter[None], 1)
                            if idx < cap:
                                rows[idx] = row_idx
                                cols[idx] = col_idx
                                vals[idx] = value

    def compute_charbonnier_regularization(self, u_free: torch.Tensor, epsilon: float = 1e-4) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Charbonnier-smoothed TV regularization on displacement gradient.
        
        R(u) = sum_k V_k * sqrt(||nabla u_k||_F^2 + epsilon^2)
        
        Args:
            u_free: Free DOF displacement vector (n_free,)
            epsilon: Charbonnier smoothing parameter
            
        Returns:
            reg_value: Regularization value (scalar)
            reg_grad: Gradient w.r.t. u_free (n_free,)
        """
        # Compute gradient field: grad_u = G @ u_free, shape (9M,)
        grad_u = torch.sparse.mm(self._G_operator, u_free.unsqueeze(1)).squeeze(1)  # (9M,)
        
        # Reshape to (M, 9) for per-element processing
        grad_u_elements = grad_u.view(self.M, 9)  # (M, 9)
        
        # Compute Frobenius norm squared for each element
        grad_norm_sq = torch.sum(grad_u_elements ** 2, dim=1)  # (M,)
        
        # Charbonnier function: sqrt(||nabla u||_F^2 + eps^2)
        charbonnier = torch.sqrt(grad_norm_sq + epsilon ** 2)  # (M,)
        
        # Get volumes
        vol_np = self.vol.to_numpy()
        vol_torch = torch.from_numpy(vol_np).to(device='cuda', dtype=torch.float64)  # (M,)
        
        # Regularization value: sum_k V_k * charbonnier_k
        reg_value = torch.sum(vol_torch * charbonnier)
        
        # Gradient: partial R/partial u = G^T @ W @ (G @ u)
        # where W is block-diagonal: W_k = V_k / charbonnier_k * I_9
        weights = vol_torch / charbonnier  # (M,)
        
        # Expand to 9M: repeat each weight 9 times
        weights_expanded = weights.repeat_interleave(9)  # (9M,)
        
        # Weighted gradient field
        weighted_grad = weights_expanded * grad_u  # (9M,)
        
        # Apply G^T
        reg_grad = torch.sparse.mm(self._G_operator.t(), weighted_grad.unsqueeze(1)).squeeze(1)  # (n_free,)
        
        return reg_value, reg_grad

    @ti.kernel
    def _initialize_aux_vectors(self):
        for i in range(self.N):
            self.boundary_displacement[i] = ti.Vector([0.0, 0.0, 0.0])

    # ------------------------------------------
    # Build gravity RHS on free DOFs (Torch)
    # ------------------------------------------
    @ti.kernel
    def _build_rhs_free_into_torch(self, b_free: ti.types.ndarray()):
        g = ti.Vector([0.0, 0.0, -9.81])
        for i in range(self.N):
            if self.is_boundary_constrained[i] == 0:
                rowx = self.dof_map[3*i + 0]
                rowy = self.dof_map[3*i + 1]
                rowz = self.dof_map[3*i + 2]
                if rowx != -1:
                    b_free[rowx] = self.mass[i] * g[0]
                    b_free[rowy] = self.mass[i] * g[1]
                    b_free[rowz] = self.mass[i] * g[2]

    # ------------------------------------------
    # Assemble K_FF as triplets into Torch
    # ------------------------------------------
    @ti.kernel
    def _assemble_triplets(
        self,
        rows: ti.types.ndarray(),    # len >= cap
        cols: ti.types.ndarray(),    
        vals: ti.types.ndarray(),    
        cap: ti.i32,              # maximum number of non-zero entries (i32 for Taichi)
        alpha: ti.types.ndarray(),   # len M
        beta:  ti.types.ndarray(),   
        kappa: ti.types.ndarray()    
    ):
        self._nnz_counter[None] = 0
        for k in range(self.M):
            V = self.vol[k]
            alpha_k = alpha[k]; beta_k = beta[k]; kappa_k = kappa[k]
            # Global free-dof indices for this tet's 12 local dofs
            g = ti.Vector.zero(ti.i32, 12)  # i32 for indexing
            for vi in ti.static(range(4)):
                node = self.tets[k][vi]
                for d in ti.static(range(3)):
                    g[3*vi + d] = self.dof_map[3*node + d]

            # Load r-rows once
            ax0 = self.r_axis[k, 0]; ax1 = self.r_axis[k, 1]; ax2 = self.r_axis[k, 2]
            sh0 = self.r_shear[k, 0]; sh1 = self.r_shear[k, 1]; sh2 = self.r_shear[k, 2]
            rv  = self.r_vol[k]

            # Each tet contributes a symmetric 12x12 block.
            for p in ti.static(range(12)):
                gp = g[p]
                for q in ti.static(range(12)):
                    gq = g[q]
                    if gp != -1 and gq != -1:
                        # Combine axis + shear + volumetric contributions
                        s_ax = ax0[p]*ax0[q] + ax1[p]*ax1[q] + ax2[p]*ax2[q]
                        s_sh = sh0[p]*sh0[q] + sh1[p]*sh1[q] + sh2[p]*sh2[q]
                        s_v  = rv[p]*rv[q]
                        val = V * (4.0 * alpha_k * s_ax + 4.0 * beta_k * s_sh + kappa_k * s_v)

                        idx = ti.atomic_add(self._nnz_counter[None], 1)
                        if idx < cap:
                            rows[idx] = gp  # already i32, stored in i64 torch tensor
                            cols[idx] = gq  # already i32, stored in i64 torch tensor
                            vals[idx] = val
                     

    def _assemble_K_torch(self,
                          alpha_t: torch.Tensor,
                          beta_t:  torch.Tensor,
                          kappa_t: torch.Tensor):
        """
        Assemble K_FF into a Torch CSR sparse tensor on CUDA.
        alpha_t, beta_t, kappa_t: shape (M,), float64, CUDA
        """
        assert alpha_t.is_cuda and beta_t.is_cuda and kappa_t.is_cuda
        assert alpha_t.dtype == torch.float64
        n_free = int(self.n_free_dof)

        # Conservative capacity: one 12x12 block per tet => 144 entries/tet.
        cap = int(144 * self.M)
        rows = torch.empty(cap, device='cuda', dtype=torch.int64)
        cols = torch.empty(cap, device='cuda', dtype=torch.int64)
        vals = torch.empty(cap, device='cuda', dtype=torch.float64)
        
        # Fill triplets from Taichi (zero-copy into Torch buffers)
        self._assemble_triplets(rows, cols, vals, cap, alpha_t, beta_t, kappa_t)
        
        # Fetch how many entries were written
        nnz = int(self._nnz_counter.to_numpy().item())  
        if nnz == 0:
            raise RuntimeError("Assembly produced zero non-zeros; check inputs/BCs")
        
        rows = rows[:nnz]
        cols = cols[:nnz]
        vals = vals[:nnz]
        
        # Build COO => combine duplicate (row,col) pairs across tets
        A_coo = torch.sparse_coo_tensor(
            torch.vstack([rows, cols]),
            vals,
            size=(n_free, n_free),
            device='cuda',
            dtype=torch.float64,
        ).coalesce()
        return A_coo.to_sparse_csr()

    def assemble_matrix(self,
                        alpha: torch.Tensor,
                        beta: torch.Tensor,
                        kappa: torch.Tensor):
        """Assemble and cache the stiffness matrix for the given parameters."""
        assert alpha.is_cuda and beta.is_cuda and kappa.is_cuda
        assert alpha.dtype == torch.float64 and beta.dtype == torch.float64 and kappa.dtype == torch.float64
        alpha = alpha.detach().contiguous()
        beta = beta.detach().contiguous()
        kappa = kappa.detach().contiguous()
        
        A_sp = self._assemble_K_torch(alpha, beta, kappa)
        self._K_sparse = A_sp
        self._K_sparse_shape = A_sp.shape
        self._last_u_star = None
        return A_sp

    def get_observed_free(self) -> torch.Tensor:
        """Convert the FEM displacement field to a free-DOF torch vector in metres."""
        if self._K_sparse is None:
            raise RuntimeError("Call assemble_matrix(...) before requesting observed displacements.")
        disp_np = self.displacement_field.to_numpy().astype(np.float64)  # (N, 3)
        flat = disp_np.reshape(-1)
        dof_map_np = self.dof_map.to_numpy()  # (3N,)
        
        free = np.zeros(self.n_free_dof, dtype=np.float64)
        mask = dof_map_np >= 0
        free[dof_map_np[mask]] = flat[mask]
        return torch.from_numpy(free).to(self._K_sparse.device)

    def get_total_mass_grams(self) -> float:
        """Return the total nodal mass expressed in grams."""
        mass_np = self.mass.to_numpy()
        return float(mass_np.sum() * GRAMS_PER_KG)

    @ti.kernel
    def _scatter_free_to_node3(self,
                               u_free: ti.types.ndarray(),
                               out_node3: ti.types.ndarray()):
        # out_node3: (N, 3)
        for i in range(self.N):
            if self.is_boundary_constrained[i] == 0:
                out_node3[i, 0] = u_free[self.dof_map[3*i + 0]]
                out_node3[i, 1] = u_free[self.dof_map[3*i + 1]]
                out_node3[i, 2] = u_free[self.dof_map[3*i + 2]]
            else:
                out_node3[i, 0] = 0.0
                out_node3[i, 1] = 0.0
                out_node3[i, 2] = 0.0

    # ------------------------------------------
    # Gradient kernel: sum per-region gradients
    # ------------------------------------------
    @ti.kernel
    def _grad_2region_kernel(self,
                             grad_bg: ti.types.ndarray(),      # shape (1,)
                             grad_sp: ti.types.ndarray(),      # shape (1,)
                             u_free:  ti.types.ndarray(),      # (n_free,)
                             lam_free: ti.types.ndarray(),     # (n_free,)
                             nu: ti.f64,                       # Poisson's ratio
                             bg_label: ti.i32                  # Background label value
                             ):
        """
        Compute per-region gradients by summing element contributions.
        
        For each element k:
            q_k = g_alpha_k + 2*g_beta_k + c_nu*g_kappa_k
        
        Then:
            grad_bg = sum_{k in background} q_k
            grad_sp = sum_{k in special} q_k
        
        Args:
            bg_label: The label value identifying background elements
        """
        # Chain rule coefficients
        coeff_beta = 2.0
        coeff_kappa = 4.0 * nu / (1.0 - 2.0 * nu)
        
        # Initialize sums
        sum_bg = 0.0
        sum_sp = 0.0
        
        for k in range(self.M):
            V = self.vol[k]
            if V <= 1e-12:
                continue
            
            # Gather local 12-DOF vectors
            u_loc   = ti.Vector.zero(ti.f64, 12)
            lam_loc = ti.Vector.zero(ti.f64, 12)
            for vi in ti.static(range(4)):
                node = self.tets[k][vi]
                for d in ti.static(range(3)):
                    dof_idx = 3 * node + d
                    gi = ti.i32(self.dof_map[dof_idx])
                    idx = 3*vi + d
                    if gi >= 0:
                        u_loc[idx]   = u_free[gi]
                        lam_loc[idx] = lam_free[gi]
                    else:
                        u_loc[idx]   = 0.0
                        lam_loc[idx] = 0.0
                        
            # Axis sum
            s_ax = 0.0
            for ell in ti.static(range(3)):
                r = self.r_axis[k, ell]
                su = 0.0
                sl = 0.0
                for q in ti.static(range(12)):
                    su += r[q] * u_loc[q]
                    sl += r[q] * lam_loc[q]
                s_ax += su * sl
            
            # Shear sum
            s_sh = 0.0
            for s in ti.static(range(3)):
                r = self.r_shear[k, s]
                su = 0.0
                sl = 0.0
                for q in ti.static(range(12)):
                    su += r[q] * u_loc[q]
                    sl += r[q] * lam_loc[q]
                s_sh += su * sl
                
            # Volumetric
            rv = self.r_vol[k]
            suv = 0.0
            slv = 0.0
            for q in ti.static(range(12)):
                suv += rv[q] * u_loc[q]
                slv += rv[q] * lam_loc[q]
                
            # Compute individual gradients
            g_alpha = -4.0 * V * s_ax
            g_beta  = -4.0 * V * s_sh
            g_kappa = -1.0 * V * (suv * slv)
            
            # Combine with chain rule
            q_k = g_alpha + coeff_beta * g_beta + coeff_kappa * g_kappa
            
            # Add to appropriate region sum using the explicit background label
            label = self.labels[k]
            if label == bg_label:  # background
                sum_bg += q_k
            else:  # special (anything that's not background)
                sum_sp += q_k
        
        # Write out the sums
        grad_bg[0] = sum_bg
        grad_sp[0] = sum_sp

    # ============================================================
    # Public functions
    # ============================================================

    def forward(self,
                b_free: torch.Tensor | None = None,
                tol: float = 1e-6,
                max_iter: int = 200):
        """Forward solve using the matrix assembled via `assemble_matrix`."""
       
        n_free = int(self.n_free_dof)
        A_sp = self._K_sparse

        if b_free is None:
            b_free = torch.empty(n_free, device='cuda', dtype=torch.float64)
            self._build_rhs_free_into_torch(b_free)
        else:
            if not (b_free.is_cuda and b_free.dtype == torch.float64 and b_free.numel() == n_free):
                raise ValueError("b_free must be a CUDA float64 tensor with length n_free")

        u_star, iters, res, converged, spd_ok = torch_solve_sparse(
            A_sp, b_free, tol=tol, max_iter=max_iter
        )
        self._last_u_star = u_star
        self._last_forward_status = {
            "iterations": iters,
            "residual_norm": res,
            "converged": bool(converged),
            "tolerance": tol,
            "max_iter": max_iter,
            "spd": spd_ok
        }
        return u_star

    def backward(self,
                 u_obs_free: torch.Tensor,
                 alpha_bg_val: torch.Tensor,
                 alpha_sp_val: torch.Tensor,
                 bg_label: int,
                 tol: float = 1e-6,
                 max_iter: int = 200,
                 reg_weight: float = REG_WEIGHT,
                 eps_reg: float = EPS_REG,
                 eps_div: float = EPS_DIV):
        """Backward pass: compute per-region gradients with Charbonnier-smoothed TV regularization on displacement gradient.
        
        Args:
            u_obs_free: Observed displacement on free DOFs
            alpha_bg_val: Current background alpha value (scalar tensor)
            alpha_sp_val: Current special region alpha value (scalar tensor)
            bg_label: Label value for background region (for gradient splitting)
            tol: Solver tolerance
            max_iter: Maximum solver iterations
            reg_weight: Regularization weight (lambda_F)
            eps_reg: Charbonnier smoothing parameter (epsilon)
            eps_div: Stabilizer for data loss denominator
        
        Returns:
            loss_total: total loss (data + regularization)
            loss_data: data term only
            loss_reg: regularization term only
            grad_alpha_bg: gradient for background alpha
            grad_alpha_sp: gradient for special alpha
        """

        u_obs_free = u_obs_free.contiguous()
        u_obs_free_m = u_obs_free.to(torch.float64)
        u_star_free = self._last_u_star
        
        # Data loss 
        obs_norm_sq = torch.dot(u_obs_free_m, u_obs_free_m)
        denom = obs_norm_sq + torch.tensor(eps_div, device=obs_norm_sq.device, dtype=obs_norm_sq.dtype)

        diff = u_star_free - u_obs_free_m
        loss_data = torch.dot(diff, diff) / denom
        
        # Charbonnier regularization on displacement gradient
        loss_reg, reg_grad_u = self.compute_charbonnier_regularization(u_star_free, epsilon=eps_reg)
        loss_reg = reg_weight * loss_reg
        
        # Combined adjoint RHS: data term + regularization term
        inverse_rhs = (2.0 / denom) * diff + reg_weight * reg_grad_u

        lam_free, iters, res, converged, spd_ok = torch_solve_sparse(
            self._K_sparse, inverse_rhs, tol=tol, max_iter=max_iter
        )

        # Compute per-region data+reg gradients (combined in lambda)
        grad_bg_total = torch.zeros(1, device='cuda', dtype=torch.float64)
        grad_sp_total = torch.zeros(1, device='cuda', dtype=torch.float64)

        self._grad_2region_kernel(
            grad_bg_total, grad_sp_total,
            u_star_free, lam_free, POISSON_RATIO, int(bg_label)
        )
        
        # Total loss
        loss_total = loss_data + loss_reg
        
        self._last_backward_status = {
            "iterations": iters,
            "residual_norm": res,
            "converged": bool(converged),
            "tolerance": tol,
            "max_iter": max_iter,
            "spd": bool(spd_ok),
            "loss_denominator": float(denom.item()),
            "eps_div": eps_div,
            "loss_data": float(loss_data.item()),
            "loss_reg": float(loss_reg.item()),
            "loss_total": float(loss_total.item()),
            "reg_weight": reg_weight,
            "eps_reg": eps_reg,
        }

        return loss_total, loss_data, loss_reg, grad_bg_total[0], grad_sp_total[0]

    def get_last_forward_status(self):
        """Return metadata for the most recent forward torch solve."""
        return self._last_forward_status

    def get_last_forward_solution_m(self) -> Optional[torch.Tensor]:
        """Return the latest forward displacement vector in metres."""
        if self._last_u_star is None:
            return None
        return self._last_u_star

    def get_last_backward_status(self):
        """Return metadata for the most recent backward torch solve."""
        return self._last_backward_status


def save_parameter_heatmap(
    sim: ConeStaticEquilibrium,
    alpha_t: torch.Tensor,
    output_path: Path,
    labels_np: Optional[np.ndarray] = None,
    log_fn: Optional[Callable[[str], None]] = None,
) -> None:

    if log_fn is None:
        log_fn = print

    alpha_np = alpha_t.detach().cpu().numpy().astype(np.float64)

    points = sim.initial_positions.to_numpy().astype(np.float64)
    connectivity = sim.tets.to_numpy().astype(np.int32)

    cell_data = {"alpha": [alpha_np]}
    if labels_np is None:
        labels_np = sim.labels.to_numpy()
    cell_data["labels"] = [labels_np.astype(np.int32)]

    mesh = meshio.Mesh(points=points, cells=[("tetra", connectivity)], cell_data=cell_data)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    meshio.write(str(output_path), mesh, file_format="xdmf")
    log_fn(f"Saved parameter heatmap to {output_path}")


# ------------------------------------------
# FEM Generation and Preprocessing
# ------------------------------------------
def run_fem_generation(base_dir: Path, log_fn: Callable[[str], None]) -> bool:
    """Run FEM generation using hetero_cone_fem.py"""
    log_fn("\n=== STEP 1: FEM Generation ===")
    
    mesh_path = base_dir / "xdmf_visualization" / "cone.xdmf"
    if not mesh_path.exists():
        log_fn(f"ERROR: Cone mesh not found at {mesh_path}")
        log_fn("Please run hetero_cone_gen.py first to generate the cone mesh")
        return False
    
    log_fn(f"Found cone mesh: {mesh_path}")
    
    # Run FEM simulation
    log_fn("Running heterogeneous cone FEM simulation...")
    try:
        hetero_cone_fem.solve_cone_droop(
            mesh_path=mesh_path,
            output_dir=base_dir,
            E=1e4,                      # Young's modulus baseline (Pa)
            rho=200.0,                  # Density (kg/m^3)
            nu=0.4,                     # Poisson ratio
            g_vec=(0.0, 0.0, -9.81),   # Gravity acceleration (m/s^2)
            top_tol=1e-4,               # Tolerance for top face boundary detection
            background_fluctuation=0.0, # No random material variation
            E_special_mult=2.0,         # Elastic modulus multiplier for special region (2x stiffer)
            rho_special_mult=1.0,       # Density multiplier for special region
            seed=42,                    # Fixed seed for reproducibility
            region_params=None,
        )
        log_fn("FEM simulation completed successfully")
        return True
    except Exception as e:
        log_fn(f"ERROR during FEM simulation: {e}")
        return False


def run_preprocessing(base_dir: Path, log_fn: Callable[[str], None]) -> bool:
    """Run deformation preprocessing using hetero_cone_deformation_processor.py"""
    log_fn("\n=== STEP 2: Deformation Preprocessing ===")
    
    xdmf_dir = base_dir / "xdmf_visualization"
    data_dir = base_dir / "cone_data"
    
    original_mesh_path = xdmf_dir / "cone_fem_initial.xdmf"
    deformed_mesh_path = xdmf_dir / "cone_fem_droop.xdmf"
    fem_data_path = data_dir / "cone_fem_droop.npz"
    output_path = base_dir / "cone_verification_deformation"
    
    # Check required files
    for path in (original_mesh_path, deformed_mesh_path, fem_data_path):
        if not path.exists():
            log_fn(f"ERROR: Required file missing: {path}")
            return False
    
    log_fn("Loading cone mesh data...")
    try:
        pts, tets, lbls, displacement_vectors = hetero_cone_deformation_processor.load_cone_mesh_data(
            str(original_mesh_path), str(deformed_mesh_path)
        )
        
        # Load exact material properties from FEM simulation
        fem_data = np.load(fem_data_path)
        
        mu_cells = fem_data["mu_cells"].astype(np.float64)
        lambda_cells = fem_data["lambda_cells"].astype(np.float64)
        rho_cells = fem_data["rho_cells"].astype(np.float64)
        
        # Compute SMS material parameters
        alpha_cells = 0.5 * mu_cells
        beta_cells = mu_cells
        kappa_cells = lambda_cells
        
        log_fn(f"Material properties: mu={mu_cells.mean():.2f}±{mu_cells.std():.2f} Pa")
        log_fn(f"SMS parameters: alpha={alpha_cells.mean():.2f}±{alpha_cells.std():.2f} Pa")
        
        log_fn("Creating deformation processor...")
        processor = hetero_cone_deformation_processor.ConeDeformationProcessor(
            pts, tets, lbls,
            rho_np=rho_cells,
            alpha_np=alpha_cells,
            beta_np=beta_cells,
            kappa_np=kappa_cells,
            nu_value=None,
        )
        
        log_fn("Computing deformation analysis...")
        processor.set_displacement_field(displacement_vectors)
        
        log_fn("Saving preprocessed results...")
        processor.save_results(str(output_path))
        
        log_fn(f"Preprocessing completed: {output_path}.npz")
        return True
        
    except Exception as e:
        log_fn(f"ERROR during preprocessing: {e}")
        import traceback
        log_fn(traceback.format_exc())
        return False


# ------------------------------------------
# Main optimization loop
# ------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Two-region constant alpha inverse problem with L-BFGS optimizer")
    parser.add_argument(
        "--u-obs-source",
        type=str,
        default=None,
        help="Path to alternative preprocessed .npz file to use for u_obs generation (e.g., FEM results). "
             "If not provided, u_obs will be generated from ground truth in the main preprocessed file."
    )
    args = parser.parse_args()
    
    base_dir = Path(__file__).resolve().parent
    
    # Initial learning rate (reduced for better stability with L-BFGS)
    initial_lr = 0.05
    
    output_dir = base_dir / "results" / f"2region_constant_inv_reg_LBFGS_{initial_lr}"
    output_dir.mkdir(parents=True, exist_ok=True)

    log_lines: list[str] = []
    history_entries: list[dict[str, object]] = []

    def log(message: str) -> None:
        print(message)
        log_lines.append(message)

    def flush_history() -> None:
        if not log_lines and not history_entries:
            return
        if log_lines:
            log_path = output_dir / "optimization.log"
            with log_path.open("w", encoding="utf-8") as log_file:
                log_file.write("\n".join(log_lines))
                log_file.write("\n")
        if history_entries:
            history_path = output_dir / "optimization_history.jsonl"
            with history_path.open("w", encoding="utf-8") as history_file:
                for entry in history_entries:
                    json.dump(entry, history_file)
                    history_file.write("\n")

    log("TWO-REGION CONSTANT ALPHA INVERSE PROBLEM (L-BFGS Optimizer)")
    
    # Log command-line arguments
    if args.u_obs_source:
        log(f"Using alternative u_obs source: {args.u_obs_source}")
    else:
        log("Using default u_obs generation from ground truth")
    
    # Check if preprocessing data exists, if not run the full pipeline
    pre = base_dir / "cone_verification_deformation.npz"
    if not pre.exists():
        log(f"Preprocessed data not found: {pre}")
        log("Running complete pipeline: FEM generation -> Preprocessing -> Optimization")
        
        # Step 1: FEM Generation
        if not run_fem_generation(base_dir, log):
            log("FEM generation failed. Exiting.")
            flush_history()
            return
        
        # Step 2: Preprocessing
        if not run_preprocessing(base_dir, log):
            log("Preprocessing failed. Exiting.")
            flush_history()
            return
        
        # Check if preprocessing succeeded
        if not pre.exists():
            log(f"Preprocessing did not create expected file: {pre}")
            flush_history()
            return
        
        log("\nPipeline preparation complete. Starting optimization...")
    else:
        log(f"Found preprocessed data: {pre}")
        log("Skipping FEM generation and preprocessing. Starting optimization...")

    log("\n=== STEP 3: Two-Region Constant Inverse Optimization ===")
    
    try:
        sim = ConeStaticEquilibrium(str(pre))
        total_mass_g = sim.get_total_mass_grams()
        log(f"Total nodal mass: {total_mass_g:.6f} g")
        log(f"Mesh: {sim.N} nodes, {sim.M} elements")

        # Load ground truth data
        with np.load(pre, allow_pickle=True) as data:
            alpha_np_data = data["alpha_k"].astype(np.float64)
            beta_np_data = data["beta_k"].astype(np.float64)
            kappa_np_data = data["kappa_k"].astype(np.float64)
            labels_data = data["labels"]

        # Use the same label source as the Taichi kernel for consistency
        labels_np = sim.labels.to_numpy()
        
        # Verify consistency
        if not np.array_equal(labels_data, labels_np):
            log("WARNING: Labels from NPZ differ from Taichi field. Using Taichi field for consistency.")
            labels_data = labels_np
        
        # Build label masks robustly - don't assume labels are 0 and 1
        unique_labels = np.unique(labels_data)
        log(f"Unique labels in mesh: {unique_labels}")
        
        if len(unique_labels) != 2:
            raise ValueError(f"Expected exactly 2 labels, got {len(unique_labels)}: {unique_labels}")
        
        # Background is the smaller label value, special is the larger
        bg_label_value = int(unique_labels.min())
        sp_label_value = int(unique_labels.max())
        
        log(f"Background label: {bg_label_value}, Special label: {sp_label_value}")
        
        # Create masks
        background_mask = labels_data == bg_label_value
        special_mask = labels_data == sp_label_value
        
        # Convert to torch for broadcasting (do this once, not in every iteration)
        labels_torch = torch.from_numpy(labels_data).to(device='cuda', dtype=torch.int32)
        is_bg_torch = labels_torch == bg_label_value
        is_sp_torch = labels_torch == sp_label_value
        
        # Sanity checks
        n_bg = int(is_bg_torch.sum().item())
        n_sp = int(is_sp_torch.sum().item())
        assert n_bg + n_sp == sim.M, f"Label masks don't cover all elements: {n_bg} + {n_sp} != {sim.M}"
        assert n_bg > 0 and n_sp > 0, f"Empty regions: bg={n_bg}, sp={n_sp}"
        log(f"Region sizes: background={n_bg} ({100*n_bg/sim.M:.1f}%), special={n_sp} ({100*n_sp/sim.M:.1f}%)")
        
        # Compute ground truth per-region means
        background_mask = labels_data == bg_label_value
        special_mask = labels_data == sp_label_value
        
        alpha_gt_bg = float(alpha_np_data[background_mask].mean())
        alpha_gt_sp = float(alpha_np_data[special_mask].mean())
        
        log(f"Ground truth alpha_bg: {alpha_gt_bg:.6e}")
        log(f"Ground truth alpha_sp: {alpha_gt_sp:.6e}")
        
        # Generate or load u_obs (observed displacement)
        if args.u_obs_source is not None:
            # Load u_obs from alternative source (e.g., FEM results)
            u_obs_source_path = Path(args.u_obs_source)
            if not u_obs_source_path.exists():
                raise FileNotFoundError(f"u_obs source file not found: {u_obs_source_path}")
            
            log(f"\nLoading u_obs from alternative source: {u_obs_source_path}")
            
            with np.load(u_obs_source_path, allow_pickle=True) as data:
                alpha_true = data["alpha_k"].astype(np.float64)
                beta_true = data["beta_k"].astype(np.float64)
                kappa_true = data["kappa_k"].astype(np.float64)
                
                alpha_true_torch = torch.from_numpy(alpha_true).to(device='cuda', dtype=torch.float64)
                beta_true_torch = torch.from_numpy(beta_true).to(device='cuda', dtype=torch.float64)
                kappa_true_torch = torch.from_numpy(kappa_true).to(device='cuda', dtype=torch.float64)
                
                sim.assemble_matrix(alpha_true_torch, beta_true_torch, kappa_true_torch)
                u_obs = sim.forward().to(dtype=torch.float64)
                log(f"Loaded material parameters from: {u_obs_source_path}")
                log(f"  alpha range: [{alpha_true.min():.6e}, {alpha_true.max():.6e}]")
                log(f"  beta range: [{beta_true.min():.6e}, {beta_true.max():.6e}]")
                log(f"  kappa range: [{kappa_true.min():.6e}, {kappa_true.max():.6e}]")
        else:
            # Default: Create ground truth field for observation from main preprocessed file
            log("\nGenerating u_obs from ground truth in main preprocessed file")
            alpha_gt = torch.from_numpy(alpha_np_data).cuda()
            beta_gt = torch.from_numpy(beta_np_data).cuda()
            kappa_gt = torch.from_numpy(kappa_np_data).cuda()

            sim.assemble_matrix(alpha_gt, beta_gt, kappa_gt)
            u_obs = sim.get_observed_free()
        
        mean_obs_disp_m = float(torch.mean(torch.abs(u_obs)).item())
        log(f"Mean observed displacement (free DOF): {mean_obs_disp_m:.6e} m")

        # Define initialization scenarios for sensitivity analysis
        # Format: (scenario_name, alpha_bg_multiplier, alpha_sp_multiplier, bg_reference, sp_reference)
        # bg_reference/sp_reference: 'bg' or 'sp' to specify which ground truth to use
        
        # For separate scenarios: each parameter has 6 choices relative to bg or sp ground truth
        # Choices: lower_bg (0.8*bg), higher_bg (1.2*bg), almost_bg (1.02*bg),
        #          lower_sp (0.8*sp), higher_sp (1.2*sp), almost_sp (1.02*sp)
        
        param_choices = [
            ("lower_bg", 0.8, "bg"),
            ("higher_bg", 1.2, "bg"),
            ("almost_bg", 1.02, "bg"),
            ("lower_sp", 0.8, "sp"),
            ("higher_sp", 1.2, "sp"),
            ("almost_sp", 1.02, "sp"),
        ]
        
        # Generate all 36 combinations for separate scenario (6 choices for bg Ã— 6 choices for sp)
        init_scenarios = []
        for bg_choice_name, bg_mult, bg_ref in param_choices:
            for sp_choice_name, sp_mult, sp_ref in param_choices:
                scenario_name = f"sep_{bg_choice_name}_{sp_choice_name}"
                init_scenarios.append((scenario_name, bg_mult, sp_mult, bg_ref, sp_ref))
        
        # Add same-value scenarios for comparison (both parameters initialized to same value)
        init_scenarios.extend([
            ("same_lower_bg", 0.8, 0.8, "bg", "bg"),  # both are 0.8 * gt_bg
            ("same_higher_bg", 1.2, 1.2, "bg", "bg"),  # both are 1.2 * gt_bg
            ("same_lower_sp", 0.8, 0.8, "sp", "sp"),  # both are 0.8 * gt_sp
            ("same_higher_sp", 1.2, 1.2, "sp", "sp"),  # both are 1.2 * gt_sp
        ])
        
        log(f"\n=== Running {len(init_scenarios)} initialization scenarios ===")
        
        # Store results from all scenarios
        all_scenarios_results = []
        
        for scenario_idx, (scenario_name, bg_mult, sp_mult, bg_ref, sp_ref) in enumerate(init_scenarios):
            log(f"\n--- Scenario {scenario_idx+1}/{len(init_scenarios)}: {scenario_name} ---")
            
            # Reset history for this scenario
            scenario_history = []
            
            # Initialize two scalar parameters based on scenario and reference
            # bg_ref and sp_ref can each be 'bg' or 'sp'
            if bg_ref == "bg":
                alpha_bg_init = torch.tensor([alpha_gt_bg * bg_mult], device='cuda', dtype=torch.float64)
            else:  # bg_ref == "sp"
                alpha_bg_init = torch.tensor([alpha_gt_sp * bg_mult], device='cuda', dtype=torch.float64)
            
            if sp_ref == "bg":
                alpha_sp_init = torch.tensor([alpha_gt_bg * sp_mult], device='cuda', dtype=torch.float64)
            else:  # sp_ref == "sp"
                alpha_sp_init = torch.tensor([alpha_gt_sp * sp_mult], device='cuda', dtype=torch.float64)
            
            log(f"Initial alpha_bg: {float(alpha_bg_init.item()):.6e} (multiplier: {bg_mult}, ref: {bg_ref})")
            log(f"Initial alpha_sp: {float(alpha_sp_init.item()):.6e} (multiplier: {sp_mult}, ref: {sp_ref})")
            
            # Use exp reparameterization: alpha = exp(theta) to ensure positivity
            # This guarantees alpha > 0 during line search without needing clamps
            theta_bg_init = torch.log(alpha_bg_init)
            theta_sp_init = torch.log(alpha_sp_init)
            
            # Optimize in log-space (theta), not alpha directly
            theta_bg = torch.nn.Parameter(theta_bg_init.clone())
            theta_sp = torch.nn.Parameter(theta_sp_init.clone())

            # Broadcast function: create per-element alpha from two scalars
            # Uses pre-computed masks (is_bg_torch, is_sp_torch) for consistency
            def broadcast_alpha(alpha_bg_scalar: torch.Tensor, alpha_sp_scalar: torch.Tensor) -> torch.Tensor:
                """Broadcast two scalars to element field based on labels.
                
                Uses pre-computed torch masks that are guaranteed to match the Taichi kernel's
                label interpretation. This avoids the lock-step gradient problem.
                """
                alpha_field = torch.empty(sim.M, device='cuda', dtype=torch.float64)
                alpha_field[is_bg_torch] = alpha_bg_scalar
                alpha_field[is_sp_torch] = alpha_sp_scalar
                return alpha_field

            # Save initial state (convert theta to alpha for saving)
            with torch.no_grad():
                alpha_bg_val = torch.exp(theta_bg)
                alpha_sp_val = torch.exp(theta_sp)
            
            alpha_init_field = broadcast_alpha(alpha_bg_val, alpha_sp_val)
            beta_init_field = compute_beta_from_alpha(alpha_init_field)
            kappa_init_field = compute_kappa_from_alpha(alpha_init_field)
            
            # Safety check: verify broadcast worked correctly
            with torch.no_grad():
                bg_values = alpha_init_field[is_bg_torch].unique()
                sp_values = alpha_init_field[is_sp_torch].unique()
                if len(bg_values) != 1 or len(sp_values) != 1:
                    raise RuntimeError(f"Broadcast failed: bg has {len(bg_values)} values, sp has {len(sp_values)} values")
                if not torch.allclose(bg_values[0], alpha_bg_val):
                    raise RuntimeError(f"Background broadcast mismatch: {bg_values[0]} != {alpha_bg_val.item()}")
                if not torch.allclose(sp_values[0], alpha_sp_val):
                    raise RuntimeError(f"Special broadcast mismatch: {sp_values[0]} != {alpha_sp_val.item()}")
                log(f"Broadcast verification passed: bg={bg_values[0].item():.4e}, sp={sp_values[0].item():.4e}")
            
            scenario_output_dir = output_dir / scenario_name
            scenario_output_dir.mkdir(parents=True, exist_ok=True)
            
            init_params_path = scenario_output_dir / "initial_params.xdmf"
            save_parameter_heatmap(sim, alpha_init_field, init_params_path, labels_data, log_fn=log)
            
            # Optimization setup with L-BFGS (optimize theta, not alpha)
            optimizer = torch.optim.LBFGS(
                [theta_bg, theta_sp], 
                lr=initial_lr,
                max_iter=20,
                line_search_fn='strong_wolfe'
            )

            # Gaussian smoothing parameters (following solver_torch.md section 7.5)
            # After each L-BFGS step, add perturbations: theta += varepsilon, varepsilon ~ N(0, sigma^2 I)
            # Then reset L-BFGS memory and decay sigma
            sigma = 1e-2  # Initial perturbation magnitude
            sigma_decay = 0.995  # Decay rate per iteration
            sigma_min = 1e-3  # Minimum threshold (stop perturbing when sigma < sigma_min)

            num_iterations = 1000
            
            scenario_history.append({
                "scenario": scenario_name,
                "iteration": 0,
                "phase": "init",
                "bg_reference": bg_ref,
                "sp_reference": sp_ref,
                "alpha_bg_init": float(alpha_bg_init.item()),
                "alpha_sp_init": float(alpha_sp_init.item()),
                "alpha_bg_multiplier": bg_mult,
                "alpha_sp_multiplier": sp_mult,
                "alpha_bg": float(alpha_bg_val.item()),
                "alpha_sp": float(alpha_sp_val.item()),
                "alpha_bg_gt": alpha_gt_bg,
                "alpha_sp_gt": alpha_gt_sp,
                "alpha_bg_error": float(abs(alpha_bg_val.item() - alpha_gt_bg)),
                "alpha_sp_error": float(abs(alpha_sp_val.item() - alpha_gt_sp)),
                "initial_learning_rate": initial_lr,
                "learning_rate": initial_lr,
            })
            
            # Storage for closure access
            closure_state = {
                'loss_total': None,
                'loss_data': None,
                'loss_reg': None,
                'grad_bg': None,
                'grad_sp': None,
                'fw_status': None,
                'bw_status': None,
            }
            
            for it in range(num_iterations):
                
                def closure():
                    """Closure function for L-BFGS optimizer with exp reparameterization.
                    
                    Reparameterization: alpha = exp(theta)
                    Chain rule: dL/dtheta = dL/dalpha * dalpha/dtheta = dL/dalpha * alpha
                    """
                    optimizer.zero_grad(set_to_none=True)
                    
                    # Convert theta to alpha: alpha = exp(theta)
                    # This ensures alpha > 0 even during line search
                    alpha_bg = torch.exp(theta_bg)
                    alpha_sp = torch.exp(theta_sp)
                    
                    # Broadcast scalars to field
                    alpha_field = broadcast_alpha(alpha_bg, alpha_sp)
                    beta_field = compute_beta_from_alpha(alpha_field)
                    kappa_field = compute_kappa_from_alpha(alpha_field)

                    # Forward and backward (computes dL/dalpha)
                    sim.assemble_matrix(alpha_field, beta_field, kappa_field)
                    sim.forward(tol=1e-6, max_iter=200)
                    fw_status = sim.get_last_forward_status()
                    
                    loss_total, loss_data, loss_reg, grad_alpha_bg, grad_alpha_sp = sim.backward(
                        u_obs, alpha_bg, alpha_sp, bg_label_value, tol=1e-6, max_iter=200
                    )
                    bw_status = sim.get_last_backward_status()
                    
                    # Apply chain rule: dL/dtheta = dL/dalpha * dalpha/dtheta
                    # Since alpha = exp(theta), we have dalpha/dtheta = exp(theta) = alpha
                    grad_theta_bg = grad_alpha_bg * alpha_bg
                    grad_theta_sp = grad_alpha_sp * alpha_sp
                    
                    # Assign gradients to theta parameters (not alpha)
                    theta_bg.grad = grad_theta_bg.unsqueeze(0) if grad_theta_bg.dim() == 0 else grad_theta_bg
                    theta_sp.grad = grad_theta_sp.unsqueeze(0) if grad_theta_sp.dim() == 0 else grad_theta_sp
                    
                    # Store for logging (closure may be called multiple times)
                    closure_state['loss_total'] = loss_total
                    closure_state['loss_data'] = loss_data
                    closure_state['loss_reg'] = loss_reg
                    closure_state['alpha_bg'] = alpha_bg
                    closure_state['alpha_sp'] = alpha_sp
                    closure_state['grad_alpha_bg'] = grad_alpha_bg
                    closure_state['grad_alpha_sp'] = grad_alpha_sp
                    closure_state['fw_status'] = fw_status
                    closure_state['bw_status'] = bw_status
                    
                    return loss_total
                
                # L-BFGS step (will call closure multiple times during line search)
                optimizer.step(closure)
                
                # Gaussian smoothing: add perturbations after L-BFGS step (solver_torch.md section 7.5)
                # This helps escape local minima and improves exploration
                if sigma >= sigma_min:
                    with torch.no_grad():
                        theta_bg.data += torch.randn_like(theta_bg) * sigma
                        theta_sp.data += torch.randn_like(theta_sp) * sigma
                    
                    # Reset L-BFGS memory after perturbation (important for convergence)
                    # Clear optimizer state to avoid using stale curvature information
                    # Must recreate optimizer to properly reset L-BFGS history
                    optimizer = torch.optim.LBFGS(
                        [theta_bg, theta_sp], 
                        lr=initial_lr,
                        max_iter=20,
                        line_search_fn='strong_wolfe'
                    )
                    
                    # Decay sigma for next iteration
                    sigma *= sigma_decay
                
                # No clamping needed - exp reparameterization guarantees alpha > 0
                
                # Recompute at final accepted parameters for accurate logging
                # (closure_state may reflect an intermediate line search evaluation)
                with torch.no_grad():
                    alpha_bg_final = torch.exp(theta_bg)
                    alpha_sp_final = torch.exp(theta_sp)
                    alpha_field_final = broadcast_alpha(alpha_bg_final, alpha_sp_final)
                    beta_field_final = compute_beta_from_alpha(alpha_field_final)
                    kappa_field_final = compute_kappa_from_alpha(alpha_field_final)
                
                # Final forward/backward for accurate loss/gradient logging
                sim.assemble_matrix(alpha_field_final, beta_field_final, kappa_field_final)
                sim.forward(tol=1e-6, max_iter=200)
                fw_status = sim.get_last_forward_status()
                
                loss_total, loss_data, loss_reg, grad_alpha_bg, grad_alpha_sp = sim.backward(
                    u_obs, alpha_bg_final, alpha_sp_final, bg_label_value, tol=1e-6, max_iter=200
                )
                bw_status = sim.get_last_backward_status()
                
                # First iteration: verify gradients are different (not lock-step)
                if it == 0:
                    grad_bg_val = float(grad_alpha_bg.item())
                    grad_sp_val = float(grad_alpha_sp.item())
                    grad_ratio = abs(grad_bg_val / (grad_sp_val + 1e-12))
                    log(f"  First iter gradients: grad_alpha_bg={grad_bg_val:.4e}, grad_alpha_sp={grad_sp_val:.4e}, ratio={grad_ratio:.4f}")
                    log(f"  Loss breakdown: total={float(loss_total.item()):.4e}, data={float(loss_data.item()):.4e}, reg={float(loss_reg.item()):.4e}")
                    if abs(grad_ratio - 1.0) < 0.01:
                        log(f"  WARNING: Gradients are nearly identical (ratio={grad_ratio:.4f}), may indicate label mismatch!")
                    else:
                        log(f"  Gradients are distinct, label splitting is working correctly.")

                # Compute errors
                with torch.no_grad():
                    err_bg = float(abs(alpha_bg_final.item() - alpha_gt_bg) / (abs(alpha_gt_bg) + 1e-6))
                    err_sp = float(abs(alpha_sp_final.item() - alpha_gt_sp) / (abs(alpha_gt_sp) + 1e-6))
                    abs_err_bg = float(abs(alpha_bg_final.item() - alpha_gt_bg))
                    abs_err_sp = float(abs(alpha_sp_final.item() - alpha_gt_sp))

                lr = float(optimizer.param_groups[0]["lr"])
                loss_total_value = float(loss_total.item())
                loss_data_value = float(loss_data.item())
                loss_reg_value = float(loss_reg.item())
                
                log(
                    f"Iter {it+1:03d}: loss={loss_total_value:.6e} (data={loss_data_value:.6e}, reg={loss_reg_value:.6e}), "
                    f"alpha_bg={float(alpha_bg_final.item()):.4e} (err={err_bg:.3e}), "
                    f"alpha_sp={float(alpha_sp_final.item()):.4e} (err={err_sp:.3e}), "
                    f"lr={lr:.2e}, sigma={sigma:.2e}, "
                    f"fw_conv={fw_status.get('converged', False)}, "
                    f"bw_conv={bw_status.get('converged', False)}"
                )
                
                scenario_history.append({
                    "scenario": scenario_name,
                    "iteration": it + 1,
                    "phase": "iterate",
                    "bg_reference": bg_ref,
                    "sp_reference": sp_ref,
                    "alpha_bg_init": float(alpha_bg_init.item()),
                    "alpha_sp_init": float(alpha_sp_init.item()),
                    "alpha_bg_multiplier": bg_mult,
                    "alpha_sp_multiplier": sp_mult,
                    "loss_total": loss_total_value,
                    "loss_data": loss_data_value,
                    "loss_reg": loss_reg_value,
                    "alpha_bg": float(alpha_bg_final.item()),
                    "alpha_sp": float(alpha_sp_final.item()),
                    "alpha_bg_gt": alpha_gt_bg,
                    "alpha_sp_gt": alpha_gt_sp,
                    "alpha_bg_rel_error": err_bg,
                    "alpha_sp_rel_error": err_sp,
                    "alpha_bg_abs_error": abs_err_bg,
                    "alpha_sp_abs_error": abs_err_sp,
                    "grad_alpha_bg": float(grad_alpha_bg.item()),
                    "grad_alpha_sp": float(grad_alpha_sp.item()),
                    "initial_learning_rate": initial_lr,
                    "learning_rate": lr,
                    "sigma": sigma,  # Gaussian smoothing perturbation magnitude
                    "forward_status": fw_status,
                    "backward_status": bw_status,
                })

            # Save final state (convert theta to alpha)
            with torch.no_grad():
                alpha_bg_final_save = torch.exp(theta_bg)
                alpha_sp_final_save = torch.exp(theta_sp)
                alpha_final_field = broadcast_alpha(alpha_bg_final_save, alpha_sp_final_save)
                beta_final_field = compute_beta_from_alpha(alpha_final_field)
                kappa_final_field = compute_kappa_from_alpha(alpha_final_field)
            
            final_params_path = scenario_output_dir / "final_params.xdmf"
            save_parameter_heatmap(sim, alpha_final_field, final_params_path, labels_data, log_fn=log)
            
            log(f"Scenario {scenario_name} optimization complete.")
            log(f"Final alpha_bg: {float(alpha_bg_final_save.item()):.6e} (GT: {alpha_gt_bg:.6e})")
            log(f"Final alpha_sp: {float(alpha_sp_final_save.item()):.6e} (GT: {alpha_gt_sp:.6e})")
            log(f"Final errors: bg={abs(alpha_bg_final_save.item() - alpha_gt_bg):.6e}, sp={abs(alpha_sp_final_save.item() - alpha_gt_sp):.6e}")
            
            scenario_history.append({
                "scenario": scenario_name,
                "iteration": num_iterations,
                "phase": "final",
                "bg_reference": bg_ref,
                "sp_reference": sp_ref,
                "alpha_bg_init": float(alpha_bg_init.item()),
                "alpha_sp_init": float(alpha_sp_init.item()),
                "alpha_bg_multiplier": bg_mult,
                "alpha_sp_multiplier": sp_mult,
                "alpha_bg": float(alpha_bg_final_save.item()),
                "alpha_sp": float(alpha_sp_final_save.item()),
                "alpha_bg_gt": alpha_gt_bg,
                "alpha_sp_gt": alpha_gt_sp,
                "alpha_bg_error": float(abs(alpha_bg_final_save.item() - alpha_gt_bg)),
                "alpha_sp_error": float(abs(alpha_sp_final_save.item() - alpha_gt_sp)),
                "initial_learning_rate": initial_lr,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            })
            
            # Save scenario history
            scenario_history_path = scenario_output_dir / "optimization_history.jsonl"
            with scenario_history_path.open("w", encoding="utf-8") as history_file:
                for entry in scenario_history:
                    json.dump(entry, history_file)
                    history_file.write("\n")
            
            # Collect scenario summary for comparison
            all_scenarios_results.append({
                "scenario": scenario_name,
                "bg_reference": bg_ref,
                "sp_reference": sp_ref,
                "alpha_bg_multiplier": bg_mult,
                "alpha_sp_multiplier": sp_mult,
                "alpha_bg_init": float(alpha_bg_init.item()),
                "alpha_sp_init": float(alpha_sp_init.item()),
                "alpha_bg_final": float(alpha_bg_final_save.item()),
                "alpha_sp_final": float(alpha_sp_final_save.item()),
                "alpha_bg_gt": alpha_gt_bg,
                "alpha_sp_gt": alpha_gt_sp,
                "alpha_bg_error": float(abs(alpha_bg_final_save.item() - alpha_gt_bg)),
                "alpha_sp_error": float(abs(alpha_sp_final_save.item() - alpha_gt_sp)),
                "alpha_bg_rel_error": float(abs(alpha_bg_final_save.item() - alpha_gt_bg) / (abs(alpha_gt_bg) + 1e-6)),
                "alpha_sp_rel_error": float(abs(alpha_sp_final_save.item() - alpha_gt_sp) / (abs(alpha_gt_sp) + 1e-6)),
            })
            
            # Add scenario results to main history
            history_entries.extend(scenario_history)
        
        # Save comparison results
        log("\n=== All Scenarios Complete ===")
        log("Summary of all scenarios:")
        for result in all_scenarios_results:
            log(f"  {result['scenario']}: "
                f"init=[{result['alpha_bg_init']:.4e}, {result['alpha_sp_init']:.4e}], "
                f"final=[{result['alpha_bg_final']:.4e}, {result['alpha_sp_final']:.4e}], "
                f"errors=[{result['alpha_bg_error']:.4e}, {result['alpha_sp_error']:.4e}]")
        
        # Save comparison summary
        comparison_path = output_dir / "scenarios_comparison.json"
        with comparison_path.open("w", encoding="utf-8") as f:
            json.dump(all_scenarios_results, f, indent=2)
        log(f"Saved scenarios comparison to: {comparison_path}")
        
    finally:
        flush_history()


if __name__ == "__main__":
    main()
