# Cone static equilibrium (SMS) with Taichi assembly + Torch solve.
# - Torch-based forward/adjoint interface
# - Gradients dL/dalpha, dL/dbeta, dL/dkappa via adjoint method
#
# Notes:
# 1) PyTorch's sparse direct solver requires a special build with CUDA cuDSS and is not supported in ROCm.
#    Therefore, we use torch.linalg.solve (direct dense solve) on CUDA instead.
# 2) K is assembled on GPU with taichi kernel into Torch COO triplets, combined, converted to CSR.
# 3) Solves use torch.linalg.solve (direct dense solve) on CUDA.
# 4) The "free DOF" system is solved; constrained DOFs are eliminated by mapping.
"""
Don't optimize three params
use E as param to optimized, transform to SMS params inside the script.
let alpha, use relationship to E, nu to get beta, kappa.
simple param with a constant alpha and see the optimization result.

"""

"""
easier condition:
just two constants for two regions.
"""

"""
mu = E/(2(1+nu)); lambda = E*nu/((1+nu)(1-2nu)). alpha=mu/2; beta=mu; kappa=lambda
Background (label 0): E=1.0e4 Pa, rho=200 kg/m³, nu=0.4 => mu≈3571.4286 Pa, lambda≈14285.7143 Pa => alpha≈1785.7143 Pa, beta≈3571.4286 Pa, kappa≈14285.7143 Pa.
Special (label 1): E=1.5e4 Pa, rho=200 kg/m³, nu=0.4 => mu≈5357.1429 Pa, lambda≈21428.5714 Pa => alpha≈2678.5714 Pa, beta≈5357.1429 Pa, kappa≈21428.5714 Pa.

units illustration in form of quantity, symbol, unit, value range, comments:
Length: L, m, Lung domain size 0.1-0.3
displacement: u, m, 0.001-0.01

Density: rho, kg/m³, 200, the exact value is not found. \
    In the pde script, it's a param. Here we use a typical value
Gravity: g, m/s², 9.81

Elastic modulus(Young's modulus): E, Pa(=N/m²), 0.1-100 kPa. Consistent with paper's\
    Here we use 10 kPa for background, 15 kPa for special.
lame parameters: {lambda; mu}, Pa(=N/m²), derived from E and nu.
SMS params: {alpha; beta; kappa}, Pa, derived from lambda and mu.

stress: sigma, Pa(=N/m²)

Density with gravity tells us the tissue's weight per tetra, which pulls everything downward.
    All the stiffness quantities—Young’s modulus and the Lamé/SMS parameters—
    are pressure-like numbers (in Pascals) that say how hard it is to compress or shear the lung. 
    Stress is the local pressure the tissue actually feels 
    For the same shape and loading, higher pressure-like stiffness means smaller displacements. 
"""

import json
import taichi as ti
import numpy as np
import torch
import meshio
from pathlib import Path
from typing import Optional, Callable


GRAMS_PER_KG = 1000.0

"""
the stiffness matrix K is always symmetric positive definite (SPD) 
for physically meaningful parameters (alpha, beta, kappa > 0).
It's is the Hessian of a quadratic potential energy.
Per-tet structure = R^T D R.

For tet $k$ it's stacked with the seven strain rows we already have
R_k = [r_ax,k,1; r_ax,k,2; r_ax,k,3; r_sh,k,1; r_sh,k,2; r_sh,k,3; r_v,k] in R^{7x12}


and assemble

K_k = V_K(4 alpha_k R_{ax,k}^T R_{Ax,k} + 4 beta_k R_{sh,k}^T R_{sh,k} + kappa_k R_{v,k}^T R_{v,k})
=V_k R_k^T D_k R_k in R^{12x12}

with a diagonal weight D_k=diag(4 alpha_k I_3, 4 beta_k I_3, kappa_k).\
 
The kernel computes the entrywise form of this (those s_ax, s_sh, s_v linear in each argument separately), 
so every block added is symmetric by construction.


"""


"""
# Debugging: manually compute one element's value of  dL/dalpha, dL/dbeta, dL/dkappa
nu=0.4
dof_map = self.dof_map.to_numpy()
tets = self.tets.to_numpy()
r_ax = self.r_axis.to_numpy()      # (M, 3, 12)
r_sh = self.r_shear.to_numpy()     # (M, 3, 12)
r_vol = self.r_vol.to_numpy()      # (M, 12)
vol = self.vol.to_numpy()          # (M,)
#vol.mean() #5.4248e-06
u_free_np = self._last_u_star.detach().cpu().numpy()
lam_free_np = lam_free.detach().cpu().numpy()

grad_E = np.zeros(self.M, dtype=np.float64)
inv_one_plus_nu = 1.0 / (1.0 + nu)
volumetric_coeff = nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
k=300
u_loc = np.zeros(12, dtype=np.float64)
lam_loc = np.zeros(12, dtype=np.float64)
for local_v, node in enumerate(tets[k]):
    base = 3 * node
    for d in range(3):
        gi = dof_map[base + d]
        idx = 3 * local_v + d
        if gi >= 0:
            u_loc[idx] = u_free_np[gi]
            lam_loc[idx] = lam_free_np[gi]

ax_u = r_ax[k] @ u_loc        # (3,)
ax_lam = r_ax[k] @ lam_loc
sh_u = r_sh[k] @ u_loc        # (3,)
sh_lam = r_sh[k] @ lam_loc
vol_u = np.dot(r_vol[k], u_loc)
vol_lam = np.dot(r_vol[k], lam_loc)
print(inv_one_plus_nu * (ax_u @ ax_lam + 2.0 * (sh_u @ sh_lam))
            + volumetric_coeff * (vol_u * vol_lam))

ax_u @ ax_lam
-0.0005621177273998926
sh_u @ sh_lam
-5.083599422701972e-06
vol_u * vol_lam
-2.1144451925554907e-05
sh_u
array([-0.00216337, -0.00021726, -0.00074254])
sh_lam
array([ 3.50495560e-03, -4.20629248e-05, -3.35303241e-03])
ax_lam
array([-0.01482676,  0.00590202, -0.00555085])
ax_u
array([ 0.0290982 , -0.01125531,  0.011576  ])
vol_u
0.006267009941194113
vol_lam
-0.0033739298523477457

(ax_u @ ax_lam)/(1500*vol[k])
-0.04933830899089157
dof_map = self.dof_map.to_numpy()
"""

"""
1. units in every steps.

relative loss instead of mse
loss is smaller 

"""

ti.init(arch=ti.cuda, debug=True, kernel_profiler=False)

def torch_solve_sparse(
    A_sp: torch.Tensor,
    b: torch.Tensor,
    tol: float = 1e-4,
    max_iter: int = 2000,
):
    """
    Solve the linear system A x = b using torch's sparse direct solver.

    `A_sp` is the CSR matrix assembled by `_assemble_K_torch` on CUDA.
    However, torch.sparse.spsolve calling linear solver with sparse tensors requires compiling PyTorch with CUDA cuDSS and is not supported in ROCm build.
    Instead, solve the linear system A x = b using torch.linalg API.
    Note torch just returns RuntimeError – if the A matrix is not invertible or any matrix in a batched A is not invertible.
    So we need to check the residual norm and convergence manually.
    Returns `(solution, iterations_used, residual_norm, converged_flag)`.
    """
    
    _ = max_iter  

    A = A_sp.to_dense().to(torch.float64)
    b = b.reshape(-1).to(torch.float64)
    if b.numel() == 0:
        return torch.zeros_like(b), 0, 0.0, True
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
    #   rows, cols: int64
    #   vals:       float32
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
            # We add each (row,col) entry once (duplicates across tets combine on Torch side).
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
    # Gradient kernel (adjoint formulas)
    # ------------------------------------------
    @ti.kernel
    def _grad_params_kernel(self,
                            grad_alpha: ti.types.ndarray(),   
                            grad_beta:  ti.types.ndarray(),   
                            grad_kappa: ti.types.ndarray(),   
                            u_free:     ti.types.ndarray(),   # (n_free,)
                            lam_free:   ti.types.ndarray()    # (n_free,)
                            ):
        for k in range(self.M):
            V = self.vol[k]
            if V <= 1e-12:
                grad_alpha[k] = 0.0
                grad_beta[k]  = 0.0
                grad_kappa[k] = 0.0
                continue
            
            # Gather local 12-DOF vectors (u^star, lambda) using dof_map
            u_loc   = ti.Vector.zero(ti.f64, 12)
            lam_loc = ti.Vector.zero(ti.f64, 12)
            for vi in ti.static(range(4)):
                node = self.tets[k][vi]
                for d in ti.static(range(3)):
                    dof_idx = 3 * node + d  # i32 * i32 -> i32
                    gi = ti.i32(self.dof_map[dof_idx])  # Ensure gi is i32
                    idx = 3*vi + d
                    if gi >= 0:  # Use >= 0 instead of != -1
                        u_loc[idx]   = u_free[gi]
                        lam_loc[idx] = lam_free[gi]
                    else:
                        u_loc[idx]   = 0.0
                        lam_loc[idx] = 0.0
                        
            # Axis sum
            s_ax = 0.0
            for ell in ti.static(range(3)):
                r = self.r_axis[k, ell]; su = 0.0; sl = 0.0
                for q in ti.static(range(12)):
                    su += r[q] * u_loc[q]
                    sl += r[q] * lam_loc[q]
                s_ax += su * sl
            
            # Shear sum
            s_sh = 0.0
            for s in ti.static(range(3)):
                r = self.r_shear[k, s]; su = 0.0; sl = 0.0
                for q in ti.static(range(12)):
                    su += r[q] * u_loc[q]
                    sl += r[q] * lam_loc[q]
                s_sh += su * sl
                
            # Volumetric
            rv = self.r_vol[k]; suv = 0.0; slv = 0.0
            for q in ti.static(range(12)):
                suv += rv[q] * u_loc[q]
                slv += rv[q] * lam_loc[q]
                
            # Gradients with formulas
            # dL/dalpha_k = -4 V_k sum_ell (r_ax u*)(r_ax lambda)
            # dL/dbeta_k  = -4 V_k sum_s   (r_sh u*)(r_sh lambda)
            # dL/dkappa_k = -1 V_k (r_vol u*)(r_vol lambda)
            grad_alpha[k] = -4.0 * V * s_ax
            grad_beta[k]  = -4.0 * V * s_sh
            grad_kappa[k] = -1.0 * V * (suv * slv)

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
                 tol: float = 1e-6,
                 max_iter: int = 200,
                 epsilon_rel: float = 1e-6):
        """Backward pass using the same matrix and latest forward solution.

        The loss is the global relative L2 error with a small epsilon scaled by
        ``epsilon_rel`` to stabilise the denominator when the observation is
        near zero.
        denorm = epsilon_rel^2 + ||u_obs||_2^2
        """

        u_obs_free = u_obs_free.contiguous()
        u_obs_free_m = u_obs_free.to(torch.float64)
        u_star_free = self._last_u_star
        
        obs_norm_sq = torch.dot(u_obs_free_m, u_obs_free_m)
        obs_norm = torch.sqrt(obs_norm_sq)
        eps_rel = max(float(epsilon_rel), 0.0)
        eps = eps_rel * obs_norm
        denom = obs_norm_sq + eps * eps
        tiny = torch.tensor(
            torch.finfo(obs_norm_sq.dtype).eps,
            device=obs_norm_sq.device,
            dtype=obs_norm_sq.dtype,
        )
        denom = torch.maximum(denom, tiny)

        diff = u_star_free - u_obs_free_m
        loss = torch.dot(diff, diff) / denom
        inverse_rhs = (2.0 / denom) * diff

        lam_free, iters, res, converged, spd_ok = torch_solve_sparse(
            self._K_sparse, inverse_rhs, tol=tol, max_iter=max_iter
        )

        grad_alpha = torch.zeros(self.M, device='cuda', dtype=torch.float64)
        grad_beta  = torch.zeros(self.M, device='cuda', dtype=torch.float64)
        grad_kappa = torch.zeros(self.M, device='cuda', dtype=torch.float64)

        self._grad_params_kernel(grad_alpha, grad_beta, grad_kappa, u_star_free, lam_free)
        
        self._last_backward_status = {
            "iterations": iters,
            "residual_norm": res,
            "converged": bool(converged),
            "tolerance": tol,
            "max_iter": max_iter,
            "spd": bool(spd_ok),
            "loss_denominator": float(denom.item()),
            "epsilon_rel": eps_rel,
        }

        return loss, grad_alpha, grad_beta, grad_kappa

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
    connectivity = sim.tets.to_numpy().astype(np.int32)  # Keep as int32 from Taichi

    cell_data = {"alpha": [alpha_np]}
    if labels_np is None:
        labels_np = sim.labels.to_numpy()
    cell_data["labels"] = [labels_np.astype(np.int32)]  # Keep as int32

    mesh = meshio.Mesh(points=points, cells=[("tetra", connectivity)], cell_data=cell_data)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    meshio.write(str(output_path), mesh, data_format="XML")
    log_fn(f"Saved parameter heatmap to {output_path}")


# ------------------------------------------
# Example usage
# ------------------------------------------
def main():
    base_dir = Path(__file__).resolve().parent
    history_dir = base_dir / "inv_his_records"

    log_lines: list[str] = []
    history_entries: list[dict[str, object]] = []

    def log(message: str) -> None:
        print(message)
        log_lines.append(message)

    def flush_history() -> None:
        if not log_lines and not history_entries:
            return
        history_dir.mkdir(parents=True, exist_ok=True)
        if log_lines:
            log_path = history_dir / "hetero_cone_inverse.log"
            with log_path.open("w", encoding="utf-8") as log_file:
                log_file.write("\n".join(log_lines))
                log_file.write("\n")
        if history_entries:
            history_path = history_dir / "hetero_cone_inverse_history.jsonl"
            with history_path.open("w", encoding="utf-8") as history_file:
                for entry in history_entries:
                    json.dump(entry, history_file)
                    history_file.write("\n")

    pre = base_dir / "cone_verification_deformation.npz"
    if not pre.exists():
        log(f"Missing preprocessed data: {pre}")

    try:
        if not pre.exists():
            return

        sim = ConeStaticEquilibrium(str(pre))
        total_mass_g = sim.get_total_mass_grams()
        log(f"Total nodal mass: {total_mass_g:.6f} g")

        xdmf_dir = base_dir / "xdmf_visualization"

        with np.load(pre, allow_pickle=True) as data:
            alpha_np_data = data["alpha_k"].astype(np.float64)
            beta_np_data = data["beta_k"].astype(np.float64)
            kappa_np_data = data["kappa_k"].astype(np.float64)
            labels_data = data["labels"]

        alpha_gt = torch.from_numpy(alpha_np_data).cuda()
        beta_gt = torch.from_numpy(beta_np_data).cuda()
        kappa_gt = torch.from_numpy(kappa_np_data).cuda()

        sim.assemble_matrix(alpha_gt, beta_gt, kappa_gt)
        u_obs = sim.get_observed_free()
        mean_obs_disp_m = float(torch.mean(torch.abs(u_obs)).item())
        log(f"Mean observed displacement (free DOF): {mean_obs_disp_m:.6e} m")

        background_mask = labels_data == 0
        low = 0.9 * float(alpha_np_data[background_mask].min())
        high = 1.1 * float(alpha_np_data[background_mask].max())
        alpha_init = low + (high - low) * torch.rand_like(alpha_gt)
        low = 0.9 * float(beta_np_data[background_mask].min())
        high = 1.1 * float(beta_np_data[background_mask].max())
        beta_init = low + (high - low) * torch.rand_like(beta_gt)
        low = 0.9 * float(kappa_np_data[background_mask].min())
        high = 1.1 * float(kappa_np_data[background_mask].max())
        kappa_init = low + (high - low) * torch.rand_like(kappa_gt)

        alpha_est = torch.nn.Parameter(alpha_init)
        beta_est = torch.nn.Parameter(beta_init)
        kappa_est = torch.nn.Parameter(kappa_init)

        alpha_initial_np = alpha_est.detach().cpu().numpy().astype(np.float64)
        beta_initial_np = beta_est.detach().cpu().numpy().astype(np.float64)
        kappa_initial_np = kappa_est.detach().cpu().numpy().astype(np.float64)

        labels_np = sim.labels.to_numpy()
        init_params_path = xdmf_dir / "cone_sms_inverse_initial_params.xdmf"
        final_params_path = xdmf_dir / "cone_sms_inverse_final_params.xdmf"
        parameter_change_path = history_dir / "hetero_cone_inverse_parameter_changes.npz"

        def _compute_region_stats(
            alpha_t: torch.Tensor,
            beta_t: torch.Tensor,
            kappa_t: torch.Tensor,
        ) -> dict[str, dict[str, float]]:
            alpha_np = alpha_t.detach().cpu().numpy()
            beta_np = beta_t.detach().cpu().numpy()
            kappa_np = kappa_t.detach().cpu().numpy()

            stats: dict[str, dict[str, float]] = {}
            for value, name in ((0, "background"), (1, "special")):
                mask = labels_np == value
                count = int(mask.sum())
                if count == 0:
                    stats[name] = {"count": 0}
                    continue

                a = alpha_np[mask]
                b = beta_np[mask]
                k = kappa_np[mask]

                stats[name] = {
                    "count": count,
                    "alpha_mean": float(a.mean()),
                    "alpha_min": float(a.min()),
                    "alpha_max": float(a.max()),
                    "beta_mean": float(b.mean()),
                    "kappa_mean": float(k.mean()),
                }
            return stats

        def _print_region_stats(
            tag: str,
            alpha_t: torch.Tensor,
            beta_t: torch.Tensor,
            kappa_t: torch.Tensor,
            log_fn: Callable[[str], None] = print,
        ) -> dict[str, dict[str, float]]:
            stats = _compute_region_stats(alpha_t, beta_t, kappa_t)
            for name, region_stats in stats.items():
                count = region_stats["count"]
                if count == 0:
                    log_fn(f"{tag} [{name}] count=0")
                    continue
                log_fn(
                    f"{tag} [{name}] count={count} "
                    f"alpha_mean={region_stats['alpha_mean']:.4e} "
                    f"alpha_min={region_stats['alpha_min']:.4e} "
                    f"alpha_max={region_stats['alpha_max']:.4e} "
                    f"beta_mean={region_stats['beta_mean']:.4e} "
                    f"kappa_mean={region_stats['kappa_mean']:.4e}"
                )
            return stats

        optimizer = torch.optim.Adam(
            [
                {"params": [alpha_est], "lr": 1},
                {"params": [beta_est], "lr": 1},
                {"params": [kappa_est], "lr": 1},
            ]
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[100, 200, 400, 800], gamma=0.5
        )

        init_region_stats = _print_region_stats("Init", alpha_est, beta_est, kappa_est, log_fn=log)
        save_parameter_heatmap(sim, alpha_est, init_params_path, labels_np, log_fn=log)
        mae_alpha = torch.mean(torch.abs(alpha_est - alpha_gt)).item()
        mae_beta = torch.mean(torch.abs(beta_est - beta_gt)).item()
        mae_kappa = torch.mean(torch.abs(kappa_est - kappa_gt)).item()
        log(f"MAE alpha={mae_alpha:.6e}, beta={mae_beta:.6e}, kappa={mae_kappa:.6e}")
        history_entries.append(
            {
                "iteration": 0,
                "phase": "init",
                "mae": {
                    "alpha": float(mae_alpha),
                    "beta": float(mae_beta),
                    "kappa": float(mae_kappa),
                },
                "region_stats": init_region_stats,
                "mean_observed_displacement_m": mean_obs_disp_m,
                "total_nodal_mass_g": total_mass_g,
            }
        )

        num_iterations = 1000
        for it in range(num_iterations):
            optimizer.zero_grad(set_to_none=True)

            sim.assemble_matrix(alpha_est, beta_est, kappa_est)
            sim.forward(tol=1e-6, max_iter=200)
            fw_status = sim.get_last_forward_status()
            loss, g_alpha, g_beta, g_kappa = sim.backward(u_obs, tol=1e-6, max_iter=200)
            bw_status = sim.get_last_backward_status()

            alpha_est.grad = g_alpha
            beta_est.grad = g_beta
            kappa_est.grad = g_kappa

            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                err_alpha = torch.mean(
                    torch.abs(alpha_est - alpha_gt) / (torch.abs(alpha_gt) + 1e-6)
                ).item()
                err_beta = torch.mean(
                    torch.abs(beta_est - beta_gt) / (torch.abs(beta_gt) + 1e-6)
                ).item()
                err_kappa = torch.mean(
                    torch.abs(kappa_est - kappa_gt) / (torch.abs(kappa_gt) + 1e-6)
                ).item()
                mae_alpha_iter = torch.mean(torch.abs(alpha_est - alpha_gt)).item()
                mae_beta_iter = torch.mean(torch.abs(beta_est - beta_gt)).item()
                mae_kappa_iter = torch.mean(torch.abs(kappa_est - kappa_gt)).item()

            lr_alpha = float(optimizer.param_groups[0]["lr"])
            lr_beta = float(optimizer.param_groups[1]["lr"])
            lr_kappa = float(optimizer.param_groups[2]["lr"])
            loss_value = float(loss.item())
            log(
                f"Iter {it+1:03d}: loss={loss_value:.6e}, "
                f"rel_err=(alpha {err_alpha:.3e}, beta {err_beta:.3e}, kappa {err_kappa:.3e}), "
                f"lr=(alpha {lr_alpha:.2e}, beta {lr_beta:.2e}, kappa {lr_kappa:.2e}), "
                f"forward_finished={fw_status.get('converged', False)}, "
                f"backward_finished={bw_status.get('converged', False)}, "
                f"forward_spd={fw_status.get('spd', False)}, "
                f"backward_spd={bw_status.get('spd', False)}"
            )
            iter_region_stats = _print_region_stats(
                f"Iter {it+1:03d}", alpha_est, beta_est, kappa_est, log_fn=log
            )
            history_entries.append(
                {
                    "iteration": it + 1,
                    "phase": "iterate",
                    "loss": loss_value,
                    "relative_error": {
                        "alpha": float(err_alpha),
                        "beta": float(err_beta),
                        "kappa": float(err_kappa),
                    },
                    "mae": {
                        "alpha": float(mae_alpha_iter),
                        "beta": float(mae_beta_iter),
                        "kappa": float(mae_kappa_iter),
                    },
                    "learning_rate": {
                        "alpha": lr_alpha,
                        "beta": lr_beta,
                        "kappa": lr_kappa,
                    },
                    "forward_status": fw_status,
                    "backward_status": bw_status,
                    "region_stats": iter_region_stats,
                }
            )

        with torch.no_grad():
            mae_alpha = torch.mean(torch.abs(alpha_est - alpha_gt)).item()
            mae_beta = torch.mean(torch.abs(beta_est - beta_gt)).item()
            mae_kappa = torch.mean(torch.abs(kappa_est - kappa_gt)).item()
            save_parameter_heatmap(sim, alpha_est, final_params_path, labels_np, log_fn=log)
        final_region_stats = _print_region_stats("Final", alpha_est, beta_est, kappa_est, log_fn=log)
        log("Optimization complete.")
        log(f"MAE alpha={mae_alpha:.6e}, beta={mae_beta:.6e}, kappa={mae_kappa:.6e}")

        alpha_final_np = alpha_est.detach().cpu().numpy().astype(np.float64)
        beta_final_np = beta_est.detach().cpu().numpy().astype(np.float64)
        kappa_final_np = kappa_est.detach().cpu().numpy().astype(np.float64)

        alpha_change_np = alpha_final_np - alpha_initial_np
        beta_change_np = beta_final_np - beta_initial_np
        kappa_change_np = kappa_final_np - kappa_initial_np

        def _summarize_change(values: np.ndarray) -> dict[str, float]:
            return {
                "min": float(values.min()),
                "max": float(values.max()),
                "mean": float(values.mean()),
                "abs_mean": float(np.mean(np.abs(values))),
                "std": float(values.std()),
            }

        change_summary = {
            "alpha": _summarize_change(alpha_change_np),
            "beta": _summarize_change(beta_change_np),
            "kappa": _summarize_change(kappa_change_np),
        }

        log("Parameter change summary (final - initial):")
        for name, stats in change_summary.items():
            log(
                f"  {name}: mean={stats['mean']:.4e}, abs_mean={stats['abs_mean']:.4e}, "
                f"min={stats['min']:.4e}, max={stats['max']:.4e}"
            )

        history_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            parameter_change_path,
            alpha_initial=alpha_initial_np,
            alpha_final=alpha_final_np,
            alpha_change=alpha_change_np,
            beta_initial=beta_initial_np,
            beta_final=beta_final_np,
            beta_change=beta_change_np,
            kappa_initial=kappa_initial_np,
            kappa_final=kappa_final_np,
            kappa_change=kappa_change_np,
            labels=labels_np.astype(np.int32),  # Keep as int32
        )
        log(f"Saved parameter change data to {parameter_change_path}")

        history_entries.append(
            {
                "iteration": num_iterations,
                "phase": "final",
                "mae": {
                    "alpha": float(mae_alpha),
                    "beta": float(mae_beta),
                    "kappa": float(mae_kappa),
                },
                "region_stats": final_region_stats,
                "parameter_change_summary": change_summary,
                "parameter_change_file": str(parameter_change_path),
            }
        )
    finally:
        flush_history()


if __name__ == "__main__":
    main()
