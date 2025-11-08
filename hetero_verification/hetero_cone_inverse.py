# cone_sms_torch_only.py
#
# Cone static equilibrium (SMS) with Taichi assembly + Torch solve.
# - No SciPy
# - Torch-based forward/adjoint interface
# - Gradients dL/dalpha, dL/dbeta, dL/dkappa via adjoint method
#
# Notes:
# 1) We keep your Taichi fields (mesh, per-tet data, r-rows, BCs).
# 2) K is assembled on GPU into Torch COO triplets, coalesced, converted to CSR.
# 3) Solves use torch.sparse.spsolve (direct sparse solve) on CUDA.
# 4) The "free DOF" system is solved; constrained DOFs are eliminated by mapping.
# 5) Function name "foward()" uses the exact spelling you requested.\

"""
Background (label 0): E=1.0e4 Pa, rho=200 kg/m³, nu=0.4 → mu≈3571.4286, lambda≈14285.7143 → alpha≈1785.7143, beta≈3571.4286, kappa≈14285.7143.
Special (label 1): E=1.5e4 Pa, rho=200 kg/m³, nu=0.4 → mu≈5357.1429, lambda≈21428.5714 → alpha≈2678.5714, beta≈5357.1429, kappa≈21428.5714.
"""

import taichi as ti
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Optional

# ----------------------------
# Taichi init (CUDA expected)
# ----------------------------
ti.init(arch=ti.cuda, debug=True, kernel_profiler=False)

# ----------------------------
# Small numeric helpers (Torch)
# ----------------------------
def _cg_solve_sparse(
    A_sp: torch.Tensor,
<<<<<<< ours
    b: torch.Tensor,
    tol: float = 1e-6
=======
    b: torch.Tensor
>>>>>>> theirs
):
    """
    Solve the linear system A x = b using torch's sparse direct solver.

    `A_sp` is the CSR matrix assembled by `_assemble_K_torch` on CUDA.
    However, calling linear solver with sparse tensors requires compiling PyTorch with CUDA cuDSS and is not supported in ROCm build.
    Instead, solve the linear system A x = b using torch.linalg API.
    Returns `(solution, iterations_used, residual_norm, converged_flag)`.
    """
    
    A = A_sp.to_dense()
    b = b.reshape(-1)
    if b.numel() == 0:
        return torch.zeros_like(b), 0, 0.0, True
    x = torch.linalg.solve(A, b.unsqueeze(-1)).squeeze(-1)
<<<<<<< ours
    res = torch.norm(A @ x - b)
=======

>>>>>>> theirs
    converged = torch.allclose( A @ x, b)
    return x, 1, bool(converged)

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

        # Core fields (Taichi f32/i32 for compute)
        self.x        = ti.Vector.field(3, ti.f32, shape=self.N)       # positions
        self.tets     = ti.Vector.field(4, ti.i32, shape=self.M)       # tet node ids
        self.vol      = ti.field(ti.f32, shape=self.M)                 # per-tet volume
        self.mass     = ti.field(ti.f32, shape=self.N)                 # per-node mass
        self.labels   = ti.field(ti.i32, shape=self.M)

        # BCs
        self.boundary_nodes        = ti.field(ti.i32, shape=self.N)
        self.boundary_displacement = ti.Vector.field(3, ti.f32, shape=self.N)
        self.is_boundary_constrained = ti.field(ti.i32, shape=self.N)

        # SMS rows: r in R^{1x12} (3 axis, 3 shear, 1 volumetric)
        self.r_axis  = ti.Vector.field(12, ti.f32, shape=(self.M, 3))
        self.r_shear = ti.Vector.field(12, ti.f32, shape=(self.M, 3))
        self.r_vol   = ti.Vector.field(12, ti.f32, shape=self.M)

        # Misc fields kept from your script
        self.initial_positions  = ti.Vector.field(3, ti.f32, shape=self.N)
        self.displacement_field = ti.Vector.field(3, ti.f32, shape=self.N)

        # Internal buffers / mapping
        self._cur_k = ti.field(ti.i32, shape=())
        self._nnz_counter = ti.field(ti.i32, shape=())  # for triplet assembly

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
        self._last_forward_status: Optional[Dict[str, object]] = None
        self._last_backward_status: Optional[Dict[str, object]] = None

    # ----------------------------
    # Data loading 
    # ----------------------------
    def _load_preprocessed_data(self, data):
        self.x.from_numpy(data['mesh_points'].astype(np.float32))
        self.tets.from_numpy(data['tetrahedra'].astype(np.int32))
        self.vol.from_numpy(data['volume'].astype(np.float32))
        self.mass.from_numpy(data['mass'].astype(np.float32))

        self.boundary_nodes.from_numpy(data['boundary_nodes'].astype(np.int32))
        self.initial_positions.from_numpy(data['initial_positions'].astype(np.float32))
        self.displacement_field.from_numpy(data['displacement_field'].astype(np.float32))

        self.r_axis.from_numpy(data['r_axis'].astype(np.float32))
        self.r_shear.from_numpy(data['r_shear'].astype(np.float32))
        self.r_vol.from_numpy(data['r_vol'].astype(np.float32))
        if 'labels' not in data:
            raise ValueError("Preprocessed data missing 'labels'")
        self.labels.from_numpy(data['labels'].astype(np.int32))

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
        dof_map_np = -np.ones(3 * self.N, dtype=np.int32)
        cnt = 0
        is_bc = self.is_boundary_constrained.to_numpy()
        for i in range(self.N):
            if is_bc[i] == 0:
                for d in range(3):
                    dof_map_np[3 * i + d] = cnt
                    cnt += 1
        self.n_free_dof = int(cnt)
        self.dof_map = ti.field(dtype=ti.i32, shape=3*self.N)
        self.dof_map.from_numpy(dof_map_np)

    def _alloc_free_buffers(self):
        n = int(self.n_free_dof)
        self._b_free_ti = ti.field(dtype=ti.f32, shape=n)  # Taichi-side RHS if needed

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
    # Assemble K_FF as COO triplets into Torch
    #   rows, cols: int64
    #   vals:       float32
    # ------------------------------------------
    @ti.kernel
    def _assemble_triplets(
        self,
        rows: ti.types.ndarray(),    # len >= cap
        cols: ti.types.ndarray(),    
        vals: ti.types.ndarray(),    
        cap: ti.i32,              # maximum number of non-zero entries
        alpha: ti.types.ndarray(),   # len M
        beta:  ti.types.ndarray(),   
        kappa: ti.types.ndarray()    
    ):
        self._nnz_counter[None] = 0
        for k in range(self.M):
            V = self.vol[k]
            a = alpha[k]; b = beta[k]; c = kappa[k]
            # Global free-dof indices for this tet's 12 local dofs
            g = ti.Vector.zero(ti.i32, 12)
            for vi in ti.static(range(4)):
                node = self.tets[k][vi]
                for d in ti.static(range(3)):
                    g[3*vi + d] = self.dof_map[3*node + d]

            # Load r-rows once
            ax0 = self.r_axis[k, 0]; ax1 = self.r_axis[k, 1]; ax2 = self.r_axis[k, 2]
            sh0 = self.r_shear[k, 0]; sh1 = self.r_shear[k, 1]; sh2 = self.r_shear[k, 2]
            rv  = self.r_vol[k]

            # Each tet contributes a symmetric 12x12 block.
            # We add each (row,col) entry once (duplicates across tets coalesce on Torch side).
            for p in ti.static(range(12)):
                gp = g[p]
                for q in ti.static(range(12)):
                    gq = g[q]
                    if gp != -1 and gq != -1:
                        # Combine axis + shear + volumetric contributions
                        s_ax = ax0[p]*ax0[q] + ax1[p]*ax1[q] + ax2[p]*ax2[q]
                        s_sh = sh0[p]*sh0[q] + sh1[p]*sh1[q] + sh2[p]*sh2[q]
                        s_v  = rv[p]*rv[q]
                        val = V * (4.0 * a * s_ax + 4.0 * b * s_sh + c * s_v)

                        idx = ti.atomic_add(self._nnz_counter[None], 1)
                        if idx < cap:
                            rows[idx] = ti.cast(gp, ti.i64)
                            cols[idx] = ti.cast(gq, ti.i64)
                            vals[idx] = val
                    # else: silently drop overflow 

    def _assemble_K_torch(self,
                          alpha_t: torch.Tensor,
                          beta_t:  torch.Tensor,
                          kappa_t: torch.Tensor):
        """
        Assemble K_FF into a Torch CSR sparse tensor on CUDA.
        alpha_t, beta_t, kappa_t: shape (M,), float32, CUDA
        """
        assert alpha_t.is_cuda and beta_t.is_cuda and kappa_t.is_cuda
        assert alpha_t.dtype == torch.float32
        n_free = int(self.n_free_dof)

        # Conservative capacity: one 12x12 block per tet => 144 entries/tet.
        cap = int(144 * self.M)
        rows = torch.empty(cap, device='cuda', dtype=torch.int64)
        cols = torch.empty(cap, device='cuda', dtype=torch.int64)
        vals = torch.empty(cap, device='cuda', dtype=torch.float32)

        # Fill triplets from Taichi (zero-copy into Torch buffers)
        self._assemble_triplets(rows, cols, vals, cap, alpha_t, beta_t, kappa_t)

        # Fetch how many entries were written
        nnz = int(self._nnz_counter.to_numpy().item())  
        if nnz == 0:
            raise RuntimeError("Assembly produced zero non-zeros; check inputs/BCs")

        rows = rows[:nnz]
        cols = cols[:nnz]
        vals = vals[:nnz]

        # Build COO → coalesce duplicate (row,col) pairs across tets
        A_coo = torch.sparse_coo_tensor(
            torch.vstack([rows, cols]),
            vals,
            size=(n_free, n_free),
            device='cuda',
            dtype=torch.float32,
        ).coalesce()
        return A_coo.to_sparse_csr()

    def assemble_matrix(self,
                        alpha: torch.Tensor,
                        beta: torch.Tensor,
                        kappa: torch.Tensor):
        """Assemble and cache the stiffness matrix for the given parameters."""
        assert alpha.is_cuda and beta.is_cuda and kappa.is_cuda
        assert alpha.dtype == torch.float32 and beta.dtype == torch.float32 and kappa.dtype == torch.float32
        alpha = alpha.detach().contiguous()
        beta = beta.detach().contiguous()
        kappa = kappa.detach().contiguous()

        A_sp = self._assemble_K_torch(alpha, beta, kappa)
        self._K_sparse = A_sp
        self._K_sparse_shape = A_sp.shape
        self._last_u_star = None
        return A_sp

    def get_observed_free(self) -> torch.Tensor:
        """Convert the FEM displacement field to a free-DOF torch vector."""
        if self._K_sparse is None:
            raise RuntimeError("Call assemble_matrix(...) before requesting observed displacements.")

        disp_np = self.displacement_field.to_numpy().astype(np.float32)  # (N, 3)
        flat = disp_np.reshape(-1)
        dof_map_np = self.dof_map.to_numpy()  # (3N,)

        free = np.zeros(self.n_free_dof, dtype=np.float32)
        mask = dof_map_np >= 0
        free[dof_map_np[mask]] = flat[mask]

        return torch.from_numpy(free).to(self._K_sparse.device)

    # ------------------------------------------
    # Scatter helpers (for completeness)
    # ------------------------------------------
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
            u_loc   = ti.Vector.zero(ti.f32, 12)
            lam_loc = ti.Vector.zero(ti.f32, 12)
            for vi in ti.static(range(4)):
                node = self.tets[k][vi]
                for d in ti.static(range(3)):
                    gi = self.dof_map[3*node + d]
                    idx = 3*vi + d
                    if gi != -1:
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

    def foward(self,
               b_free: torch.Tensor | None = None,
               tol: float = 1e-6,
               max_iter: int = 200):
        """Forward solve using the matrix assembled via `assemble_matrix`."""
        if self._K_sparse is None:
            raise RuntimeError("Call assemble_matrix(alpha, beta, kappa) before foward().")

        n_free = int(self.n_free_dof)
        A_sp = self._K_sparse

        if b_free is None:
            b_free = torch.empty(n_free, device='cuda', dtype=torch.float32)
            self._build_rhs_free_into_torch(b_free)
        else:
            if not (b_free.is_cuda and b_free.dtype == torch.float32 and b_free.numel() == n_free):
                raise ValueError("b_free must be a CUDA float32 tensor with length n_free")

        u_star, iters, res, converged = _cg_solve_sparse(
            A_sp, b_free, tol=tol, max_iter=max_iter
        )
        self._last_u_star = u_star
        self._last_forward_status = {
            "iterations": iters,
            "residual_norm": res,
            "converged": bool(converged),
            "tolerance": tol,
            "max_iter": max_iter,
        }
        return u_star

    def backward(self,
                 u_obs_free: torch.Tensor,
                 tol: float = 1e-6,
                 max_iter: int = 200):
        """Backward pass using the same matrix and latest forward solution."""
        if self._K_sparse is None or self._last_u_star is None:
            raise RuntimeError("Call assemble_matrix(...) and foward() before backward().")

        if not (u_obs_free.is_cuda and u_obs_free.dtype == torch.float32):
            raise ValueError("u_obs_free must be a CUDA float32 tensor")

        u_obs_free = u_obs_free.contiguous()
        u_star_free = self._last_u_star
        if u_star_free.shape != u_obs_free.shape:
            raise ValueError("u_obs_free must match the shape of the last forward solution")

        diff = u_star_free - u_obs_free
        loss = torch.dot(diff, diff)
        rhs = 2.0 * diff

        lam_free, iters, res, converged = _cg_solve_sparse(
            self._K_sparse, rhs, tol=tol, max_iter=max_iter
        )

        grad_alpha = torch.zeros(self.M, device='cuda', dtype=torch.float32)
        grad_beta  = torch.zeros(self.M, device='cuda', dtype=torch.float32)
        grad_kappa = torch.zeros(self.M, device='cuda', dtype=torch.float32)

        self._grad_params_kernel(grad_alpha, grad_beta, grad_kappa, u_star_free, lam_free)

        self._last_backward_status = {
            "iterations": iters,
            "residual_norm": res,
            "converged": bool(converged),
            "tolerance": tol,
            "max_iter": max_iter,
        }

        return loss, grad_alpha, grad_beta, grad_kappa

    def get_last_forward_status(self) -> Optional[Dict[str, object]]:
        """Return metadata for the most recent forward torch solve."""
        return self._last_forward_status

    def get_last_backward_status(self) -> Optional[Dict[str, object]]:
        """Return metadata for the most recent backward torch solve."""
        return self._last_backward_status

# ------------------------------------------
# Example usage
# ------------------------------------------
def main():
    base_dir = Path(__file__).resolve().parent
    pre = base_dir / "cone_verification_deformation.npz"
    if not pre.exists():
        print(f"Missing preprocessed data: {pre}")
        return
    sim = ConeStaticEquilibrium(str(pre))

    data = np.load(pre, allow_pickle=True)
    alpha_gt = torch.from_numpy(data['alpha_k'].astype(np.float32)).cuda()
    beta_gt  = torch.from_numpy(data['beta_k'].astype(np.float32)).cuda()
    kappa_gt = torch.from_numpy(data['kappa_k'].astype(np.float32)).cuda()

    sim.assemble_matrix(alpha_gt, beta_gt, kappa_gt)
    u_obs = sim.get_observed_free()

    # Initialize within a large range relative to ground truth (element-wise)
    #low, high = 1, 10.0
    low=data['alpha_k'][data['labels']==0].min(); high=data['alpha_k'][data['labels']==1].max()
    alpha_init = (low + (high - low) * torch.rand_like(alpha_gt)) 
    low=data['beta_k'][data['labels']==0].min();  high=data['beta_k'][data['labels']==1].max()
    beta_init  = (low + (high - low) * torch.rand_like(beta_gt))  
    low=data['kappa_k'][data['labels']==0].min(); high=data['kappa_k'][data['labels']==1].max()
    kappa_init = (low + (high - low) * torch.rand_like(kappa_gt)) 
    
    """alpha_init = (alpha_gt.min() + (alpha_gt.max() - alpha_gt.min()) * torch.rand_like(alpha_gt))
    beta_init = (beta_gt.min() + (beta_gt.max() - beta_gt.min()) * torch.rand_like(beta_gt))
    kappa_init = (kappa_gt.min() + (kappa_gt.max() - kappa_gt.min()) * torch.rand_like(kappa_gt))"""
    
    alpha_est = torch.nn.Parameter(alpha_init)
    beta_est  = torch.nn.Parameter(beta_init)
    kappa_est = torch.nn.Parameter(kappa_init)

    labels_np = sim.labels.to_numpy()

    def _print_region_stats(tag: str, alpha_t: torch.Tensor, beta_t: torch.Tensor, kappa_t: torch.Tensor) -> None:
        alpha_np = alpha_t.detach().cpu().numpy()
        beta_np = beta_t.detach().cpu().numpy()
        kappa_np = kappa_t.detach().cpu().numpy()
        for value, name in ((0, "background"), (1, "special")):
            mask = labels_np == value
            count = int(mask.sum())
            if count == 0:
                print(f"{tag} [{name}] count=0")
                continue
            a = alpha_np[mask]
            b = beta_np[mask]
            k = kappa_np[mask]
            print(
                f"{tag} [{name}] count={count} "
                f"alpha_mean={a.mean():.4e} alpha_min={a.min():.4e} alpha_max={a.max():.4e} "
                f"beta_mean={b.mean():.4e} kappa_mean={k.mean():.4e}"
            )

    optimizer = torch.optim.Adam(
        [
            {"params": [alpha_est], "lr": 2.5e-2},
            {"params": [beta_est], "lr": 2.5e-2},
            {"params": [kappa_est], "lr": 5e-2},
        ]
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[200, 400, 600, 800], gamma=0.5
    )

    _print_region_stats("Init", alpha_est, beta_est, kappa_est)

    for it in range(100):
        optimizer.zero_grad(set_to_none=True)

        sim.assemble_matrix(alpha_est, beta_est, kappa_est)
        # Forward solve; cached result is used by backward for gradients
        sim.foward(tol=1e-6, max_iter=200)
        fw_status = sim.get_last_forward_status() or {}
        loss, g_alpha, g_beta, g_kappa = sim.backward(u_obs, tol=1e-6, max_iter=200)
        bw_status = sim.get_last_backward_status() or {}

        alpha_est.grad = g_alpha
        beta_est.grad = g_beta
        kappa_est.grad = g_kappa

        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            err_alpha = torch.mean(torch.abs(alpha_est - alpha_gt) / (torch.abs(alpha_gt) + 1e-6)).item()
            err_beta = torch.mean(torch.abs(beta_est - beta_gt) / (torch.abs(beta_gt) + 1e-6)).item()
            err_kappa = torch.mean(torch.abs(kappa_est - kappa_gt) / (torch.abs(kappa_gt) + 1e-6)).item()
        lr_alpha = optimizer.param_groups[0]['lr']
        lr_beta = optimizer.param_groups[1]['lr']
        lr_kappa = optimizer.param_groups[2]['lr']
        print(
            f"Iter {it+1:03d}: loss={float(loss.item()):.6e}, "
            f"rel_err=(alpha {err_alpha:.3e}, beta {err_beta:.3e}, kappa {err_kappa:.3e}), "
            f"lr=(alpha {lr_alpha:.2e}, beta {lr_beta:.2e}, kappa {lr_kappa:.2e}), "
            f"forward_finished={fw_status.get('converged', False)}, "
            f"backward_finished={bw_status.get('converged', False)}"
        )
        _print_region_stats(f"Iter {it+1:03d}", alpha_est, beta_est, kappa_est)

    with torch.no_grad():
        mae_alpha = torch.mean(torch.abs(alpha_est - alpha_gt)).item()
        mae_beta = torch.mean(torch.abs(beta_est - beta_gt)).item()
        mae_kappa = torch.mean(torch.abs(kappa_est - kappa_gt)).item()
    _print_region_stats("Final", alpha_est, beta_est, kappa_est)
    print("Optimization complete.")
    print(f"MAE alpha={mae_alpha:.6e}, beta={mae_beta:.6e}, kappa={mae_kappa:.6e}")

if __name__ == "__main__":
    main()
