# SMS solver module (Second Moment of Stiffness).
# - Provides importable helpers (`initialize`, `forward`, `backward`) so other scripts can reuse the Taichi FEM core.
# - Parameterization now uses a global log-parameter (theta) plus per-tetra log offsets (delta_k) with alpha_k = exp(theta + delta_k),
#   while retaining the Charbonnier-TV adjoint gradients documented in doc/solver_torch.md §7.5.


import json
import math
import sys
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import taichi as ti
import numpy as np
import torch
import meshio
import cupy as cp
from torch.utils import dlpack
from cupyx.scipy.sparse import csr_matrix
from cupyx.scipy.sparse.linalg import spsolve

# Repository paths for intra-project imports
current_dir = Path(__file__).resolve().parent
repo_root = current_dir.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
project_root = repo_root / "lung_project_git"
if project_root.is_dir() and str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.sms_precompute_utils import (
    get_emory_example_paths,
    normalize_mesh_tag,
    run_sms_preprocessor,
    validate_preprocessed_states,
)


GRAMS_PER_KG = 1000.0

# Poisson's ratio for lung tissue (constant)
POISSON_RATIO = 0.4

# Regularization hyperparameters 
REG_WEIGHT = 5e-2   # lambda_F: weight for Charbonnier TV regularization on displacement gradient
EPS_REG    = 1e-4   # epsilon: Charbonnier smoothing parameter
EPS_DIV    = 1e-12  # Stabilizer for data loss denominator

# Physical box for alpha (Pa)
DEFAULT_ALPHA_MIN = 500.0
DEFAULT_ALPHA_MAX = 1.0e4
DEFAULT_MAX_DTHETA = 0.5
DEFAULT_MAX_DDELTA = 0.25

ALPHA_MIN = DEFAULT_ALPHA_MIN
ALPHA_MAX = DEFAULT_ALPHA_MAX
LOG_ALPHA_MIN = math.log(ALPHA_MIN)
LOG_ALPHA_MAX = math.log(ALPHA_MAX)
EPS_ACTIVE = 1e-6 * (LOG_ALPHA_MAX - LOG_ALPHA_MIN)  # relative tolerance near bounds

# Trust-region style caps for LBFGS steps in log-space
MAX_DTHETA = DEFAULT_MAX_DTHETA
MAX_DDELTA = DEFAULT_MAX_DDELTA

# Projection heuristics for optimizer resets
PROJECTION_REBUILD_THRESHOLD = 0.1
PROJECTION_ACTIVE_FRACTION = 0.05


def set_alpha_box(alpha_min: float, alpha_max: float) -> None:
    """Update the global alpha bounds and derived quantities."""
    if alpha_min <= 0.0:
        raise ValueError("alpha_min must be positive")
    if not alpha_min < alpha_max:
        raise ValueError("alpha_min must be strictly less than alpha_max")
    global ALPHA_MIN, ALPHA_MAX, LOG_ALPHA_MIN, LOG_ALPHA_MAX, EPS_ACTIVE
    ALPHA_MIN = float(alpha_min)
    ALPHA_MAX = float(alpha_max)
    LOG_ALPHA_MIN = math.log(ALPHA_MIN)
    LOG_ALPHA_MAX = math.log(ALPHA_MAX)
    EPS_ACTIVE = 1e-6 * (LOG_ALPHA_MAX - LOG_ALPHA_MIN)


def set_step_caps(max_dtheta: float, max_ddelta: float) -> None:
    """Update global trust-region style caps for log-parameter steps."""
    if max_dtheta <= 0.0 or max_ddelta <= 0.0:
        raise ValueError("Step caps must be positive")
    global MAX_DTHETA, MAX_DDELTA
    MAX_DTHETA = float(max_dtheta)
    MAX_DDELTA = float(max_ddelta)

def compute_beta_from_alpha(alpha: torch.Tensor) -> torch.Tensor:
    """Compute beta (shear modulus) from alpha via beta = 2*alpha."""
    return 2.0 * alpha

def compute_kappa_from_alpha(alpha: torch.Tensor, nu: float = POISSON_RATIO) -> torch.Tensor:
    """Compute kappa from alpha using the relationship kappa = lambda  = 4*alpha*nu/(1-2*nu)."""
    return 4.0 * alpha * nu / (1.0 - 2.0 * nu)


def clamp_log_alpha(theta_scalar: torch.Tensor, delta_field: torch.Tensor) -> torch.Tensor:
    """Clamp log-alpha in-place to guarantee physical bounds."""
    loga = theta_scalar + delta_field
    return torch.clamp(loga, min=LOG_ALPHA_MIN, max=LOG_ALPHA_MAX)


def broadcast_alpha(theta_scalar: torch.Tensor, delta_field: torch.Tensor) -> torch.Tensor:
    """Exponentiate bounded log-parameters to obtain per-tetra alpha values."""
    loga = clamp_log_alpha(theta_scalar, delta_field)
    return torch.exp(loga)


@dataclass
class ProjectionOutcome:
    updated: bool = False
    theta_correction: float = 0.0
    delta_correction: float = 0.0
    delta_frac_at_bound: float = 0.0


def project_log_parameters(
    theta_param: torch.nn.Parameter,
    delta_param: torch.nn.Parameter,
    prev_theta: torch.Tensor | None = None,
    prev_delta: torch.Tensor | None = None,
    mode: str = "both",
) -> ProjectionOutcome:
    """
    Project (theta, delta) onto the feasible log-alpha box and optionally
    clamp step sizes relative to provided previous copies.

    Args:
        theta_param: Global scalar parameter.
        delta_param: Per-element field parameter.
        prev_theta: Optional previous copy used for trust-region caps.
        prev_delta: Optional previous copy used for trust-region caps.
        mode: Which parameter to project. One of {"theta", "delta", "both"}.

    Returns:
        ProjectionOutcome summarizing whether an update occurred and its magnitude.
    """
    if mode not in {"theta", "delta", "both"}:
        raise ValueError(f"Unsupported projection mode: {mode}")

    outcome = ProjectionOutcome()
    with torch.no_grad():
        # Track theta caps when requested
        if mode in {"theta", "both"} and prev_theta is not None:
            theta_step = (theta_param - prev_theta).clamp(min=-MAX_DTHETA, max=MAX_DTHETA)
            theta_target = prev_theta + theta_step
            if not torch.allclose(theta_target, theta_param):
                diff = theta_target - theta_param
                outcome.theta_correction = max(
                    outcome.theta_correction,
                    float(torch.max(torch.abs(diff)).item()),
                )
                theta_param.copy_(theta_target)
                outcome.updated = True

        if mode in {"delta", "both"} and prev_delta is not None:
            delta_step = (delta_param - prev_delta).clamp(min=-MAX_DDELTA, max=MAX_DDELTA)
            delta_target = prev_delta + delta_step
            if not torch.allclose(delta_target, delta_param):
                diff = delta_target - delta_param
                outcome.delta_correction = max(
                    outcome.delta_correction,
                    float(torch.max(torch.abs(diff)).item()),
                )
                delta_param.copy_(delta_target)
                outcome.updated = True

        if mode in {"theta", "both"}:
            theta_low = LOG_ALPHA_MIN - torch.max(delta_param)
            theta_high = LOG_ALPHA_MAX - torch.min(delta_param)
            theta_clamped = theta_param.clamp(min=theta_low, max=theta_high)
            if not torch.allclose(theta_clamped, theta_param):
                diff = theta_clamped - theta_param
                outcome.theta_correction = max(
                    outcome.theta_correction,
                    float(torch.max(torch.abs(diff)).item()),
                )
                theta_param.copy_(theta_clamped)
                outcome.updated = True

        if mode in {"delta", "both"}:
            loga = theta_param + delta_param
            loga_clamped = torch.clamp(loga, min=LOG_ALPHA_MIN, max=LOG_ALPHA_MAX)
            bound_mask = loga_clamped.ne(loga)
            if bound_mask.any():
                delta_clamped = loga_clamped - theta_param
                diff = delta_clamped - delta_param
                outcome.delta_correction = max(
                    outcome.delta_correction,
                    float(torch.max(torch.abs(diff)).item()),
                )
                outcome.delta_frac_at_bound = max(
                    outcome.delta_frac_at_bound,
                    float(torch.count_nonzero(bound_mask).item()) / bound_mask.numel(),
                )
                delta_param.copy_(delta_clamped)
                outcome.updated = True

    return outcome


def ensure_preprocessed_file(args, log_fn: Callable[[str], None]) -> tuple[Path, dict, bool]:
    if args.preprocessed:
        pre = Path(args.preprocessed)
        if not pre.exists():
            raise FileNotFoundError(f"Specified preprocessed file not found: {pre}")
        validate_preprocessed_states(pre, args.subject, args.fixed_state, args.moving_state, log_fn)
        return pre, {}, False

    mesh_tag_raw = args.mesh_tag
    mesh_tag = normalize_mesh_tag(args.mask_name, mesh_tag_raw)

    paths = get_emory_example_paths(
        data_root=args.data_root,
        subject=args.subject,
        variant=args.variant,
        mask_name=args.mask_name,
        mesh_tag=mesh_tag,
        fixed_state=args.fixed_state,
        moving_state=args.moving_state,
    )

    cache_dir = Path(args.cache_dir) / args.subject
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_name = f"{args.subject}_{args.fixed_state}_to_{args.moving_state}_{args.mask_name}_{mesh_tag}"
    pre_npz = cache_dir / f"{output_name}.npz"
    preprocess_metadata = {
        "subject": args.subject,
        "fixed_state": args.fixed_state,
        "moving_state": args.moving_state,
        "mask_name": args.mask_name,
        "mesh_tag": mesh_tag_raw,
        "variant": args.variant,
    }

    needs_precompute = args.force_preprocess or not pre_npz.exists()
    if needs_precompute:
        mesh_path = Path(paths['fixed_mesh'])
        disp_path = Path(paths['disp_field'])
        log_fn(
            "Precomputing deformation data via utils/sms_precompute.py "
            "(uses deformation_processor_v2.py on CUDA)."
        )
        run_sms_preprocessor(mesh_path, disp_path, pre_npz, metadata=preprocess_metadata, log_fn=log_fn)
    else:
        log_fn(f"Using cached deformation data: {pre_npz}")

    validate_preprocessed_states(pre_npz, args.subject, args.fixed_state, args.moving_state, log_fn)
    return pre_npz, paths, needs_precompute


_TAICHI_INITIALIZED = False


def _ensure_taichi_initialized() -> None:
    """Lazily initialize Taichi so deformation preprocessing can run first if needed."""
    global _TAICHI_INITIALIZED
    if not _TAICHI_INITIALIZED:
        ti.init(arch=ti.cuda, debug=False, kernel_profiler=False, log_level=ti.ERROR)
        _TAICHI_INITIALIZED = True


def _torch_tensor_to_cupy(t: torch.Tensor) -> cp.ndarray:
    """Zero-copy Torch CUDA tensor -> CuPy array via DLPack."""
    if not isinstance(t, torch.Tensor):
        raise TypeError("Expected torch.Tensor for Torch->CuPy conversion")
    if not t.is_cuda:
        raise ValueError("Tensor for CuPy conversion must live on CUDA")
    return cp.from_dlpack(dlpack.to_dlpack(t))


def _torch_csr_to_cupy(A_sp: torch.Tensor) -> csr_matrix:
    """Zero-copy convert a Torch CSR tensor to a CuPy csr_matrix."""
    if A_sp.layout != torch.sparse_csr:
        raise ValueError("Input matrix must be a torch.sparse_csr_tensor")
    crow = _torch_tensor_to_cupy(A_sp.crow_indices())
    col = _torch_tensor_to_cupy(A_sp.col_indices())
    val = _torch_tensor_to_cupy(A_sp.values())
    return csr_matrix((val, col, crow), shape=A_sp.shape)


def torch_solve_sparse(
    A_sp: torch.Tensor,
    b: torch.Tensor,
    tol: float = 1e-4,
    max_iter: int = 2000,
):
    """
    Solve the linear system A x = b using CuPy's sparse direct solver.
    Returns `(solution, iterations_used, residual_norm, converged_flag, spd_ok)`.
    """

    _ = max_iter

    if not A_sp.is_cuda:
        raise ValueError("Sparse matrix must be a CUDA tensor")
    if A_sp.layout != torch.sparse_csr:
        raise ValueError("Sparse matrix must be in CSR layout")

    b = b.reshape(-1).to(device=A_sp.device, dtype=torch.float64)
    if b.numel() == 0:
        return torch.zeros_like(b), 0, 0.0, True, True

    A_sp = A_sp.to(torch.float64)

    # Zero-copy Torch -> CuPy conversion for CSR solve
    A_cu = _torch_csr_to_cupy(A_sp)
    b_cu = _torch_tensor_to_cupy(b)
    x_cu = spsolve(A_cu, b_cu)

    # Bring solution back to Torch without data movement
    x = torch.from_dlpack(x_cu)

    residual = torch.sparse.mm(A_sp, x.unsqueeze(1)).squeeze(1) - b
    res_norm = float(torch.linalg.norm(residual).item())
    rhs_norm = float(torch.linalg.norm(b).item())
    conv_tol = tol * max(1.0, rhs_norm)
    converged = res_norm <= conv_tol

    return x, 1, res_norm, bool(converged), True

# --------------------------------
# Data-oriented solver class
# --------------------------------
@ti.data_oriented
class ConeStaticEquilibrium:
    def __init__(self, preprocessed_data_path: str):
        _ensure_taichi_initialized()
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
        
        # Cache volume tensor on CUDA (avoid repeated CPU->GPU transfers in regularization)
        vol_np = self.vol.to_numpy().astype(np.float64)
        self._vol_torch = torch.from_numpy(vol_np).to(device='cuda', dtype=torch.float64)
        
        # Build gradient operator G for regularization
        self._build_gradient_operator()

    def _build_gradient_operator(self):
        """
        Build the gradient operator G in the doc that maps free DOFs to element-wise deformation gradients.
        G: R^{3N_f} -> R^{9M}
        For each element k, G_k = (B_k otimes I_3) where B_k contains shape function gradients.
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
        
        # Assemble triplets with taichi kernel
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
            D_m_inv_T = D_m.inverse().transpose()
            self.shape_grad[k, 1] = ti.Vector([D_m_inv_T[0, 0], D_m_inv_T[0, 1], D_m_inv_T[0, 2]])
            self.shape_grad[k, 2] = ti.Vector([D_m_inv_T[1, 0], D_m_inv_T[1, 1], D_m_inv_T[1, 2]])
            self.shape_grad[k, 3] = ti.Vector([D_m_inv_T[2, 0], D_m_inv_T[2, 1], D_m_inv_T[2, 2]])

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
            # For each element, build G_k = (B_k otimes I_3)
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
        # gradient field: grad_u = G @ u_free, shape (9M,)
        grad_u = torch.sparse.mm(self._G_operator, u_free.unsqueeze(1)).squeeze(1)  # (9M,)
        
        # Reshape to (M, 9) for per-element processing
        grad_u_elements = grad_u.view(self.M, 9)  # (M, 9)
        
        # Compute Frobenius norm squared for each element
        grad_norm_sq = torch.sum(grad_u_elements ** 2, dim=1)  # (M,)
        
        # Charbonnier function: sqrt(||nabla u||_F^2 + eps^2)
        charbonnier = torch.sqrt(grad_norm_sq + epsilon ** 2)  # (M,)
        
        # Regularization value: sum_k V_k * charbonnier_k 
        reg_value = torch.sum(self._vol_torch * charbonnier)
        
        # Gradient: partial R/partial u = G^T @ W @ (G @ u)
        # block-diagonal W: W_k = V_k / charbonnier_k * I_9
        weights = self._vol_torch / charbonnier  # (M,)
        
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
    # Gradient kernel: per-element gradients
    # ------------------------------------------
    @ti.kernel
    def _grad_alpha_kernel(self,
                           grad_alpha: ti.types.ndarray(dtype=ti.f64),
                           u_free:  ti.types.ndarray(dtype=ti.f64),
                           lam_free: ti.types.ndarray(dtype=ti.f64),
                           nu: ti.f64):
        coeff_beta = 2.0
        coeff_kappa = 4.0 * nu / (1.0 - 2.0 * nu)

        for k in range(self.M):
            V = self.vol[k]
            if V <= 1e-12:
                grad_alpha[k] = 0.0
                continue

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

            s_ax = 0.0
            for ell in ti.static(range(3)):
                r = self.r_axis[k, ell]
                su = 0.0
                sl = 0.0
                for q in ti.static(range(12)):
                    su += r[q] * u_loc[q]
                    sl += r[q] * lam_loc[q]
                s_ax += su * sl

            s_sh = 0.0
            for s in ti.static(range(3)):
                r = self.r_shear[k, s]
                su = 0.0
                sl = 0.0
                for q in ti.static(range(12)):
                    su += r[q] * u_loc[q]
                    sl += r[q] * lam_loc[q]
                s_sh += su * sl

            rv = self.r_vol[k]
            suv = 0.0
            slv = 0.0
            for q in ti.static(range(12)):
                suv += rv[q] * u_loc[q]
                slv += rv[q] * lam_loc[q]

            g_alpha = -4.0 * V * s_ax
            g_beta  = -4.0 * V * s_sh
            g_kappa = -1.0 * V * (suv * slv)

            grad_alpha[k] = g_alpha + coeff_beta * g_beta + coeff_kappa * g_kappa

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
                 reg_weight: float = REG_WEIGHT,
                 eps_reg: float = EPS_REG,
                 eps_div: float = EPS_DIV):
        """Backward pass: compute per-element gradients with Charbonnier-smoothed TV regularization."""

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

        grad_alpha = torch.zeros(self.M, device='cuda', dtype=torch.float64)

        self._grad_alpha_kernel(
            grad_alpha,
            u_star_free,
            lam_free,
            POISSON_RATIO
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

        return loss_total, loss_data, loss_reg, grad_alpha

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
# Main optimization loop
# ------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="SMS inverse problem (global scalar + local per-tetra field).\n"
                    "Requires Taichi CUDA build installed in the current environment."
    )
    parser.add_argument("--data-root", default="data/Emory-4DCT", help="Root directory for Emory 4DCT data")
    parser.add_argument("--subject", default="Case1Pack", help="Subject folder name, e.g., Case1Pack")
    parser.add_argument("--fixed-state", default="T00", help="Fixed respiratory state (e.g., T00)")
    parser.add_argument("--moving-state", default="T10", help="Moving respiratory state (e.g., T50)")
    parser.add_argument("--variant", default="NIFTI", help="Dataset variant (default: NIFTI)")
    parser.add_argument("--mask-name", default="lung_regions", help="Mask name used during preprocessing")
    parser.add_argument("--mesh-tag", default="lung_regions_11", help="Mesh tag suffix under pygalmesh/")
    parser.add_argument("--cache-dir", default="data_processed_deformation", help="Directory to store generated .npz files")
    parser.add_argument("--preprocessed", help="Path to existing SMS preprocessing .npz (skip generation)")
    parser.add_argument("--force-preprocess", action="store_true", help="Regenerate preprocessing even if cache exists")
    parser.add_argument("--alpha-min", type=float, default=DEFAULT_ALPHA_MIN, help="Lower physical bound for alpha (Pa)")
    parser.add_argument("--alpha-max", type=float, default=DEFAULT_ALPHA_MAX, help="Upper physical bound for alpha (Pa)")
    parser.add_argument("--max-dtheta", type=float, default=DEFAULT_MAX_DTHETA, help="LBFGS step cap for theta (log space)")
    parser.add_argument("--max-ddelta", type=float, default=DEFAULT_MAX_DDELTA, help="LBFGS step cap for delta field")
    parser.add_argument("--stage1-max-iters", type=int, default=1, help="Maximum number of Stage 1 (theta) LBFGS iterations")
    args = parser.parse_args()

    set_alpha_box(args.alpha_min, args.alpha_max)
    set_step_caps(args.max_dtheta, args.max_ddelta)
    
    stage1_max_iters = max(1, args.stage1_max_iters)
    base_dir = Path(__file__).resolve().parent
    
    # Initial learning rate 
    initial_lr = 1.0
    
    output_dir = base_dir / "results" / f"sms_global_local_{initial_lr}"
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

    log("SMS INVERSE PROBLEM (global scalar + local field)")
    log(f"Emory case: {args.subject} {args.fixed_state}->{args.moving_state} (variant={args.variant})")
    log(f"Mask: {args.mask_name}, mesh tag: {args.mesh_tag}")
    log(
        f"Alpha bounds: [{ALPHA_MIN:.3e}, {ALPHA_MAX:.3e}] Pa | "
        f"step caps: dtheta={MAX_DTHETA:.3e}, ddelta={MAX_DDELTA:.3e} | "
        f"Stage1 max iters: {stage1_max_iters}"
    )
    
    pre, example_paths, ran_precompute = ensure_preprocessed_file(args, log)
    
    log(f"Using preprocessed data: {pre}")
    if example_paths:
        log(f"  mesh: {example_paths.get('fixed_mesh')}")
        log(f"  displacement: {example_paths.get('disp_field')}")
    if ran_precompute:
        log("Finished deformation preprocessing (deformation_processor_v2.py); Taichi will initialize next.")

    log("\n=== STEP 3: SMS Optimization (global + local) ===")

    log("[Step] initializing solver")
    _ensure_taichi_initialized()
    sim = ConeStaticEquilibrium(str(pre))
    total_mass_g = sim.get_total_mass_grams()
    log(f"  total nodal mass: {total_mass_g:.6f} g")
    log(f"  mesh: {sim.N} nodes, {sim.M} elements")

    with np.load(pre, allow_pickle=True) as data:
        labels_data = data["labels"]
        alpha_np_data = data["alpha_k"].astype(np.float64) if "alpha_k" in data.files else None

    labels_np = sim.labels.to_numpy()
    
    labels_data = labels_np

    alpha_gt_torch = (
        torch.from_numpy(alpha_np_data).to(device='cuda', dtype=torch.float64)
        if alpha_np_data is not None else None
    )

    log("[Step] extracting observed displacement from preprocessing data")
    alpha_init_field = torch.ones(sim.M, device='cuda', dtype=torch.float64)
    beta_init_field = compute_beta_from_alpha(alpha_init_field)
    kappa_init_field = compute_kappa_from_alpha(alpha_init_field)
    log("[Step] assembling baseline stiffness matrix")
    sim.assemble_matrix(alpha_init_field, beta_init_field, kappa_init_field)
    log("[Step] solving baseline system for observed displacement")
    u_obs = sim.get_observed_free()
    mean_obs_disp_m = float(torch.mean(torch.abs(u_obs)).item())
    log(f"Mean observed displacement (free DOF): {mean_obs_disp_m:.6e} m")

    def record_history(scenario_output_dir: Path, scenario_history: list[dict[str, object]]):
        scenario_output_dir.mkdir(parents=True, exist_ok=True)
        scenario_history_path = scenario_output_dir / "optimization_history.jsonl"
        with scenario_history_path.open("w", encoding="utf-8") as history_file:
            for entry in scenario_history:
                json.dump(entry, history_file)
                history_file.write("\n")
        history_entries.extend(scenario_history)

    def run_alternating_optimization():
        scenario_global = "stage1_global"
        scenario_local = "stage2_local"
        scenario_global_dir = output_dir / scenario_global
        scenario_local_dir = output_dir / scenario_local
        history_global: list[dict[str, object]] = []
        history_local: list[dict[str, object]] = []

        theta_param = torch.nn.Parameter(torch.zeros(1, device='cuda', dtype=torch.float64))
        delta_param = torch.nn.Parameter(torch.zeros(sim.M, device='cuda', dtype=torch.float64))
        project_log_parameters(theta_param, delta_param)

        save_parameter_heatmap(
            sim,
            broadcast_alpha(theta_param.detach(), delta_param.detach()),
            scenario_global_dir / "initial_params.xdmf",
            labels_data,
            log_fn=log,
        )
        save_parameter_heatmap(
            sim,
            broadcast_alpha(theta_param.detach(), delta_param.detach()),
            scenario_local_dir / "initial_params.xdmf",
            labels_data,
            log_fn=log,
        )
        log(f"[Stage1] starting optimization (history dir: {scenario_global_dir})")
        log(f"[Stage2] starting optimization (history dir: {scenario_local_dir})")

        def _build_theta_optimizer():
            return torch.optim.LBFGS(
                [theta_param],
                lr=initial_lr,
                max_iter=20,
                line_search_fn='strong_wolfe'
            )

        def _build_delta_optimizer():
            return torch.optim.LBFGS(
                [delta_param],
                lr=initial_lr,
                max_iter=20,
                line_search_fn='strong_wolfe'
            )

        optimizer_theta = _build_theta_optimizer()
        optimizer_delta = _build_delta_optimizer()

        num_iterations = 200
        loss_nochange_tol = 1e-6
        param_update_tol = 1e-10
        grad_delta_tol = 1e-7

        global_prev_loss: float | None = None
        local_prev_loss: float | None = None
        global_nochange = 0
        local_nochange = 0
        global_converged = False
        local_converged = False
        global_iter = 0
        local_iter = 0

        global_closure_state: dict[str, object] = {}
        local_closure_state: dict[str, object] = {}

        def global_closure():
            optimizer_theta.zero_grad(set_to_none=True)
            loga = clamp_log_alpha(theta_param, delta_param.detach())
            alpha_field = torch.exp(loga)
            beta_field = compute_beta_from_alpha(alpha_field)
            kappa_field = compute_kappa_from_alpha(alpha_field)

            log("[Stage1] assembling stiffness matrix (closure)")
            sim.assemble_matrix(alpha_field, beta_field, kappa_field)
            log("[Stage1] forward solve (closure)")
            sim.forward(tol=1e-6, max_iter=200)

            loss_total, loss_data, loss_reg, grad_alpha = sim.backward(
                u_obs, tol=1e-6, max_iter=200
            )

            g_log = grad_alpha * alpha_field
            at_lower = loga <= (LOG_ALPHA_MIN + EPS_ACTIVE)
            at_upper = loga >= (LOG_ALPHA_MAX - EPS_ACTIVE)
            block_lower = at_lower & (g_log > 0)
            block_upper = at_upper & (g_log < 0)
            g_log = g_log.masked_fill(block_lower | block_upper, 0.0)
            total_elements = float(alpha_field.numel())
            lower_frac = float(torch.count_nonzero(at_lower).item()) / total_elements
            upper_frac = float(torch.count_nonzero(at_upper).item()) / total_elements

            grad_theta = g_log.sum()
            theta_param.grad = grad_theta.view_as(theta_param)

            global_closure_state['loss_total'] = float(loss_total.item())
            global_closure_state['loss_data'] = float(loss_data.item())
            global_closure_state['loss_reg'] = float(loss_reg.item())
            global_closure_state['alpha_mean'] = float(alpha_field.mean().item())
            global_closure_state['alpha_std'] = float(alpha_field.std().item())
            global_closure_state['grad_norm'] = float(abs(grad_theta.item()))
            global_closure_state['active_lower_frac'] = lower_frac
            global_closure_state['active_upper_frac'] = upper_frac
            global_closure_state['fw_status'] = sim.get_last_forward_status()
            global_closure_state['bw_status'] = sim.get_last_backward_status()
            if alpha_gt_torch is not None:
                global_closure_state['alpha_mae'] = float(
                    torch.mean(torch.abs(alpha_field - alpha_gt_torch)).item()
                )
            else:
                global_closure_state['alpha_mae'] = None
            log(
                f"[Stage1][closure] loss={global_closure_state['loss_total']:.6e}, "
                f"loss_data={global_closure_state['loss_data']:.6e}, "
                f"loss_reg={global_closure_state['loss_reg']:.6e}, "
                f"theta={float(theta_param.item()):.6e}"
            )
            return loss_total

        def local_closure():
            optimizer_delta.zero_grad(set_to_none=True)
            loga = clamp_log_alpha(theta_param.detach(), delta_param)
            alpha_field = torch.exp(loga)
            beta_field = compute_beta_from_alpha(alpha_field)
            kappa_field = compute_kappa_from_alpha(alpha_field)

            log("[Stage2] assembling stiffness matrix (closure)")
            sim.assemble_matrix(alpha_field, beta_field, kappa_field)
            log("[Stage2] forward solve (closure)")
            sim.forward(tol=1e-6, max_iter=200)

            loss_total, loss_data, loss_reg, grad_alpha = sim.backward(
                u_obs, tol=1e-6, max_iter=200
            )

            g_log = grad_alpha * alpha_field
            at_lower = loga <= (LOG_ALPHA_MIN + EPS_ACTIVE)
            at_upper = loga >= (LOG_ALPHA_MAX - EPS_ACTIVE)
            block_lower = at_lower & (g_log > 0)
            block_upper = at_upper & (g_log < 0)
            g_log = g_log.masked_fill(block_lower | block_upper, 0.0)
            total_elements = float(alpha_field.numel())
            lower_frac = float(torch.count_nonzero(at_lower).item()) / total_elements
            upper_frac = float(torch.count_nonzero(at_upper).item()) / total_elements

            delta_param.grad = g_log.clone()

            local_closure_state['loss_total'] = float(loss_total.item())
            local_closure_state['loss_data'] = float(loss_data.item())
            local_closure_state['loss_reg'] = float(loss_reg.item())
            local_closure_state['alpha_mean'] = float(alpha_field.mean().item())
            local_closure_state['alpha_std'] = float(alpha_field.std().item())
            local_closure_state['grad_norm'] = float(g_log.abs().mean().item())
            local_closure_state['active_lower_frac'] = lower_frac
            local_closure_state['active_upper_frac'] = upper_frac
            local_closure_state['fw_status'] = sim.get_last_forward_status()
            local_closure_state['bw_status'] = sim.get_last_backward_status()
            if alpha_gt_torch is not None:
                local_closure_state['alpha_mae'] = float(
                    torch.mean(torch.abs(alpha_field - alpha_gt_torch)).item()
                )
            else:
                local_closure_state['alpha_mae'] = None
            log(
                f"[Stage2][closure] loss={local_closure_state['loss_total']:.6e}, "
                f"loss_data={local_closure_state['loss_data']:.6e}, "
                f"loss_reg={local_closure_state['loss_reg']:.6e}"
            )
            return loss_total

            for _ in range(num_iterations):
                if not global_converged:
                    global_iter += 1
                    prev_theta = theta_param.detach().clone()
                    optimizer_theta.step(global_closure)
                    projection = project_log_parameters(
                        theta_param,
                        delta_param,
                        prev_theta=prev_theta,
                        mode="theta",
                    )
                    theta_update = float(torch.max(torch.abs(theta_param.detach() - prev_theta)).item())
                    if projection.updated:
                        optimizer_theta.state.clear()
                        optimizer_delta.state.clear()
                        if (
                            projection.theta_correction >= PROJECTION_REBUILD_THRESHOLD
                            or projection.delta_frac_at_bound >= PROJECTION_ACTIVE_FRACTION
                        ):
                            optimizer_theta = _build_theta_optimizer()

                    loss_value_raw = global_closure_state.get('loss_total')
                    if loss_value_raw is None:
                        log("[Stage1] Closure returned no loss; stopping.")
                        global_converged = True
                    else:
                        loss_value = float(loss_value_raw)
                        if global_prev_loss is not None:
                            if abs(global_prev_loss - loss_value) <= loss_nochange_tol * max(1.0, abs(global_prev_loss)):
                                global_nochange += 1
                            else:
                                global_nochange = 0
                        global_prev_loss = loss_value

                        alpha_field = broadcast_alpha(theta_param.detach(), delta_param.detach())
                        lr_theta = float(optimizer_theta.param_groups[0]["lr"])
                        theta_value = float(theta_param.detach().item())
                        theta_low = float((LOG_ALPHA_MIN - torch.max(delta_param.detach())).item())
                        theta_high = float((LOG_ALPHA_MAX - torch.min(delta_param.detach())).item())
                        log(
                            f"[Stage1] iter {global_iter:03d}: loss={loss_value:.6e}, "
                            f"alpha_mean={alpha_field.mean().item():.4e}, alpha_std={alpha_field.std().item():.4e}, "
                            f"grad_theta={global_closure_state['grad_norm']:.4e}, theta={theta_value:.6e}, "
                            f"update={theta_update:.3e}, lr={lr_theta:.2e}, "
                            f"theta_box=[{theta_low:.6e}, {theta_high:.6e}], "
                            f"active_lower={global_closure_state['active_lower_frac']:.2%}, "
                            f"active_upper={global_closure_state['active_upper_frac']:.2%}"
                        )
                        history_entry = {
                            "scenario": scenario_global,
                            "iteration": global_iter,
                            "phase": "iterate",
                            "loss_total": loss_value,
                            "loss_data": global_closure_state['loss_data'],
                            "loss_reg": global_closure_state['loss_reg'],
                            "alpha_mean": global_closure_state['alpha_mean'],
                            "alpha_std": global_closure_state['alpha_std'],
                            "grad_theta_norm": global_closure_state['grad_norm'],
                            "theta_value": theta_value,
                            "theta_box_low": theta_low,
                            "theta_box_high": theta_high,
                            "active_lower_frac": global_closure_state['active_lower_frac'],
                            "active_upper_frac": global_closure_state['active_upper_frac'],
                            "param_update": theta_update,
                            "learning_rate": lr_theta,
                            "forward_status": global_closure_state['fw_status'],
                            "backward_status": global_closure_state['bw_status'],
                        }
                        if global_closure_state['alpha_mae'] is not None:
                            history_entry['alpha_mae'] = global_closure_state['alpha_mae']
                        history_global.append(history_entry)

                        if global_nochange >= 2:
                            log("[Stage1] Loss stalled; stopping early.")
                            global_converged = True
                        elif theta_update <= param_update_tol:
                            log("[Stage1] Parameter update below tolerance; stopping early.")
                            global_converged = True
                        elif global_iter >= stage1_max_iters:
                            log(f"[Stage1] Reached iteration cap ({stage1_max_iters}); stopping.")
                            global_converged = True

                if not local_converged:
                    local_iter += 1
                    prev_delta = delta_param.detach().clone()
                    optimizer_delta.step(local_closure)
                    projection = project_log_parameters(
                        theta_param,
                        delta_param,
                        prev_delta=prev_delta,
                        mode="delta",
                    )
                    delta_update = float(torch.max(torch.abs(delta_param.detach() - prev_delta)).item())
                    if projection.updated:
                        optimizer_theta.state.clear()
                        optimizer_delta.state.clear()
                        if (
                            projection.delta_correction >= PROJECTION_REBUILD_THRESHOLD
                            or projection.delta_frac_at_bound >= PROJECTION_ACTIVE_FRACTION
                        ):
                            optimizer_delta = _build_delta_optimizer()

                    loss_value_raw = local_closure_state.get('loss_total')
                    if loss_value_raw is None:
                        log("[Stage2] Closure returned no loss; stopping.")
                        local_converged = True
                    else:
                        loss_value = float(loss_value_raw)
                        if local_prev_loss is not None:
                            if abs(local_prev_loss - loss_value) <= loss_nochange_tol * max(1.0, abs(local_prev_loss)):
                                local_nochange += 1
                            else:
                                local_nochange = 0
                        local_prev_loss = loss_value

                        alpha_field = broadcast_alpha(theta_param.detach(), delta_param.detach())
                        lr_delta = float(optimizer_delta.param_groups[0]["lr"])
                        delta_rms = float(torch.norm(delta_param.detach()).item() / math.sqrt(sim.M))
                        delta_mean = float(delta_param.detach().mean().item())
                        delta_std = float(delta_param.detach().std().item())
                        grad_delta_mean = float(local_closure_state['grad_norm'])
                        log(
                            f"[Stage2] iter {local_iter:03d}: loss={loss_value:.6e}, "
                            f"alpha_mean={alpha_field.mean().item():.4e}, alpha_std={alpha_field.std().item():.4e}, "
                            f"grad_delta_mean={grad_delta_mean:.4e}, delta_mean={delta_mean:.4e}, "
                            f"delta_std={delta_std:.4e}, delta_rms={delta_rms:.4e}, "
                            f"update={delta_update:.3e}, lr={lr_delta:.2e}, "
                            f"active_lower={local_closure_state['active_lower_frac']:.2%}, "
                            f"active_upper={local_closure_state['active_upper_frac']:.2%}"
                        )
                        history_entry = {
                            "scenario": scenario_local,
                            "iteration": local_iter,
                            "phase": "iterate",
                            "loss_total": loss_value,
                            "loss_data": local_closure_state['loss_data'],
                            "loss_reg": local_closure_state['loss_reg'],
                            "alpha_mean": local_closure_state['alpha_mean'],
                            "alpha_std": local_closure_state['alpha_std'],
                            "grad_delta_mean": grad_delta_mean,
                            "delta_rms": delta_rms,
                            "delta_mean": delta_mean,
                            "delta_std": delta_std,
                            "active_lower_frac": local_closure_state['active_lower_frac'],
                            "active_upper_frac": local_closure_state['active_upper_frac'],
                            "param_update": delta_update,
                            "learning_rate": lr_delta,
                            "forward_status": local_closure_state['fw_status'],
                            "backward_status": local_closure_state['bw_status'],
                        }
                        if local_closure_state['alpha_mae'] is not None:
                            history_entry['alpha_mae'] = local_closure_state['alpha_mae']
                        history_local.append(history_entry)

                        if grad_delta_mean < grad_delta_tol:
                            log(
                                f"[Stage2] Gradient mean {grad_delta_mean:.4e} below "
                                f"threshold {grad_delta_tol:.1e}; stopping."
                            )
                            local_converged = True
                        elif local_nochange >= 2:
                            log("[Stage2] Loss stalled; stopping early.")
                            local_converged = True
                        elif delta_update <= param_update_tol:
                            log("[Stage2] Parameter update below tolerance; stopping early.")
                            local_converged = True

                if global_converged and local_converged:
                    break

            theta_final = theta_param.detach().clone()
            delta_final = delta_param.detach().clone()
            alpha_final_field = broadcast_alpha(theta_final, delta_final)

            save_parameter_heatmap(
                sim,
                alpha_final_field,
                scenario_global_dir / "final_params.xdmf",
                labels_data,
                log_fn=log,
            )
            save_parameter_heatmap(
                sim,
                alpha_final_field,
                scenario_local_dir / "final_params.xdmf",
                labels_data,
                log_fn=log,
            )

            history_global.append({
                "scenario": scenario_global,
                "iteration": len(history_global) + 1,
                "phase": "final",
                "loss_total": global_closure_state.get('loss_total'),
                "alpha_mean": float(alpha_final_field.mean().item()),
                "alpha_std": float(alpha_final_field.std().item()),
                "theta_value": float(theta_final.item()),
            })
            history_local.append({
                "scenario": scenario_local,
                "iteration": len(history_local) + 1,
                "phase": "final",
                "loss_total": local_closure_state.get('loss_total'),
                "alpha_mean": float(alpha_final_field.mean().item()),
                "alpha_std": float(alpha_final_field.std().item()),
                "delta_rms": float(torch.norm(delta_final).item() / math.sqrt(sim.M)),
            })

            record_history(scenario_global_dir, history_global)
            record_history(scenario_local_dir, history_local)
            log("[Stage1] completed")
            log("[Stage2] completed")
            return theta_final, delta_final, alpha_final_field

    theta_final, delta_final, alpha_final_field = run_alternating_optimization()

    log("\nOptimization complete.")
    log(f"  Global log-param theta = {float(theta_final.item()):.6e}")
    log(f"  Local delta stats: mean={float(delta_final.mean().item()):.4e}, "
        f"std={float(delta_final.std().item()):.4e}")
    if alpha_gt_torch is not None:
        final_mae = float(torch.mean(torch.abs(alpha_final_field - alpha_gt_torch)).item())
        log(f"  Final |alpha - alpha_gt| mean: {final_mae:.6e}")

    flush_history()


if __name__ == "__main__":
    main()
