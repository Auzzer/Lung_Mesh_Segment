# Combined SMS solver module (Second Moment of Stiffness).
# - Unifies displacement-TV and alpha log-TV regularization.
# - Exposes a ConeStaticEquilibrium solver plus PyTorch autograd wrappers.

import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import cupy as cp
import meshio
import numpy as np
import taichi as ti
import torch
from cupyx.scipy.sparse import csr_matrix
from cupyx.scipy.sparse.linalg import spsolve
from torch.utils import dlpack

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

# ------------------------------------------------------------
# Physical constants and defaults
# ------------------------------------------------------------

GRAMS_PER_KG = 1000.0
POISSON_RATIO = 0.4

# Displacement-TV regularization defaults (omega_1)
U_REG_WEIGHT_DEFAULT = 5e-1
U_EPS_REG = 1e-4
U_EPS_DIV = 1e-12

# Alpha log-TV regularization defaults (omega_2)
ALPHA_REG_WEIGHT_DEFAULT = 5e-1
ALPHA_EPS_REG = 1e-4
ALPHA_EPS_DIV = 1e-12
ALPHA_DENOM_MIN = 5e2
ALPHA_DENOM_MAX = 1.2e4
ALPHA_ELEMENT_CAP = 25.0

# Physical alpha bounds and trust-region caps
ALPHA_MIN = 500.0
ALPHA_MAX = 1.0e4
LOG_ALPHA_MIN = math.log(ALPHA_MIN)
LOG_ALPHA_MAX = math.log(ALPHA_MAX)
LOG_ALPHA_MID = 0.5 * (LOG_ALPHA_MIN + LOG_ALPHA_MAX)
EPS_ACTIVE = 1e-6 * (LOG_ALPHA_MAX - LOG_ALPHA_MIN)
MAX_DTHETA = 0.5
MAX_DDELTA = 0.25

# Driver-only heuristics kept for compatibility
DELTA_JITTER_RANGE = 0.15
DELTA_SHRINK_WEIGHT = 5e-3
DELTA_PRECOND_EPS = 1e-12
DELTA_SHRINK_MIN_COUNT = 4
EARLY_DDELTA_CAP = 0.10
ACTIVE_UPPER_TIGHT_THRESHOLD = 0.35
ACTIVE_UPPER_RELAX_THRESHOLD = 0.15
MEAN_OBSERVED_DISP_WARNING = 1.0e-4

# Label-aware initialization ranges (Pa)
LABEL_PA_BANDS = {
    0: (3e3, 8e3),  # background fill
    1: (5.5e3, 8e3),  # RU lobe
    2: (4.5e3, 7e3),  # RM lobe
    3: (3.0e3, 6e3),  # RL lobe
    4: (5.5e3, 8e3),  # LU lobe
    5: (3.0e3, 6e3),  # LL lobe
    6: (8.0e3, 1.0e4),  # Airways
    7: (8.5e3, 1.0e4),  # Vessels
}
DEFAULT_LABEL_PA_BAND = (4.0e3, 7.5e3)
LOBE_LABELS = (1, 2, 3, 4, 5)
FIXED_LABELS = (0, 6, 7)


@dataclass
class ProjectionOutcome:
    updated: bool = False
    theta_correction: float = 0.0
    delta_correction: float = 0.0
    delta_frac_at_bound: float = 0.0


@dataclass
class SMSLossConfig:
    eps_div_data: float = U_EPS_DIV
    u_eps_reg: float = U_EPS_REG
    alpha_eps_reg: float = ALPHA_EPS_REG
    alpha_eps_div: float = ALPHA_EPS_DIV
    alpha_denom_min: float = ALPHA_DENOM_MIN
    alpha_denom_max: float = ALPHA_DENOM_MAX
    alpha_element_cap: float = ALPHA_ELEMENT_CAP
    forward_tol: float = 1e-6
    forward_max_iter: int = 200
    adjoint_tol: float = 1e-6
    adjoint_max_iter: int = 200


def _coerce_loss_config(config: SMSLossConfig | dict | None) -> SMSLossConfig:
    """Accept dataclass or plain dict and return a validated SMSLossConfig."""
    if config is None:
        return SMSLossConfig()
    if isinstance(config, SMSLossConfig):
        return config
    return SMSLossConfig(**dict(config))


# ------------------------------------------------------------
# Parameter utilities
# ------------------------------------------------------------

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
    """Compute kappa from alpha using the relationship kappa = lambda = 4*alpha*nu/(1-2*nu)."""
    return 4.0 * alpha * nu / (1.0 - 2.0 * nu)


def clamp_log_alpha_box(theta_scalar: torch.Tensor, delta_field: torch.Tensor) -> torch.Tensor:
    """Hard clamp log-alpha to keep stiffness physical."""
    loga = theta_scalar + delta_field
    return torch.clamp(loga, min=LOG_ALPHA_MIN, max=LOG_ALPHA_MAX)


def log_alpha_unclamped(theta_scalar: torch.Tensor, delta_field: torch.Tensor) -> torch.Tensor:
    """Return raw log-alpha without any physical box."""
    return theta_scalar + delta_field


def broadcast_alpha(theta_scalar: torch.Tensor, delta_field: torch.Tensor, clamp: bool = True) -> torch.Tensor:
    """Exponentiate log-parameters to obtain per-tetra alpha values.

    theta_scalar may be a scalar or a field broadcastable to delta_field.
    """
    loga = clamp_log_alpha_box(theta_scalar, delta_field) if clamp else log_alpha_unclamped(theta_scalar, delta_field)
    return torch.exp(loga)


# ------------------------------------------------------------
# Optimizer utilities (for drivers)
# ------------------------------------------------------------

def project_log_parameters(
    theta_param: torch.nn.Parameter,
    delta_param: torch.nn.Parameter,
    prev_theta: torch.Tensor | None = None,
    prev_delta: torch.Tensor | None = None,
    mode: str = "both",
    max_dtheta: float | None = None,
    max_ddelta: float | None = None,
) -> ProjectionOutcome:
    """
    Project (theta, delta) onto the feasible log-alpha box and optionally
    clamp step sizes relative to provided previous copies.
    """
    if mode not in {"theta", "delta", "both"}:
        raise ValueError(f"Unsupported projection mode: {mode}")

    if max_dtheta is None:
        max_dtheta = MAX_DTHETA
    if max_ddelta is None:
        max_ddelta = MAX_DDELTA

    outcome = ProjectionOutcome()
    with torch.no_grad():
        if mode in {"theta", "both"} and prev_theta is not None:
            theta_step = (theta_param - prev_theta).clamp(min=-max_dtheta, max=max_dtheta)
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
            delta_step = (delta_param - prev_delta).clamp(min=-max_ddelta, max=max_ddelta)
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


def project_lobe_parameters(
    theta_param: torch.nn.Parameter,
    delta_lobe_param: torch.nn.Parameter,
    label_band_lookup: dict[int, tuple[float, float]],
    prev_theta: torch.Tensor | None = None,
    prev_delta: torch.Tensor | None = None,
    max_dtheta: float | None = None,
    max_ddelta: float | None = None,
) -> None:
    """Clamp coarse per-lobe deltas inside both global and label-aware boxes."""
    if max_dtheta is None:
        max_dtheta = MAX_DTHETA
    if max_ddelta is None:
        max_ddelta = MAX_DDELTA

    with torch.no_grad():
        if prev_theta is not None:
            theta_step = (theta_param - prev_theta).clamp(min=-max_dtheta, max=max_dtheta)
            theta_param.copy_(prev_theta + theta_step)
        if prev_delta is not None:
            delta_step = (delta_lobe_param - prev_delta).clamp(min=-max_ddelta, max=max_ddelta)
            delta_lobe_param.copy_(prev_delta + delta_step)

        theta_value = float(theta_param.item())
        for idx, label in enumerate(LOBE_LABELS):
            log_low, log_high = label_band_lookup.get(label, (LOG_ALPHA_MIN, LOG_ALPHA_MAX))
            band_low = max(LOG_ALPHA_MIN, log_low)
            band_high = min(LOG_ALPHA_MAX, log_high)
            delta_low = band_low - theta_value
            delta_high = band_high - theta_value
            clamped = delta_lobe_param[idx].clamp(min=delta_low, max=delta_high)
            delta_lobe_param[idx].copy_(clamped)


def apply_step_limits(
    theta_param: torch.nn.Parameter,
    delta_param: torch.nn.Parameter,
    prev_theta: torch.Tensor | None = None,
    prev_delta: torch.Tensor | None = None,
    max_dtheta: float | None = None,
    max_ddelta: float | None = None,
) -> None:
    """Apply soft step size limits (no hard box)."""
    if max_dtheta is None:
        max_dtheta = MAX_DTHETA
    if max_ddelta is None:
        max_ddelta = MAX_DDELTA

    with torch.no_grad():
        if prev_theta is not None:
            theta_step = (theta_param - prev_theta).clamp(min=-max_dtheta, max=max_dtheta)
            theta_param.copy_(prev_theta + theta_step)

        if prev_delta is not None:
            delta_step = (delta_param - prev_delta).clamp(min=-max_ddelta, max=max_ddelta)
            delta_param.copy_(prev_delta + delta_step)


def apply_lobe_step_limits(
    theta_param: torch.nn.Parameter,
    delta_lobe_param: torch.nn.Parameter,
    prev_theta: torch.Tensor | None = None,
    prev_delta: torch.Tensor | None = None,
    max_dtheta: float | None = None,
    max_ddelta: float | None = None,
) -> None:
    """Apply step caps to theta and the per-lobe delta parameters."""
    if max_dtheta is None:
        max_dtheta = MAX_DTHETA
    if max_ddelta is None:
        max_ddelta = MAX_DDELTA

    with torch.no_grad():
        if prev_theta is not None:
            theta_step = (theta_param - prev_theta).clamp(min=-max_dtheta, max=max_dtheta)
            theta_param.copy_(prev_theta + theta_step)
        if prev_delta is not None:
            delta_step = (delta_lobe_param - prev_delta).clamp(min=-max_ddelta, max=max_ddelta)
            delta_lobe_param.copy_(prev_delta + delta_step)


# ------------------------------------------------------------
# Label-aware helpers
# ------------------------------------------------------------

def _label_log_band(label: int, shrink: float = 1.0) -> tuple[float, float]:
    pa_low, pa_high = LABEL_PA_BANDS.get(label, DEFAULT_LABEL_PA_BAND)
    pa_low = max(ALPHA_MIN * 1.05, min(ALPHA_MAX * 0.95, pa_low))
    pa_high = max(pa_low + 1.0, min(ALPHA_MAX * 0.99, pa_high))
    log_low = math.log(pa_low)
    log_high = math.log(pa_high)
    shrink = max(0.0, min(1.0, shrink))
    if shrink < 1.0:
        log_low = LOG_ALPHA_MID + (log_low - LOG_ALPHA_MID) * shrink
        log_high = LOG_ALPHA_MID + (log_high - LOG_ALPHA_MID) * shrink
    return log_low, log_high


def label_log_midpoint(label: int, shrink: float = 1.0) -> float:
    """Return the log-space midpoint for a label-aware Pa band."""
    log_low, log_high = _label_log_band(label, shrink=shrink)
    return 0.5 * (log_low + log_high)


def sample_log_alpha_by_label(labels: np.ndarray, rng: np.random.Generator, shrink: float = 1.0) -> np.ndarray:
    """Return per-tet log-alpha samples drawn from label-aware bands."""
    samples = np.empty_like(labels, dtype=np.float64)
    unique_labels = np.unique(labels)
    for lbl in unique_labels:
        band_low, band_high = _label_log_band(int(lbl), shrink=shrink)
        mask = labels == lbl
        n = int(mask.sum())
        if n == 0:
            continue
        draws = rng.uniform(low=band_low, high=band_high, size=n)
        jitter = rng.normal(scale=0.05 * (band_high - band_low), size=n)
        draws = np.clip(draws + jitter, band_low, band_high)
        samples[mask] = draws
    return samples


def compute_delta_shrinkage(
    delta_field: torch.Tensor,
    labels_tensor: torch.Tensor,
    weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute label-wise shrinkage penalty on delta_k so each lobe stays near its mean.

    Returns:
        shrink_loss: scalar tensor (float64)
        shrink_grad: per-element gradient contribution (float64)
    """
    if weight <= 0.0:
        zero = torch.zeros(1, dtype=delta_field.dtype, device=delta_field.device)
        return zero.squeeze(0), torch.zeros_like(delta_field)

    shrink_loss = torch.zeros(1, dtype=delta_field.dtype, device=delta_field.device).squeeze(0)
    shrink_grad = torch.zeros_like(delta_field)
    for lbl in LOBE_LABELS:
        mask = labels_tensor == lbl
        count = torch.count_nonzero(mask).item()
        if count < DELTA_SHRINK_MIN_COUNT:
            continue
        delta_subset = delta_field[mask]
        delta_mean = torch.mean(delta_subset)
        diff = delta_subset - delta_mean
        shrink_loss = shrink_loss + 0.5 * weight * torch.sum(diff * diff)
        shrink_grad[mask] = shrink_grad[mask] + weight * diff
    return shrink_loss, shrink_grad


def build_volume_preconditioner(vol_tensor: torch.Tensor) -> torch.Tensor:
    """Return diagonal preconditioner ~ 1 / volume with mean scaled to 1."""
    vol_tensor = vol_tensor.to(dtype=torch.float64)
    vol_mean = torch.mean(vol_tensor)
    denom = vol_tensor + DELTA_PRECOND_EPS
    precond = vol_mean / denom
    return precond


# ------------------------------------------------------------
# Preprocessing helper
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# Low-level solver helpers
# ------------------------------------------------------------

_TAICHI_INITIALIZED = False


def _ensure_taichi_initialized() -> None:
    """Lazily initialize Taichi so deformation preprocessing can run first if needed."""
    global _TAICHI_INITIALIZED
    if not _TAICHI_INITIALIZED:
        mem_frac = os.environ.get("TAICHI_MEM_FRACTION")
        mem_gb = os.environ.get("TAICHI_MEM_GB")
        kwargs = dict(arch=ti.cuda, debug=False, kernel_profiler=False, log_level=ti.ERROR)
        if mem_gb:
            try:
                kwargs["device_memory_GB"] = float(mem_gb)
            except ValueError:
                pass
        elif mem_frac:
            try:
                kwargs["device_memory_fraction"] = float(mem_frac)
            except ValueError:
                pass
        else:
            kwargs["device_memory_fraction"] = 0.5
        ti.init(**kwargs)
        _TAICHI_INITIALIZED = True


def reset_taichi() -> None:
    """Release Taichi runtime and allow re-init (helps avoid snode tree buildup)."""
    global _TAICHI_INITIALIZED
    ti.reset()
    _TAICHI_INITIALIZED = False


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

    b = b.reshape(-1).to(device=A_sp.device, dtype=torch.float64)
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


# ------------------------------------------------------------
# Data-oriented solver
# ------------------------------------------------------------


@ti.data_oriented
class ConeStaticEquilibrium:
    def __init__(self, preprocessed_data_path: str):
        _ensure_taichi_initialized()
        data = np.load(preprocessed_data_path, allow_pickle=True)

        # Mesh sizes
        self.N = int(data['mesh_points'].shape[0])   # nodes
        self.M = int(data['tetrahedra'].shape[0])    # tets

        # Core fields (Taichi f64 for numerical data, i32 for indices)
        self.x = ti.Vector.field(3, ti.f64, shape=self.N)       # positions
        self.tets = ti.Vector.field(4, ti.i32, shape=self.M)    # tet node ids (must be i32 for indexing)
        self.vol = ti.field(ti.f64, shape=self.M)               # per-tet volume
        self.mass = ti.field(ti.f64, shape=self.N)              # per-node mass
        self.labels = ti.field(ti.i32, shape=self.M)            # labels (i32 for indexing)

        # BCs
        self.boundary_nodes = ti.field(ti.i32, shape=self.N)    # i32 for indexing
        self.boundary_displacement = ti.Vector.field(3, ti.f64, shape=self.N)
        self.is_boundary_constrained = ti.field(ti.i32, shape=self.N)  # i32 for indexing

        # SMS rows: r in R^{1x12} (3 axis, 3 shear, 1 volumetric)
        self.r_axis = ti.Vector.field(12, ti.f64, shape=(self.M, 3))
        self.r_shear = ti.Vector.field(12, ti.f64, shape=(self.M, 3))
        self.r_vol = ti.Vector.field(12, ti.f64, shape=self.M)

        # Misc fields kept from script
        self.initial_positions = ti.Vector.field(3, ti.f64, shape=self.N)
        self.displacement_field = ti.Vector.field(3, ti.f64, shape=self.N)

        # Shared shape gradients for both regularizers
        self.shape_grad = ti.Vector.field(3, ti.f64, shape=(self.M, 4))

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

        # Precompute shape function gradients once for both regularizers
        self._compute_shape_gradients()

        # Build gradient operator G for displacement-TV
        self._build_gradient_operator()

        # Allocate alpha-regularization buffers
        self._alloc_alpha_reg_buffers()

    def _build_gradient_operator(self):
        """
        Build the gradient operator G that maps free DOFs to element-wise deformation gradients.
        G: R^{3N_f} -> R^{9M}
        For each element k, G_k = (B_k otimes I_3) where B_k contains shape function gradients.
        """
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
    def _assemble_G_triplets(
        self,
        rows: ti.types.ndarray(),
        cols: ti.types.ndarray(),
        vals: ti.types.ndarray(),
        cap: ti.i32,
    ):
        """Assemble gradient operator G as triplets in parallel."""
        self._nnz_counter[None] = 0

        for k in range(self.M):
            for a in ti.static(range(4)):  # node index
                node_id = self.tets[k][a]
                grad_N_a = self.shape_grad[k, a]

                for d in ti.static(range(3)):  # displacement component (x, y, z)
                    global_dof = 3 * node_id + d
                    free_dof = self.dof_map[global_dof]

                    if free_dof >= 0:
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
        Returns (reg_value, reg_grad_u).
        """
        grad_u = torch.sparse.mm(self._G_operator, u_free.unsqueeze(1)).squeeze(1)  # (9M,)
        grad_u_elements = grad_u.view(self.M, 9)  # (M, 9)
        grad_norm_sq = torch.sum(grad_u_elements ** 2, dim=1)  # (M,)
        charbonnier = torch.sqrt(grad_norm_sq + epsilon ** 2)  # (M,)

        reg_value = torch.sum(self._vol_torch * charbonnier)

        weights = self._vol_torch / charbonnier  # (M,)
        weights_expanded = weights.repeat_interleave(9)  # (9M,)
        weighted_grad = weights_expanded * grad_u  # (9M,)
        reg_grad = torch.sparse.mm(self._G_operator.t(), weighted_grad.unsqueeze(1)).squeeze(1)  # (n_free,)

        return reg_value, reg_grad

    # ----------------------------
    # Alpha regularization buffers
    # ----------------------------
    def _alloc_alpha_reg_buffers(self):
        """Allocate Taichi buffers used by the alpha regularization kernel."""
        self._alpha_elem_field = ti.field(dtype=ti.f64, shape=self.M)
        self._alpha_nodal_field = ti.field(dtype=ti.f64, shape=self.N)
        self._alpha_node_counts = ti.field(dtype=ti.f64, shape=self.N)
        self._alpha_nodal_grad = ti.field(dtype=ti.f64, shape=self.N)
        self._alpha_reg_grad_field = ti.field(dtype=ti.f64, shape=self.M)
        self._alpha_reg_value = ti.field(dtype=ti.f64, shape=())

    def compute_alpha_reg(
        self,
        alpha_field: torch.Tensor,
        epsilon_reg: float = 1e-4,
        epsilon_div: float = 1e-12,
        denom_min: float = ALPHA_DENOM_MIN,
        denom_max: float = ALPHA_DENOM_MAX,
        element_cap: float = ALPHA_ELEMENT_CAP,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute alpha field regularization entirely on the GPU using Taichi kernels.

        Args:
            alpha_field: Per-element alpha values (CUDA tensor, length M)
            epsilon_reg: Charbonnier smoothing parameter
            epsilon_div: Stabilizer for denominator
            denom_min/denom_max: clamp for average alpha inside each element
            element_cap: cap extremely large residuals
        """
        alpha_tensor = alpha_field.detach().to(dtype=torch.float64).contiguous()
        if alpha_tensor.shape[0] != self.M:
            raise ValueError(f"alpha_field must have length {self.M}, got {alpha_tensor.shape[0]}")
        if not alpha_tensor.is_cuda:
            raise ValueError("alpha_field must live on CUDA for Taichi regularizer")

        self._write_alpha_elements(alpha_tensor)
        self._reset_alpha_reg_fields()
        self._scatter_alpha_values()
        self._normalize_alpha_nodes()
        self._alpha_reg_kernel(epsilon_reg, epsilon_div, denom_min, denom_max, element_cap)
        self._gather_alpha_gradients()

        reg_value = torch.zeros(1, device=alpha_tensor.device, dtype=torch.float64)
        reg_grad = torch.empty_like(alpha_tensor)
        self._export_alpha_value(reg_value)
        self._export_alpha_grad(reg_grad)
        return reg_value.squeeze(0), reg_grad

    @ti.kernel
    def _reset_alpha_reg_fields(self):
        for i in range(self.N):
            self._alpha_nodal_field[i] = 0.0
            self._alpha_node_counts[i] = 0.0
            self._alpha_nodal_grad[i] = 0.0
        for k in range(self.M):
            self._alpha_reg_grad_field[k] = 0.0
        self._alpha_reg_value[None] = 0.0

    @ti.kernel
    def _scatter_alpha_values(self):
        for k in range(self.M):
            alpha_val = self._alpha_elem_field[k]
            for a in ti.static(range(4)):
                node_id = self.tets[k][a]
                ti.atomic_add(self._alpha_nodal_field[node_id], alpha_val)
                ti.atomic_add(self._alpha_node_counts[node_id], 1.0)

    @ti.kernel
    def _normalize_alpha_nodes(self):
        for i in range(self.N):
            count = self._alpha_node_counts[i]
            if count < 1.0:
                count = 1.0
            self._alpha_nodal_field[i] = self._alpha_nodal_field[i] / count

    @ti.kernel
    def _alpha_reg_kernel(
        self,
        epsilon_reg: ti.f64,
        epsilon_div: ti.f64,
        denom_min: ti.f64,
        denom_max: ti.f64,
        element_cap: ti.f64,
    ):
        for k in range(self.M):
            grad = ti.Vector([0.0, 0.0, 0.0])
            alpha_bar = 0.0
            for a in ti.static(range(4)):
                node_id = self.tets[k][a]
                alpha_a = self._alpha_nodal_field[node_id]
                grad_N_a = self.shape_grad[k, a]
                grad += alpha_a * grad_N_a
                alpha_bar += 0.25 * alpha_a

            alpha_eff = ti.max(ti.min(alpha_bar, denom_max), denom_min)
            denom = alpha_eff + 0.5 * epsilon_div
            grad_norm_sq = grad.dot(grad)
            sqrt_term = ti.sqrt(grad_norm_sq / (denom * denom) + epsilon_reg * epsilon_reg)
            sqrt_term = ti.min(sqrt_term, element_cap)
            reg_val = self.vol[k] * sqrt_term
            ti.atomic_add(self._alpha_reg_value[None], reg_val)

            if sqrt_term >= 1e-15:
                denom_sq = denom * denom
                denom_cu = denom_sq * denom
                for a in ti.static(range(4)):
                    node_id = self.tets[k][a]
                    grad_N_a = self.shape_grad[k, a]
                    gradient_term = grad.dot(grad_N_a) / (denom_sq * sqrt_term)
                    average_term = grad_norm_sq / (denom_cu * sqrt_term) * 0.25
                    node_contrib = self.vol[k] * (gradient_term - average_term)
                    ti.atomic_add(self._alpha_nodal_grad[node_id], node_contrib)

    @ti.kernel
    def _gather_alpha_gradients(self):
        for k in range(self.M):
            grad_val = 0.0
            for a in ti.static(range(4)):
                node_id = self.tets[k][a]
                count = self._alpha_node_counts[node_id]
                if count < 1.0:
                    count = 1.0
                grad_val += self._alpha_nodal_grad[node_id] / count
            self._alpha_reg_grad_field[k] = grad_val

    @ti.kernel
    def _write_alpha_elements(self, alpha: ti.types.ndarray(dtype=ti.f64, ndim=1)):
        for k in range(self.M):
            self._alpha_elem_field[k] = alpha[k]

    @ti.kernel
    def _export_alpha_grad(self, grad_out: ti.types.ndarray(dtype=ti.f64, ndim=1)):
        for k in range(self.M):
            grad_out[k] = self._alpha_reg_grad_field[k]

    @ti.kernel
    def _export_alpha_value(self, value_out: ti.types.ndarray(dtype=ti.f64, ndim=1)):
        value_out[0] = self._alpha_reg_value[None]

    @ti.kernel
    def _initialize_aux_vectors(self):
        for i in range(self.N):
            self.boundary_displacement[i] = ti.Vector([0.0, 0.0, 0.0])

    # ------------------------------------------
    # Build gravity RHS on free DOFs (Torch)
    # ------------------------------------------
    @ti.kernel
    def _build_rhs_free_into_torch(self, b_free: ti.types.ndarray()):
        g = ti.Vector([0.0, -9.81, 0.0])# foor lying down ct images
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
            rv = self.r_vol[k]

            # Each tet contributes a symmetric 12x12 block.
            for p in ti.static(range(12)):
                gp = g[p]
                for q in ti.static(range(12)):
                    gq = g[q]
                    if gp != -1 and gq != -1:
                        s_ax = ax0[p]*ax0[q] + ax1[p]*ax1[q] + ax2[p]*ax2[q]
                        s_sh = sh0[p]*sh0[q] + sh1[p]*sh1[q] + sh2[p]*sh2[q]
                        s_v = rv[p]*rv[q]
                        val = V * (4.0 * alpha_k * s_ax + 4.0 * beta_k * s_sh + kappa_k * s_v)

                        idx = ti.atomic_add(self._nnz_counter[None], 1)
                        if idx < cap:
                            rows[idx] = gp  # already i32, stored in i64 torch tensor
                            cols[idx] = gq  # already i32, stored in i64 torch tensor
                            vals[idx] = val

    def _assemble_K_torch(
        self,
        alpha_t: torch.Tensor,
        beta_t: torch.Tensor,
        kappa_t: torch.Tensor,
    ):
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

    def assemble_matrix(
        self,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        kappa: torch.Tensor,
    ):
        """Assemble and cache the stiffness matrix for the given parameters."""
        if alpha.numel() != self.M or beta.numel() != self.M or kappa.numel() != self.M:
            raise ValueError(f"alpha/beta/kappa must have length {self.M}")

        alpha = alpha.detach().to(device='cuda', dtype=torch.float64).contiguous()
        beta = beta.detach().to(device='cuda', dtype=torch.float64).contiguous()
        kappa = kappa.detach().to(device='cuda', dtype=torch.float64).contiguous()

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
        device = self._vol_torch.device if hasattr(self, "_vol_torch") else torch.device("cuda")
        return torch.from_numpy(free).to(device=device, dtype=torch.float64)

    def get_total_mass_grams(self) -> float:
        """Return the total nodal mass expressed in grams."""
        mass_np = self.mass.to_numpy()
        return float(mass_np.sum() * GRAMS_PER_KG)

    @ti.kernel
    def _scatter_free_to_node3(
        self,
        u_free: ti.types.ndarray(),
        out_node3: ti.types.ndarray(),
    ):
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
    def _grad_alpha_kernel(
        self,
        grad_alpha: ti.types.ndarray(dtype=ti.f64),
        u_free: ti.types.ndarray(dtype=ti.f64),
        lam_free: ti.types.ndarray(dtype=ti.f64),
        nu: ti.f64,
    ):
        coeff_beta = 2.0
        coeff_kappa = 4.0 * nu / (1.0 - 2.0 * nu)

        for k in range(self.M):
            V = self.vol[k]
            if V <= 1e-12:
                grad_alpha[k] = 0.0
                continue

            u_loc = ti.Vector.zero(ti.f64, 12)
            lam_loc = ti.Vector.zero(ti.f64, 12)
            for vi in ti.static(range(4)):
                node = self.tets[k][vi]
                for d in ti.static(range(3)):
                    dof_idx = 3 * node + d
                    gi = ti.i32(self.dof_map[dof_idx])
                    idx = 3*vi + d
                    if gi >= 0:
                        u_loc[idx] = u_free[gi]
                        lam_loc[idx] = lam_free[gi]
                    else:
                        u_loc[idx] = 0.0
                        lam_loc[idx] = 0.0

            s_ax = 0.0
            for a in ti.static(range(3)):
                r = self.r_axis[k, a]
                su = 0.0
                sl = 0.0
                for q in ti.static(range(12)):
                    su += r[q] * u_loc[q]
                    sl += r[q] * lam_loc[q]
                s_ax += su * sl

            s_sh = 0.0
            for a in ti.static(range(3)):
                r = self.r_shear[k, a]
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
            g_beta = -4.0 * V * s_sh
            g_kappa = -1.0 * V * (suv * slv)

            grad_alpha[k] = g_alpha + coeff_beta * g_beta + coeff_kappa * g_kappa

    # ============================================================
    # Public functions
    # ============================================================

    def forward(
        self,
        b_free: torch.Tensor | None = None,
        tol: float = 1e-6,
        max_iter: int = 200,
    ):
        """Forward solve using the matrix assembled via `assemble_matrix`."""

        n_free = int(self.n_free_dof)
        A_sp = self._K_sparse

        if b_free is None:
            if A_sp is None:
                raise RuntimeError("Stiffness matrix _K_sparse is not assembled; call assemble_matrix() first.")
            b_free = torch.empty(n_free, device=A_sp.device, dtype=torch.float64)
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

    def loss_and_grad(
        self,
        alpha_field: torch.Tensor,
        u_obs_free: torch.Tensor,
        omega_u: float,
        omega_alpha: float,
        config: SMSLossConfig | None = None,
        tol: float | None = None,
        max_iter: int | None = None,
        normalize_data_loss: bool = True,
    ):
        """
        Compute total loss and gradient wrt alpha for the combined objective:
            L = data + omega_u * R_u + omega_alpha * R_alpha
        """
        if self._K_sparse is None:
            raise RuntimeError("Stiffness matrix _K_sparse is not assembled; call assemble_matrix() first.")
        if self._last_u_star is None:
            raise RuntimeError("Forward solve has not been run; call forward() before loss_and_grad.")

        cfg = _coerce_loss_config(config)
        if alpha_field.numel() != self.M:
            raise ValueError(f"alpha_field must have length {self.M}")
        if u_obs_free.numel() != self.n_free_dof:
            raise ValueError(f"u_obs_free must have length {self.n_free_dof}")

        alpha_field = alpha_field.detach().to(dtype=torch.float64, device=self._K_sparse.device).contiguous()
        u_obs_free = u_obs_free.contiguous().to(dtype=torch.float64, device=self._K_sparse.device)
        u_star_free = self._last_u_star

        tol = cfg.adjoint_tol if tol is None else tol
        max_iter = cfg.adjoint_max_iter if max_iter is None else max_iter

        # Data loss
        obs_norm_sq = torch.dot(u_obs_free, u_obs_free)
        diff = u_star_free - u_obs_free
        denom = obs_norm_sq + torch.tensor(cfg.eps_div_data, device=u_obs_free.device, dtype=u_obs_free.dtype)
        if normalize_data_loss:
            loss_data = torch.dot(diff, diff) / denom
            grad_coeff = 2.0 / denom
        else:
            loss_data = torch.dot(diff, diff)
            grad_coeff = 2.0

        # Displacement-TV regularization
        reg_grad_u = torch.zeros_like(u_star_free)
        loss_u_reg = torch.zeros(1, device=u_star_free.device, dtype=u_star_free.dtype).squeeze(0)
        if omega_u != 0.0:
            loss_reg_raw, reg_grad_u = self.compute_charbonnier_regularization(u_star_free, epsilon=cfg.u_eps_reg)
            loss_u_reg = omega_u * loss_reg_raw

        # Alpha log-TV regularization
        reg_grad_alpha = torch.zeros(self.M, device=u_star_free.device, dtype=torch.float64)
        loss_alpha_reg = torch.zeros(1, device=u_star_free.device, dtype=torch.float64).squeeze(0)
        if omega_alpha != 0.0:
            loss_alpha_raw, reg_grad_alpha = self.compute_alpha_reg(
                alpha_field,
                epsilon_reg=cfg.alpha_eps_reg,
                epsilon_div=cfg.alpha_eps_div,
                denom_min=cfg.alpha_denom_min,
                denom_max=cfg.alpha_denom_max,
                element_cap=cfg.alpha_element_cap,
            )
            loss_alpha_reg = omega_alpha * loss_alpha_raw

        # Adjoint RHS: data term + displacement regularizer
        inverse_rhs = grad_coeff * diff + omega_u * reg_grad_u

        lam_free, iters, res, converged, spd_ok = torch_solve_sparse(
            self._K_sparse, inverse_rhs, tol=tol, max_iter=max_iter
        )

        # Gradient from PDE constraint
        grad_alpha_pde = torch.zeros(self.M, device=u_star_free.device, dtype=torch.float64)
        self._grad_alpha_kernel(
            grad_alpha_pde,
            u_star_free,
            lam_free,
            POISSON_RATIO
        )

        grad_alpha = grad_alpha_pde + omega_alpha * reg_grad_alpha

        loss_total = loss_data + loss_u_reg + loss_alpha_reg

        self._last_backward_status = {
            "iterations": iters,
            "residual_norm": res,
            "converged": bool(converged),
            "tolerance": tol,
            "max_iter": max_iter,
            "spd": bool(spd_ok),
            "loss_denominator": float(denom.item()),
            "loss_data": float(loss_data.item()),
            "loss_u_reg": float(loss_u_reg.item()),
            "loss_alpha_reg": float(loss_alpha_reg.item()),
            "loss_total": float(loss_total.item()),
            "omega_u": omega_u,
            "omega_alpha": omega_alpha,
            "u_eps_reg": cfg.u_eps_reg,
            "alpha_eps_reg": cfg.alpha_eps_reg,
            "alpha_eps_div": cfg.alpha_eps_div,
        }

        return loss_total, loss_data, loss_u_reg, loss_alpha_reg, grad_alpha


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


# ------------------------------------------------------------
# PyTorch autograd wrapper
# ------------------------------------------------------------


class SMSFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        theta: torch.Tensor,
        delta: torch.Tensor,
        sim: ConeStaticEquilibrium,
        u_obs_free: torch.Tensor,
        omega_u: float,
        omega_alpha: float,
        config: SMSLossConfig | None = None,
        normalize_data_loss: bool = True,
    ):
        cfg = _coerce_loss_config(config)

        # Ensure parameters live on CUDA float64 to match solver expectations
        delta = delta.to(dtype=torch.float64, device='cuda')
        theta = theta.to(dtype=torch.float64, device=delta.device)
        u_obs_free = u_obs_free.to(dtype=torch.float64, device=delta.device)

        if delta.numel() != sim.M:
            raise ValueError(f"delta must have length {sim.M}")
        if theta.numel() not in {1, delta.numel()}:
            raise ValueError("theta must be scalar or same length as delta")

        # Physical alpha for PDE assembly; clamp to keep stiffness stable
        alpha_field = broadcast_alpha(theta.detach(), delta.detach(), clamp=True)
        beta_field = compute_beta_from_alpha(alpha_field)
        kappa_field = compute_kappa_from_alpha(alpha_field)

        sim.assemble_matrix(alpha_field, beta_field, kappa_field)
        sim.forward(tol=cfg.forward_tol, max_iter=cfg.forward_max_iter)

        loss_total, loss_data, loss_u_reg, loss_alpha_reg, grad_alpha = sim.loss_and_grad(
            alpha_field,
            u_obs_free,
            omega_u=omega_u,
            omega_alpha=omega_alpha,
            config=cfg,
            tol=cfg.adjoint_tol,
            max_iter=cfg.adjoint_max_iter,
            normalize_data_loss=normalize_data_loss,
        )

        ctx.save_for_backward(alpha_field.detach(), grad_alpha.detach())
        ctx.theta_shape = theta.shape
        ctx.delta_shape = delta.shape

        # Extra outputs are logged only; gradients are carried by loss_total
        loss_data_det = loss_data.detach()
        loss_u_reg_det = loss_u_reg.detach()
        loss_alpha_reg_det = loss_alpha_reg.detach()
        ctx.mark_non_differentiable(loss_data_det, loss_u_reg_det, loss_alpha_reg_det)
        return loss_total, loss_data_det, loss_u_reg_det, loss_alpha_reg_det

    @staticmethod
    def backward(ctx, *grad_outputs):
        alpha_field, grad_alpha = ctx.saved_tensors

        grad_scale = grad_outputs[0] if len(grad_outputs) > 0 else None
        if grad_scale is None:
            grad_scale = torch.tensor(1.0, device=alpha_field.device, dtype=alpha_field.dtype)

        chain = grad_alpha * alpha_field * grad_scale
        if ctx.theta_shape.numel() == 1:
            g_theta = torch.sum(chain).reshape(ctx.theta_shape)
        else:
            g_theta = chain.reshape(ctx.theta_shape)
        g_delta = chain.reshape(ctx.delta_shape)

        # None for non-tensor inputs (sim, u_obs_free, omega_u, omega_alpha, config, normalize_data_loss)
        return g_theta, g_delta, None, None, None, None, None, None


class SMSLayer(torch.nn.Module):
    def __init__(
        self,
        sim: ConeStaticEquilibrium,
        u_obs_free: torch.Tensor | None = None,
        omega_u: float = U_REG_WEIGHT_DEFAULT,
        omega_alpha: float = ALPHA_REG_WEIGHT_DEFAULT,
        config: SMSLossConfig | None = None,
        return_components: bool = True,
        normalize_data_loss: bool = True,
    ) -> None:
        super().__init__()
        if u_obs_free is None:
            u_obs_free = sim.get_observed_free()
        self.sim = sim
        self.omega_u = omega_u
        self.omega_alpha = omega_alpha
        self.config = _coerce_loss_config(config)
        self.return_components = return_components
        self.normalize_data_loss = normalize_data_loss
        self.register_buffer("u_obs_free", u_obs_free.to(dtype=torch.float64, device='cuda'))

    def forward(self, theta: torch.Tensor, delta: torch.Tensor):
        loss_total, loss_data, loss_u_reg, loss_alpha_reg = SMSFunction.apply(
            theta,
            delta,
            self.sim,
            self.u_obs_free,
            self.omega_u,
            self.omega_alpha,
            self.config,
            self.normalize_data_loss,
        )
        if self.return_components:
            extras = {
                "loss_data": loss_data,
                "loss_u_reg": loss_u_reg,
                "loss_alpha_reg": loss_alpha_reg,
            }
            return loss_total, extras
        return loss_total
