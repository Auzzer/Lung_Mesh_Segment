#!/usr/bin/env python3
"""

For each pair (alpha_bg, alpha_sp) we:
  1. Assemble the stiffness matrix using constant values per region.
  2. Run a forward solve (no adjoint/backward).
  3. Measure the relative squared error against the observed displacement field:
         ||u_pred - u_obs||^2 / (||u_obs||^2 + eps_div)

The resulting grid is saved to an NPZ file to support downstream visualization.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

from hetero_cone_inverse import ConeStaticEquilibrium
from tqdm import tqdm



POISSON_RATIO = 0.4
EPS_DIV_DEFAULT = 1e-12


def compute_beta_from_alpha(alpha: torch.Tensor) -> torch.Tensor:
    """Derive beta from alpha using beta = 2 * alpha."""
    return 2.0 * alpha


def compute_kappa_from_alpha(alpha: torch.Tensor, nu: float = POISSON_RATIO) -> torch.Tensor:
    """Derive kappa from alpha using kappa = 4 * alpha * nu / (1 - 2 * nu)."""
    return 4.0 * alpha * nu / (1.0 - 2.0 * nu)


@dataclass(frozen=True)
class LandscapeConfig:
    preprocessed_path: Path
    output_path: Path
    checkpoint_path: Path
    alpha_bg_min: float
    alpha_bg_max: float
    alpha_bg_steps: int
    alpha_sp_factor_min: float
    alpha_sp_factor_max: float
    alpha_sp_factor_steps: int
    eps_div: float
    forward_tol: float
    forward_max_iter: int
    poisson_ratio: float


# ---------------------------------------------------------------------------
# User-configurable settings (edit these values as needed).
# ---------------------------------------------------------------------------
CONFIG = LandscapeConfig(
    preprocessed_path=Path("cone_verification_deformation.npz"),
    output_path=Path("results/loss_landscape/u_obs_sms_relative_loss_surface.npz"),
    checkpoint_path=Path("results/loss_landscape/u_obs_sms_relative_loss_surface.checkpoint.npz"),
    alpha_bg_min=1000.0,
    alpha_bg_max=3000.0,
    alpha_bg_steps=200,
    alpha_sp_factor_min=0.2,
    alpha_sp_factor_max=4.0,
    alpha_sp_factor_steps=200,
    eps_div=EPS_DIV_DEFAULT,
    forward_tol=1e-6,
    forward_max_iter=200,
    poisson_ratio=POISSON_RATIO,
)


def get_config() -> LandscapeConfig:
    """Validate and normalize the static configuration."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Taichi/Torch solver, but no CUDA device is available.")

    checkpoint_path = CONFIG.checkpoint_path
    if checkpoint_path is None or str(checkpoint_path).strip() == "":
        checkpoint_path = CONFIG.output_path.with_suffix(".checkpoint.npz")

    return replace(
        CONFIG,
        preprocessed_path=CONFIG.preprocessed_path.expanduser(),
        output_path=CONFIG.output_path.expanduser(),
        checkpoint_path=checkpoint_path.expanduser(),
    )


def load_observed_displacement(
    sim: ConeStaticEquilibrium,
    preprocessed_path: Path,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Assemble with ground-truth parameters and fetch the observed free displacement."""
    with np.load(preprocessed_path, allow_pickle=True) as data:
        alpha_np = data["alpha_k"].astype(np.float64)
        beta_np = data["beta_k"].astype(np.float64)
        kappa_np = data["kappa_k"].astype(np.float64)
        labels = data['labels']

    print('EXACT VALUES IN PREPROCESSED DATA')
    print('='*80)
    print()
    

    print('Ground Truth Material Parameters (from preprocessed file):')
    print('-'*80)

    # Background region (label 0)
    bg_mask = labels == 0
    print(f'Background region (label=0): {bg_mask.sum()} elements')
    print(f'  alpha = {alpha_np[bg_mask]}')
    print(f'  beta = {beta_np[bg_mask]}')
    print(f'  kappa = {kappa_np[bg_mask]}')
    print()

    # Special region (label 1)
    sp_mask = labels == 1
    print(f'Special region (label=1): {sp_mask.sum()} elements')
    print(f'  alpha = {alpha_np[sp_mask]}')
    print(f'  beta = {beta_np[sp_mask]}')
    print(f'  kappa = {kappa_np[sp_mask]}')
    alpha_gt = torch.from_numpy(alpha_np).to(device=device, dtype=torch.float64)
    beta_gt = torch.from_numpy(beta_np).to(device=device, dtype=torch.float64)
    kappa_gt = torch.from_numpy(kappa_np).to(device=device, dtype=torch.float64)

    sim.assemble_matrix(alpha_gt, beta_gt, kappa_gt)
    u_obs = sim.get_observed_free().to(dtype=torch.float64)
    obs_norm_sq = torch.dot(u_obs, u_obs)
    return u_obs, obs_norm_sq


def initialize_or_resume(
    config: LandscapeConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Set up storage arrays, resuming from checkpoint if present."""
    alpha_bg_values = np.linspace(
        config.alpha_bg_min, config.alpha_bg_max, config.alpha_bg_steps, dtype=np.float64
    )
    alpha_sp_factors = np.linspace(
        config.alpha_sp_factor_min, config.alpha_sp_factor_max, config.alpha_sp_factor_steps, dtype=np.float64
    )
    alpha_sp_values = np.outer(alpha_bg_values, alpha_sp_factors)
    loss_grid = np.full_like(alpha_sp_values, np.nan, dtype=np.float64)
    completed_rows = np.zeros(config.alpha_bg_steps, dtype=bool)
    progress_done = 0

    checkpoint_path = config.checkpoint_path.resolve()
    if checkpoint_path.exists():
        print(f"Found checkpoint at {checkpoint_path}; attempting resume.")
        with np.load(checkpoint_path, allow_pickle=True) as data:
            loss_grid_ckpt = data["loss_grid"]
            completed_rows_ckpt = data["completed_rows"]
            alpha_bg_ckpt = data["alpha_bg_values"]
            alpha_sp_factors_ckpt = data["alpha_sp_factors"]

            if loss_grid_ckpt.shape != loss_grid.shape:
                raise ValueError("Checkpoint loss grid shape does not match current configuration.")
            if not np.allclose(alpha_bg_ckpt, alpha_bg_values):
                raise ValueError("Checkpoint alpha_bg_values differ from current configuration.")
            if not np.allclose(alpha_sp_factors_ckpt, alpha_sp_factors):
                raise ValueError("Checkpoint alpha_sp_factors differ from current configuration.")

            loss_grid = loss_grid_ckpt
            completed_rows = completed_rows_ckpt.astype(bool)
            if "alpha_sp_values" in data:
                alpha_sp_values = data["alpha_sp_values"]
            if "progress_done" in data:
                progress_done = int(data["progress_done"])
            else:
                progress_done = int(completed_rows.sum() * config.alpha_sp_factor_steps)
    return (
        alpha_bg_values,
        alpha_sp_factors,
        alpha_sp_values,
        loss_grid,
        completed_rows,
        progress_done,
    )


def save_checkpoint(
    config: LandscapeConfig,
    alpha_bg_values: np.ndarray,
    alpha_sp_factors: np.ndarray,
    alpha_sp_values: np.ndarray,
    loss_grid: np.ndarray,
    completed_rows: np.ndarray,
    progress_done: int,
) -> None:
    """Persist intermediate results so we can resume later."""
    checkpoint_path = config.checkpoint_path.resolve()
    tmp_path = checkpoint_path.parent / f"{checkpoint_path.name}.tmp"
    with tmp_path.open("wb") as tmp_file:
        np.savez(
            tmp_file,
            alpha_bg_values=alpha_bg_values,
            alpha_sp_factors=alpha_sp_factors,
            alpha_sp_values=alpha_sp_values,
            loss_grid=loss_grid,
            completed_rows=completed_rows.astype(np.uint8),
            progress_done=progress_done,
            config=np.array([asdict(config)], dtype=object),
        )
    os.replace(tmp_path, checkpoint_path)


def main() -> None:
    config = get_config()
    device = torch.device("cuda")

    preprocessed_path = config.preprocessed_path.resolve()
    output_path = config.output_path.resolve()
    checkpoint_path = config.checkpoint_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading preprocessed data from: {preprocessed_path}")
    sim = ConeStaticEquilibrium(str(preprocessed_path))

    with np.load(preprocessed_path, allow_pickle=True) as data:
        labels_np = data["labels"].astype(np.int32)

    unique_labels = np.unique(labels_np)
    if unique_labels.size != 2:
        raise ValueError(
            f"Expected exactly two region labels, found: {unique_labels}."
        )
    bg_label = int(unique_labels.min())
    sp_label = int(unique_labels.max())
    print(f"Detected background label={bg_label}, special label={sp_label}")

    labels_gpu = torch.from_numpy(labels_np).to(device=device, dtype=torch.long)
    special_mask = labels_gpu == sp_label


    # Fetch observed displacement and denominator once.
    # First one is loading u_fem as u_obs. 
    # The other is using the params to obtain a U_sms as u_obs.
    """
    u_obs, obs_norm_sq_tensor = load_observed_displacement(sim, preprocessed_path, device)
    denom = obs_norm_sq_tensor + torch.tensor(config.eps_div, device=device, dtype=torch.float64)
    obs_norm_sq = float(obs_norm_sq_tensor.item())"""
    with np.load(preprocessed_path, allow_pickle=True) as data:
        alpha_true = data["alpha_k"].astype(np.float64)
        beta_true = data["beta_k"].astype(np.float64)
        kappa_true = data["kappa_k"].astype(np.float64)
    alpha_true_torch = torch.from_numpy(alpha_true).to(device=device, dtype=torch.float64)
    beta_true_torch = torch.from_numpy(beta_true).to(device=device, dtype=torch.float64)
    kappa_true_torch = torch.from_numpy(kappa_true).to(device=device, dtype=torch.float64)
    sim.assemble_matrix(alpha_true_torch, beta_true_torch, kappa_true_torch)

    u_obs = sim.forward().to(dtype=torch.float64)
    obs_norm_sq_tensor = torch.dot(u_obs, u_obs)
    denom = obs_norm_sq_tensor + torch.tensor(config.eps_div, device=device, dtype=torch.float64)
    obs_norm_sq = float(obs_norm_sq_tensor.item())
    (
        alpha_bg_values,
        alpha_sp_factors,
        alpha_sp_values,
        loss_grid,
        completed_rows,
        progress_done,
    ) = initialize_or_resume(config)

    alpha_field = torch.empty(sim.M, device=device, dtype=torch.float64)

    special_mask_has_entries = bool(special_mask.any().item())

    total_evals = config.alpha_bg_steps * config.alpha_sp_factor_steps
    print(
        f"Evaluating grid: {config.alpha_bg_steps} x {config.alpha_sp_factor_steps} "
        f"({total_evals} solves)."
    )
    if tqdm is not None:
        progress_bar = tqdm(total=total_evals, desc="Loss landscape", leave=True, initial=progress_done)
        progress_interval = None
    else:
        progress_bar = None
        progress_interval = max(1, total_evals // 100)
    progress_counter = progress_done
    if progress_bar is None and progress_done > 0:
        pct = 100.0 * progress_counter / total_evals
        print(f"Progress: {progress_counter}/{total_evals} ({pct:6.2f}%)", end="\r", flush=True)

    with torch.no_grad():
        for i_bg, alpha_bg in enumerate(alpha_bg_values):
            print(f"  [{i_bg + 1:03d}/{config.alpha_bg_steps}] alpha_bg = {alpha_bg:.4f}")
            if completed_rows[i_bg]:
                continue

            for j_factor, factor in enumerate(alpha_sp_factors):
                alpha_sp = alpha_sp_values[i_bg, j_factor]

                alpha_field.fill_(alpha_bg)
                if special_mask_has_entries:
                    alpha_field.masked_fill_(special_mask, alpha_sp)

                beta_field = compute_beta_from_alpha(alpha_field)
                kappa_field = compute_kappa_from_alpha(alpha_field, nu=config.poisson_ratio)

                sim.assemble_matrix(alpha_field, beta_field, kappa_field)

                u_pred = sim.forward(tol=config.forward_tol, max_iter=config.forward_max_iter)

                diff = u_pred - u_obs
                loss_value = torch.dot(diff, diff) / denom
                loss_grid[i_bg, j_factor] = float(loss_value.item())

                progress_counter += 1
                if progress_bar is not None:
                    progress_bar.update(1)
                else:
                    if (
                        progress_counter % progress_interval == 0
                        or progress_counter == total_evals
                    ):
                        pct = 100.0 * progress_counter / total_evals
                        print(
                            f"Progress: {progress_counter}/{total_evals} ({pct:6.2f}%)",
                            end="\r",
                            flush=True,
                        )

            completed_rows[i_bg] = True
            progress_done = min(progress_counter, total_evals)
            save_checkpoint(
                config,
                alpha_bg_values,
                alpha_sp_factors,
                alpha_sp_values,
                loss_grid,
                completed_rows,
                progress_done,
            )

    if progress_bar is not None:
        progress_bar.close()
    else:
        print()

    metadata = {
        "bg_label": bg_label,
        "sp_label": sp_label,
        "obs_norm_sq": obs_norm_sq,
        "eps_div": config.eps_div,
        "forward_tol": config.forward_tol,
        "forward_max_iter": config.forward_max_iter,
        "poisson_ratio": config.poisson_ratio,
    }

    np.savez(
        output_path,
        alpha_bg_values=alpha_bg_values,
        alpha_sp_factors=alpha_sp_factors,
        alpha_sp_values=alpha_sp_values,
        relative_loss=loss_grid,
        metadata=np.array([metadata], dtype=object),
        config=np.array([asdict(config)], dtype=object),
    )
    print(f"Saved loss landscape to: {output_path}")
    try:
        checkpoint_path.unlink()
    except FileNotFoundError:
        pass


if __name__ == "__main__":
    main()
