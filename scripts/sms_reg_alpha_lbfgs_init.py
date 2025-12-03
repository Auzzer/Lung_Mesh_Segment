"""
Simplified SMS alpha optimizer:
- alpha_k initialized from CT (via add_alpha_from_hu if missing)
- single-stage LBFGS directly on per-tet alpha
- displacement Charbonnier-TV regularization (reuses sms_reg_u_lbfgs)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import torch

current_dir = Path(__file__).resolve().parent
repo_root = current_dir.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
project_root = repo_root / "lung_project_git"
if project_root.is_dir() and str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.utils.add_alpha_from_hu import AlphaPriorConfig, add_alpha_from_hu
from utils.sms_precompute_utils import (
    get_emory_example_paths,
    normalize_mesh_tag,
    run_sms_preprocessor,
    validate_preprocessed_states,
)
from scripts.sms_reg_u_lbfgs import (
    ConeStaticEquilibrium,
    ALPHA_MIN,
    ALPHA_MAX,
    REG_WEIGHT,
    MEAN_OBSERVED_DISP_WARNING,
    compute_beta_from_alpha,
    compute_kappa_from_alpha,
)
from scripts.sms_reg_u_lbfgs import _ensure_taichi_initialized  # reuse Taichi init
from scripts.sms_reg_u_lbfgs import save_parameter_heatmap


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
        mesh_path = Path(paths["fixed_mesh"])
        disp_path = Path(paths["disp_field"])
        log_fn("Precomputing deformation data via utils/sms_precompute.py.")
        run_sms_preprocessor(mesh_path, disp_path, pre_npz, metadata=preprocess_metadata, log_fn=log_fn)
    else:
        log_fn(f"Using cached deformation data: {pre_npz}")

    validate_preprocessed_states(pre_npz, args.subject, args.fixed_state, args.moving_state, log_fn)
    return pre_npz, paths, needs_precompute


def maybe_add_alpha_prior(
    pre_npz: Path,
    ct_path: str | None,
    alpha_output: str | None,
    hu_params: AlphaPriorConfig,
    log_fn: Callable[[str], None],
    ct_fallback: Path | None = None,
) -> Path:
    with np.load(pre_npz, allow_pickle=True) as data:
        if "alpha_k" in data.files:
            log_fn("[Alpha] Found alpha_k in preprocessing; using existing field.")
            return pre_npz
    ct_candidate = Path(ct_path) if ct_path else ct_fallback
    if ct_candidate is None:
        raise RuntimeError("alpha_k missing and no CT provided to build it (--ct-path).")
    if not ct_candidate.exists():
        raise FileNotFoundError(
            f"CT volume not found: {ct_candidate} | provide --ct-path explicitly."
        )
    out_npz = Path(alpha_output) if alpha_output else pre_npz.with_name(pre_npz.stem + "_with_alpha.npz")
    log_fn("[Alpha] Building HU-based alpha_k via add_alpha_from_hu.")
    add_alpha_from_hu(pre_npz, ct_candidate, out_npz, hu_params)
    return out_npz


def main():
    parser = argparse.ArgumentParser(
        description="SMS inverse problem (direct per-tet alpha with displacement-TV regularization)."
    )
    parser.add_argument("--data-root", default="data/Emory-4DCT", help="Root directory for Emory 4DCT data")
    parser.add_argument("--subject", default="Case2Pack", help="Subject folder name, e.g., Case1Pack")
    parser.add_argument("--fixed-state", default="T10", help="Fixed respiratory state (e.g., T00)")
    parser.add_argument("--moving-state", default="T20", help="Moving respiratory state (e.g., T50)")
    parser.add_argument("--variant", default="NIFTI", help="Dataset variant (default: NIFTI)")
    parser.add_argument("--mask-name", default="lung_regions", help="Mask name used during preprocessing")
    parser.add_argument("--mesh-tag", default="lung_regions_11", help="Mesh tag suffix under pygalmesh/")
    parser.add_argument("--cache-dir", default="data_processed_deformation", help="Directory to store generated .npz files")
    parser.add_argument("--preprocessed", help="Path to existing SMS preprocessing .npz (skip generation)")
    parser.add_argument("--force-preprocess", action="store_true", help="Regenerate preprocessing even if cache exists")
    parser.add_argument("--ct-path", required=False, help="Fixed CT (NIFTI) to build HU-based alpha_k prior when missing")
    parser.add_argument("--alpha-output", help="Output path for augmented preprocessing npz (defaults to *_with_alpha.npz)")
    parser.add_argument("--hu-low", type=float, default=-950.0, help="HU threshold where emphysema prior reaches full weight")
    parser.add_argument("--hu-high", type=float, default=-800.0, help="HU threshold where normal-lung prior reaches full weight")
    parser.add_argument("--alpha-emphysema", type=float, default=3e3, help="Alpha mapped to HU <= hu_low (soft lung / emphysema)")
    parser.add_argument("--alpha-normal", type=float, default=7e3, help="Alpha mapped to HU >= hu_high (normal parenchyma)")
    parser.add_argument("--max-iters", type=int, default=100, help="Number of LBFGS outer iterations")
    args = parser.parse_args()

    max_iters = max(1, args.max_iters)
    base_dir = Path(__file__).resolve().parent
    output_dir = base_dir / "results" / "sms_alpha_direct"
    output_dir.mkdir(parents=True, exist_ok=True)

    log_lines: list[str] = []

    def log(message: str) -> None:
        print(message)
        log_lines.append(message)

    def flush_history() -> None:
        if not log_lines:
            return
        log_path = output_dir / "optimization.log"
        with log_path.open("w", encoding="utf-8") as log_file:
            log_file.write("\n".join(log_lines))
            log_file.write("\n")

    log("SMS INVERSE PROBLEM (direct alpha_k, displacement-TV reg)")
    log(f"Emory case: {args.subject} {args.fixed_state}->{args.moving_state} (variant={args.variant})")
    log(f"Mask: {args.mask_name}, mesh tag: {args.mesh_tag}")

    pre, example_paths, ran_precompute = ensure_preprocessed_file(args, log)
    hu_params = AlphaPriorConfig(
        hu_low=args.hu_low,
        hu_high=args.hu_high,
        alpha_emphysema=args.alpha_emphysema,
        alpha_normal=args.alpha_normal,
    )
    ct_fallback = Path(args.ct_path) if args.ct_path else None
    if ct_fallback is None and example_paths.get("fixed_image"):
        candidate = Path(example_paths["fixed_image"])
        if candidate.exists():
            ct_fallback = candidate
            log(f"[Alpha] Using fixed_image from dataset as CT prior: {candidate}")

    pre = maybe_add_alpha_prior(
        pre,
        args.ct_path,
        args.alpha_output,
        hu_params,
        log,
        ct_fallback=ct_fallback,
    )

    log(f"Using preprocessed data: {pre}")
    if example_paths:
        log(f"  mesh: {example_paths.get('fixed_mesh')}")
        log(f"  displacement: {example_paths.get('disp_field')}")
    if ran_precompute:
        log("Finished deformation preprocessing; Taichi will initialize next.")

    _ensure_taichi_initialized()
    sim = ConeStaticEquilibrium(str(pre))
    total_mass_g = sim.get_total_mass_grams()
    log(f"  total nodal mass: {total_mass_g:.6f} g")
    log(f"  mesh: {sim.N} nodes, {sim.M} elements")

    with np.load(pre, allow_pickle=True) as data:
        labels_np = data["labels"]
        if "alpha_k" not in data.files:
            raise RuntimeError("alpha_k missing after preprocessing/prior augmentation.")
        alpha_np_data = data["alpha_k"].astype(np.float64)

    log("[Step] extracting observed displacement from preprocessing data")
    alpha_init_field = torch.ones(sim.M, device="cuda", dtype=torch.float64)
    beta_init_field = compute_beta_from_alpha(alpha_init_field)
    kappa_init_field = compute_kappa_from_alpha(alpha_init_field)
    sim.assemble_matrix(alpha_init_field, beta_init_field, kappa_init_field)
    u_obs = sim.get_observed_free()
    mean_obs_disp_m = float(torch.mean(torch.abs(u_obs)).item())
    log(f"Mean observed displacement (free DOF): {mean_obs_disp_m:.6e} m")
    if mean_obs_disp_m < MEAN_OBSERVED_DISP_WARNING:
        log("[Sanity] Observed field magnitude is unusually small; verify units or rescale targets.")

    alpha_seed_np = np.clip(alpha_np_data, ALPHA_MIN, ALPHA_MAX)
    alpha_param = torch.nn.Parameter(torch.from_numpy(alpha_seed_np).to(device="cuda", dtype=torch.float64))

    log("[Opt] starting LBFGS on alpha_k (single stage)")
    optimizer = torch.optim.LBFGS(
        [alpha_param],
        lr=1.0,
        max_iter=20,
        line_search_fn="strong_wolfe",
    )

    loss_history: list[float] = []

    def closure():
        optimizer.zero_grad(set_to_none=True)
        alpha_field = torch.clamp(alpha_param, min=ALPHA_MIN, max=ALPHA_MAX)
        beta_field = compute_beta_from_alpha(alpha_field)
        kappa_field = compute_kappa_from_alpha(alpha_field)

        sim.assemble_matrix(alpha_field, beta_field, kappa_field)
        sim.forward(tol=1e-6, max_iter=200)

        loss_total, loss_data, loss_reg, grad_alpha = sim.backward(
            u_obs, tol=1e-6, max_iter=200, reg_weight=REG_WEIGHT
        )

        alpha_param.grad = grad_alpha
        loss_history.append(float(loss_total.item()))
        log(
            "[Opt][closure] loss={:.6e}, data={:.6e}, reg={:.6e}, alpha_mean={:.4e}, alpha_std={:.4e}".format(
                float(loss_total.item()),
                float(loss_data.item()),
                float(loss_reg.item()),
                float(alpha_field.mean().item()),
                float(alpha_field.std().item()),
            )
        )
        return loss_total

    for _ in range(max_iters):
        optimizer.step(closure)
        if len(loss_history) >= 2 and abs(loss_history[-1] - loss_history[-2]) <= 1e-6 * max(
            1.0, abs(loss_history[-2])
        ):
            log("[Opt] Loss change below tolerance; stopping early.")
            break

    alpha_final_field = torch.clamp(alpha_param.detach(), min=ALPHA_MIN, max=ALPHA_MAX)
    save_path = output_dir / "alpha_final.xdmf"
    save_parameter_heatmap(sim, alpha_final_field, save_path, labels_np, log_fn=log)

    log("\nOptimization complete.")
    log(
        f"  alpha stats: mean={float(alpha_final_field.mean().item()):.4e}, "
        f"std={float(alpha_final_field.std().item()):.4e}"
    )

    flush_history()


if __name__ == "__main__":
    main()
