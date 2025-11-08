"""Run SMS static equilibrium with true and perturbed parameters for comparison.

This script loads the preprocessed cone verification dataset, runs the
`ConeStaticEquilibrium` solver twice (once with the original material
parameters and once with intentionally perturbed parameters), and compares the
resulting SMS displacements. It also writes XDMF files for both solutions and a
separate difference field for visualization.
"""

from __future__ import annotations

from pathlib import Path

import meshio
import numpy as np
import torch

if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parent))
    from hetero_cone_static_equilibrium import ConeStaticEquilibrium
else:
    from .hetero_cone_static_equilibrium import ConeStaticEquilibrium


TORCH_DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.set_float32_matmul_precision("high")


def _run_case(
    preprocessed_path: Path,
    alpha_vals: np.ndarray,
    beta_vals: np.ndarray,
    kappa_vals: np.ndarray,
    output_stem: Path,
    tag: str,
) -> dict[str, np.ndarray]:
    print(f"\n=== Running {tag} parameter set ===")
    sim = ConeStaticEquilibrium(str(preprocessed_path))
    sim.apply_cone_boundary_conditions()

    sim.alpha_k.from_numpy(alpha_vals.astype(np.float64))
    sim.beta_k.from_numpy(beta_vals.astype(np.float64))
    sim.kappa_k.from_numpy(kappa_vals.astype(np.float64))

    if not sim.solve_cone_static_equilibrium():
        raise RuntimeError(f"SMS solve failed for {tag} parameters")

    sim.compare_with_fem()
    sim.save_results(str(output_stem))

    displacement_np = sim.solution_increment.to_numpy().astype(np.float32)
    displacement_torch = torch.from_numpy(displacement_np.copy()).to(TORCH_DEVICE)

    return {
        "initial_positions": sim.initial_positions.to_numpy().copy(),
        "final_positions": sim.x.to_numpy().copy(),
        "displacement": displacement_np,
        "displacement_torch": displacement_torch,
        "connectivity": sim.mesh_connectivity.copy(),
        "labels": None if sim.labels_np is None else sim.labels_np.copy(),
    }


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    preprocessed_path = base_dir / "cone_verification_deformation.npz"
    if not preprocessed_path.exists():
        print(f"Missing preprocessed data: {preprocessed_path}")
        print("Run hetero_cone_deformation_processor.py first.")
        return

    with np.load(preprocessed_path, allow_pickle=True) as data:
        alpha_true = data["alpha_k"].astype(np.float64)
        beta_true = data["beta_k"].astype(np.float64)
        kappa_true = data["kappa_k"].astype(np.float64)

    alpha_wrong = alpha_true * 0.6
    beta_wrong = beta_true * 0.6
    kappa_wrong = kappa_true * 1.8

    true_results = _run_case(
        preprocessed_path,
        alpha_true,
        beta_true,
        kappa_true,
        base_dir / "cone_sms_true_params",
        "true",
    )

    wrong_results = _run_case(
        preprocessed_path,
        alpha_wrong,
        beta_wrong,
        kappa_wrong,
        base_dir / "cone_sms_wrong_params",
        "perturbed",
    )

    disp_true_t = true_results["displacement_torch"]
    disp_wrong_t = wrong_results["displacement_torch"]
    displacement_diff_t = disp_wrong_t - disp_true_t
    if TORCH_DEVICE.type == "cuda":
        torch.cuda.synchronize()

    diff_magnitude_t = displacement_diff_t.norm(dim=1)
    max_diff = float(diff_magnitude_t.max().item())
    mean_diff = float(diff_magnitude_t.mean().item())
    rms_diff = float(torch.sqrt(torch.mean(displacement_diff_t.pow(2).sum(dim=1))).item())

    displacement_diff = displacement_diff_t.cpu().numpy()
    diff_magnitude = diff_magnitude_t.cpu().numpy()

    diff_mesh = meshio.Mesh(
        points=true_results["initial_positions"],
        cells=[("tetra", true_results["connectivity"])],
        point_data={
            "disp_true": true_results["displacement"],
            "disp_wrong": wrong_results["displacement"],
            "disp_difference": displacement_diff,
            "diff_magnitude": diff_magnitude,
        },
        cell_data={"labels": [true_results["labels"]]} if true_results["labels"] is not None else {},
    )

    diff_path = base_dir / "cone_sms_true_vs_wrong_difference.xdmf"
    meshio.write(diff_path, diff_mesh, data_format="XML")
    print(f"Difference mesh saved to {diff_path}")

    print("\n=== Comparison Summary ===")
    print(f"Device: {TORCH_DEVICE.type}")
    print(f"Max difference magnitude: {max_diff:.6e} m")
    print(f"Mean difference magnitude: {mean_diff:.6e} m")
    print(f"RMS difference magnitude: {rms_diff:.6e} m")


if __name__ == "__main__":
    main()
