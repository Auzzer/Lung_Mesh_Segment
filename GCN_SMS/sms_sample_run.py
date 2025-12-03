"""
Minimal smoke test for the unified SMS layer.

Usage:
    python scripts/sms_sample_run.py --preprocessed <path/to/npz> \
        --omega-u 0.5 --omega-alpha 0.5 --forward-max-iter 200 --adjoint-max-iter 200
"""

import argparse
import math
import sys
from pathlib import Path

import torch

# Local imports
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from sms_torch_layer import (
    ConeStaticEquilibrium,
    SMSLayer,
    SMSLossConfig,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test SMSLayer forward+backward on a preprocessed case.")
    parser.add_argument("--preprocessed", required=True, help="Path to preprocessed SMS .npz file.")
    parser.add_argument("--omega-u", type=float, default=0.5, help="Weight for displacement TV (omega_1).")
    parser.add_argument("--omega-alpha", type=float, default=0.5, help="Weight for alpha log-TV (omega_2).")
    parser.add_argument("--forward-max-iter", type=int, default=200, help="Max iterations for the forward solve.")
    parser.add_argument("--adjoint-max-iter", type=int, default=200, help="Max iterations for the adjoint solve.")
    parser.add_argument("--theta-init", type=float, default=math.log(7000.0), help="Initial global log-alpha.")
    parser.add_argument("--delta-init", type=float, default=0.0, help="Initial per-tet delta value.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_default_dtype(torch.float64)

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required to run this sample.")

    pre_npz = Path(args.preprocessed)
    if not pre_npz.exists():
        raise SystemExit(f"Preprocessed file not found: {pre_npz}")

    print(f"Loading solver from {pre_npz} ...")
    sim = ConeStaticEquilibrium(str(pre_npz))
    u_obs_free = sim.get_observed_free().to(device="cuda", dtype=torch.float64)

    # Parameters (theta scalar + delta field) with gradients
    theta = torch.tensor([args.theta_init], device="cuda", requires_grad=True, dtype=torch.float64)
    delta = torch.full((sim.M,), args.delta_init, device="cuda", requires_grad=True, dtype=torch.float64)

    config = SMSLossConfig(
        forward_max_iter=args.forward_max_iter,
        adjoint_max_iter=args.adjoint_max_iter,
    )

    layer = SMSLayer(
        sim=sim,
        u_obs_free=u_obs_free,
        omega_u=args.omega_u,
        omega_alpha=args.omega_alpha,
        config=config,
        return_components=True,
    )

    print("Running forward + backward...")
    loss, extras = layer(theta, delta)
    loss.backward()

    print(f"loss_total     = {loss.item():.6e}")
    print(f"loss_data      = {extras['loss_data'].item():.6e}")
    print(f"loss_u_reg     = {extras['loss_u_reg'].item():.6e}")
    print(f"loss_alpha_reg = {extras['loss_alpha_reg'].item():.6e}")
    print(f"|dL/dtheta|    = {theta.grad.abs().item():.6e}")
    print(f"|dL/ddelta|_max= {delta.grad.abs().max().item():.6e}")


if __name__ == "__main__":
    main()
