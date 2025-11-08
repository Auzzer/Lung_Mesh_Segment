"""
Barzilai-Borwein optimizer implemented as a PyTorch Optimizer subclass.

This optimizer approximates second-order information using the most recent
differences in parameters and gradients to compute an adaptive step size.
"""

from __future__ import annotations

from typing import Callable, Iterable, Optional, List

import torch
from torch.optim import Optimizer


class BarzilaiBorwein(Optimizer):
    """
    Basic Barzilai-Borwein optimizer.

    The optimizer falls back to the provided base learning rate whenever the BB
    step size cannot be computed safely (e.g. non-positive curvature).
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-2,
        lr_min: float = 1e-6,
        lr_max: float = 1.0,
        eps: float = 1e-12,
    ) -> None:
        if lr <= 0.0:
            raise ValueError(f"BarzilaiBorwein: lr must be positive, got {lr}")
        if lr_min <= 0.0:
            raise ValueError(f"BarzilaiBorwein: lr_min must be positive, got {lr_min}")
        if lr_max <= 0.0:
            raise ValueError(f"BarzilaiBorwein: lr_max must be positive, got {lr_max}")
        if lr_min > lr_max:
            raise ValueError(
                f"BarzilaiBorwein: lr_min ({lr_min}) must not exceed lr_max ({lr_max})"
            )
        if eps <= 0.0:
            raise ValueError(f"BarzilaiBorwein: eps must be positive, got {eps}")

        defaults = dict(lr=lr, lr_min=lr_min, lr_max=lr_max, eps=eps)
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable[[], torch.Tensor]] = None) -> Optional[torch.Tensor]:
        loss: Optional[torch.Tensor] = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = float(group["lr"])
            lr_min = float(group["lr_min"])
            lr_max = float(group["lr_max"])
            eps = float(group["eps"])

            for p in group["params"]:
                if p.grad is None:
                    continue

                if p.grad.is_sparse:
                    raise RuntimeError("Barzilai-Borwein does not support sparse gradients")

                grad = p.grad.detach()
                param_data = p.data.detach().clone()
                state = self.state[p]

                prev_param = state.get("prev_param")
                prev_grad = state.get("prev_grad")
                use_bb1 = state.get("use_bb1", True)

                step_size = lr
                if prev_param is not None and prev_grad is not None:
                    s = (param_data - prev_param).reshape(-1)
                    y = (grad - prev_grad).reshape(-1)

                    sy = torch.dot(s, y)
                    ss = torch.dot(s, s)
                    yy = torch.dot(y, y)

                    sy_val = float(sy.item())
                    ss_val = float(ss.item())
                    yy_val = float(yy.item())

                    step_candidates: List[float] = []

                    if sy_val > eps and ss_val > eps:
                        step_bb1 = ss_val / sy_val
                        if step_bb1 > 0.0:
                            step_candidates.append(step_bb1)

                    if yy_val > eps and sy_val > eps:
                        step_bb2 = sy_val / yy_val
                        if step_bb2 > 0.0:
                            step_candidates.append(step_bb2)

                    if step_candidates:
                        # Choose between BB1 and BB2, alternating preference for stability.
                        idx = 0 if use_bb1 else -1
                        step_size = step_candidates[idx]
                        use_bb1 = not use_bb1

                    step_size = max(lr_min, min(lr_max, step_size))

                state["prev_param"] = param_data
                state["prev_grad"] = grad.clone()
                state["use_bb1"] = use_bb1
                state["step"] = state.get("step", 0) + 1
                state["last_step_size"] = float(step_size)

                p.data.add_(grad, alpha=-step_size)

        return loss
