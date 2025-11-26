import torch
from typing import Literal

__all__ = [
    "_lookup_alphas_cumprod",
    "_compute_snr_from_indexed",
    "_compute_snr",
    "apply_min_snr_weight",
]


@torch.no_grad()
def _lookup_alphas_cumprod(noise_scheduler, device: torch.device) -> torch.Tensor:
    ac = getattr(noise_scheduler, "alphas_cumprod", None)
    if ac is None:
        raise AttributeError("noise_scheduler has no 'alphas_cumprod'; unsupported scheduler.")
    return ac.to(device=device, dtype=torch.float32)


@torch.no_grad()
def _compute_snr_from_indexed(ac_selected: torch.Tensor) -> torch.Tensor:
    ac = ac_selected.clamp(1e-8, 1.0 - 1e-8)
    return ac / (1.0 - ac)


@torch.no_grad()
def _compute_snr(noise_scheduler, timesteps: torch.Tensor) -> torch.Tensor:
    device = timesteps.device
    ac = _lookup_alphas_cumprod(noise_scheduler, device)
    t = timesteps.to(dtype=torch.long).clamp_(0, ac.numel() - 1)
    return _compute_snr_from_indexed(ac.index_select(0, t))


def apply_min_snr_weight(
    loss_per_example: torch.Tensor,
    timesteps: torch.Tensor,
    noise_scheduler,
    *,
    gamma: float = 5.0,
    prediction_type: Literal["epsilon", "v_prediction"] = "epsilon",
    reduction: Literal["mean", "sum", "none"] = "mean",
) -> torch.Tensor:
    device = timesteps.device
    ac = _lookup_alphas_cumprod(noise_scheduler, device)
    t = timesteps.to(dtype=torch.long).clamp_(0, ac.numel() - 1)

    snr = _compute_snr_from_indexed(ac.index_select(0, t))
    snr = snr.clamp(min=1e-8)

    denom = snr if prediction_type == "epsilon" else (snr + 1.0)
    numer = torch.minimum(snr, torch.tensor(float(gamma), device=device))
    weights = (numer / denom).to(dtype=loss_per_example.dtype)

    while weights.ndim < loss_per_example.ndim:
        weights = weights.unsqueeze(-1)

    weighted = loss_per_example * weights

    if reduction == "mean":
        return weighted.mean()
    if reduction == "sum":
        return weighted.sum()
    return weighted