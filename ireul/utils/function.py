import torch
from torch.functional import F


def soft_clamp(x: torch.Tensor, _min=None, _max=None):
    # clamp tensor values while mataining the gradient
    if _max is not None:
        x = _max - F.softplus(_max - x)
    if _min is not None:
        x = _min + F.softplus(x - _min)
    return x


def asymmetric_l2_loss(u: torch.Tensor, tau: float) -> torch.Tensor:
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)


def quantile_huber_loss(
    u: torch.Tensor, tau: float, k: float = 1.0
) -> torch.Tensor:
    return torch.mean(
        torch.abs(tau - (u < 0).float())
        * F.huber_loss(u, torch.zeros_like(u), delta=k, reduction="none")
        / k
    )
