import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


def asymmetric_l2(u: torch.Tensor, tau: float) -> torch.Tensor:
    return torch.abs(tau - (u < 0).float()) * u**2


def quantile_huber(
    u: torch.Tensor, tau: float = 0.7, k: float = 1.0
) -> torch.Tensor:
    return torch.abs(tau - (u < 0).float()) * F.huber_loss(
        u, torch.zeros_like(u), delta=k, reduction="none"
    )


x = torch.linspace(-5, 5, 1000)
for tau in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    y = asymmetric_l2(x, tau)
    plt.plot(x, y, label=rf"$\tau={tau}$")
plt.legend()
plt.show()
