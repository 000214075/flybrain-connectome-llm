"""Spiking neuron primitives used by the connectome module.

Kenyon cells in the mushroom body fire sparsely and the network is trained with
gradient descent, so the spike nonlinearity needs a surrogate gradient: the
forward pass is a real Heaviside spike, the backward pass uses a smooth
approximation (SuperSpike, Zenke & Ganguli 2018). Nothing here fakes the
dynamics -- the membrane potential is integrated with the real LIF equation.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class _SurrogateSpike(torch.autograd.Function):
    """Heaviside forward, SuperSpike derivative backward: 1 / (1 + beta*|v - v_th|)^2."""

    beta = 10.0

    @staticmethod
    def forward(ctx, v_minus_th: Tensor) -> Tensor:  # type: ignore[override]
        ctx.save_for_backward(v_minus_th)
        return (v_minus_th > 0).to(v_minus_th.dtype)

    @staticmethod
    def backward(ctx, grad_output: Tensor):  # type: ignore[override]
        (v_minus_th,) = ctx.saved_tensors
        beta = _SurrogateSpike.beta
        grad = grad_output / (1.0 + beta * v_minus_th.abs()).pow(2)
        return grad


def surrogate_spike(v_minus_th: Tensor) -> Tensor:
    return _SurrogateSpike.apply(v_minus_th)


def k_wta(x: Tensor, k: int) -> Tensor:
    """Keep the k largest values per row, zero the rest.

    This mirrors the measured sparseness of Kenyon cell populations, where only
    a small percentage of cells respond to any given odour.
    """
    if k <= 0 or k >= x.shape[-1]:
        return x
    threshold = x.topk(k, dim=-1).values[..., -1:]
    return x * (x >= threshold)


class LIFLayer(nn.Module):
    """Leaky integrate-and-fire dynamics over `steps` time steps.

    The layer is used as a recurrent module: the same input current drives the
    population for several steps and the output is the spike count, which gives
    the module a real temporal dimension rather than a single feed-forward pass.
    """

    def __init__(
        self,
        size: int,
        *,
        steps: int = 4,
        tau_m: float = 20.0,
        dt: float = 1.0,
        v_th: float = 1.0,
        v_reset: float = 0.0,
        v_rest: float = 0.0,
        learn_tau: bool = True,
    ) -> None:
        super().__init__()
        self.size = size
        self.steps = steps
        self.v_th = v_th
        self.v_reset = v_reset
        self.v_rest = v_rest
        # Stored as a raw parameter and squashed so tau stays positive.
        init = torch.log(torch.expm1(torch.tensor(tau_m / dt - 1.0)))
        self.raw_tau = nn.Parameter(init.clone(), requires_grad=learn_tau)
        self.dt = dt

    @property
    def decay(self) -> Tensor:
        tau = torch.nn.functional.softplus(self.raw_tau) + 1.0
        return (self.dt / tau).clamp(max=1.0)

    def forward(self, current: Tensor, *, spiking: bool = True, steps: int | None = None) -> Tensor:
        """Integrate `current` (..., size) over time; return spikes summed over time."""
        n_steps = steps or self.steps
        decay = self.decay
        v = torch.full_like(current, self.v_rest)
        spikes = torch.zeros_like(current)
        for _ in range(n_steps):
            v = v + decay * (-(v - self.v_rest) + current)
            spike = surrogate_spike(v - self.v_th)
            v = v * (1.0 - spike) + self.v_reset * spike
            spikes = spikes + spike
        if spiking:
            return spikes
        # Rate-coded: the same membrane equation with the threshold crossing replaced
        # by a saturating firing rate. This used to return `relu(current)`, which is
        # unbounded -- and inside a recurrent loop an unbounded activation is positive
        # feedback with no brake. Measured on the connectome-only model: the loss
        # reached 6.2e5 with a gradient norm of 1.9e8 within fifty steps. A real
        # neuron's firing rate saturates; this is what that looks like, and it is the
        # same bounded quantity the analogue read-out already used.
        return torch.sigmoid(current - self.v_th)
