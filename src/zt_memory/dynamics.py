"""Newtonian memory dynamics: surprise (force), inertia (beta), friction (gamma).

Per node, at discrete time t (one tick per event batch):

    F_t = log_x(z) transported to T_0          (surprise vector; ||F_t|| = d_H(x, z))
    v_t = beta * v_{t-1} + (1 - beta) * F_t    (momentum: recurring patterns accumulate)
    E_t = gamma * E_{t-1} + ||v_t||            (energy with friction; gamma < 1)

Velocities live in the origin chart T_0 so that they remain comparable when a node's
anchor moves. Nodes not hit by any force evolve freely; that evolution has a closed form,
so decay is computed lazily on access (O(1) per node) instead of ticking the whole store.
"""
from __future__ import annotations

import torch
from torch import Tensor

from . import hyperbolic as H


def velocity_norm(v0: Tensor) -> Tensor:
    return H.origin_tangent_norm(v0)


def free_evolution(velocity: Tensor, energy: Tensor, steps: Tensor, beta: Tensor, gamma: Tensor) -> tuple[Tensor, Tensor]:
    """Closed form of k zero-force steps.

    v_{t+k} = beta^k v_t
    E_{t+k} = gamma^k E_t + ||v_t|| * sum_{j=1..k} gamma^{k-j} beta^j
            = gamma^k E_t + ||v_t|| * beta (gamma^k - beta^k) / (gamma - beta)   (gamma != beta)
            = gamma^k E_t + ||v_t|| * k beta^k                                  (gamma == beta)
    """
    dtype = velocity.dtype
    k = steps.clamp_min(0).to(dtype)
    beta = beta.to(dtype)
    gamma = gamma.to(dtype)
    bk = beta**k
    gk = gamma**k
    diff = gamma - beta
    same = diff.abs() < 1e-9
    safe_diff = torch.where(same, torch.ones_like(diff), diff)
    geo = torch.where(same, k * bk, beta * (gk - bk) / safe_diff)
    e = gk * energy.to(dtype) + velocity_norm(velocity) * geo
    v = velocity * bk.unsqueeze(-1)
    return v, e


def newtonian_step(
    velocity: Tensor,
    energy: Tensor,
    last_t: Tensor,
    now: int,
    force: Tensor,
    beta: Tensor,
    gamma: Tensor,
) -> tuple[Tensor, Tensor]:
    """Advance state stored at `last_t` to `now`: free evolution for the gap, then one forced step.

    A second update in the same tick (last_t == now) is applied as an additional step.
    """
    steps = (now - last_t - 1).clamp_min(0)
    v, e = free_evolution(velocity, energy, steps, beta, gamma)
    b = beta.to(v.dtype).unsqueeze(-1)
    v_new = b * v + (1.0 - b) * force.to(v.dtype)
    e_new = gamma.to(v.dtype) * e + velocity_norm(v_new)
    return v_new, e_new


def current_energy(velocity: Tensor, energy: Tensor, last_t: Tensor, now: int, beta: Tensor, gamma: Tensor) -> Tensor:
    return free_evolution(velocity, energy, now - last_t, beta, gamma)[1]


def surprise_force(x: Tensor, z: Tensor, c: float = 1.0) -> tuple[Tensor, Tensor]:
    """Force exerted by input z on a memory anchored at x: (F in T_0, scalar surprise d_H(x, z))."""
    x64 = x.to(torch.float64)
    z64 = z.to(torch.float64)
    f0 = H.transport_to_origin(x64, H.logmap(x64, z64, c), c)
    return f0, H.distance(x64, z64, c)
