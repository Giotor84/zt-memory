"""Poincaré-ball geometry with curvature -c, in PyTorch.

All functions operate on the last dimension and broadcast over the leading ones.
Reference: Ganea, Bécigneul, Hofmann, "Hyperbolic Neural Networks", NeurIPS 2018.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor

MIN_NORM = 1e-15


def _ball_eps(x: Tensor) -> float:
    return 1e-10 if x.dtype == torch.float64 else 1e-5


def sqnorm(x: Tensor, keepdim: bool = True) -> Tensor:
    return (x * x).sum(dim=-1, keepdim=keepdim)


def safe_norm(x: Tensor, keepdim: bool = True) -> Tensor:
    return x.norm(dim=-1, keepdim=keepdim).clamp_min(MIN_NORM)


def artanh(x: Tensor) -> Tensor:
    lim = 1.0 - _ball_eps(x)
    return torch.atanh(x.clamp(-lim, lim))


def project(x: Tensor, c: float = 1.0) -> Tensor:
    """Clip points numerically outside the open ball back inside."""
    max_norm = (1.0 - _ball_eps(x)) / math.sqrt(c)
    n = safe_norm(x)
    return torch.where(n > max_norm, x / n * max_norm, x)


def conformal_factor(x: Tensor, c: float = 1.0, keepdim: bool = True) -> Tensor:
    """lambda_x = 2 / (1 - c ||x||^2)."""
    return 2.0 / (1.0 - c * sqnorm(x, keepdim=keepdim)).clamp_min(MIN_NORM)


def mobius_add(x: Tensor, y: Tensor, c: float = 1.0) -> Tensor:
    x2 = sqnorm(x)
    y2 = sqnorm(y)
    xy = (x * y).sum(dim=-1, keepdim=True)
    num = (1.0 + 2.0 * c * xy + c * y2) * x + (1.0 - c * x2) * y
    den = 1.0 + 2.0 * c * xy + c * c * x2 * y2
    return project(num / den.clamp_min(MIN_NORM), c)


def expmap(x: Tensor, u: Tensor, c: float = 1.0) -> Tensor:
    sqrt_c = math.sqrt(c)
    u_norm = safe_norm(u)
    second = torch.tanh(sqrt_c * conformal_factor(x, c) * u_norm / 2.0) * u / (sqrt_c * u_norm)
    return mobius_add(x, second, c)


def logmap(x: Tensor, y: Tensor, c: float = 1.0) -> Tensor:
    sqrt_c = math.sqrt(c)
    sub = mobius_add(-x, y, c)
    sub_norm = safe_norm(sub)
    return (2.0 / (sqrt_c * conformal_factor(x, c))) * artanh(sqrt_c * sub_norm) * sub / sub_norm


def expmap0(u: Tensor, c: float = 1.0) -> Tensor:
    sqrt_c = math.sqrt(c)
    u_norm = safe_norm(u)
    return project(torch.tanh(sqrt_c * u_norm) * u / (sqrt_c * u_norm), c)


def logmap0(y: Tensor, c: float = 1.0) -> Tensor:
    sqrt_c = math.sqrt(c)
    y_norm = safe_norm(y)
    return artanh(sqrt_c * y_norm) * y / (sqrt_c * y_norm)


def distance(x: Tensor, y: Tensor, c: float = 1.0) -> Tensor:
    """Geodesic distance, broadcast over leading dims (returns shape without the last dim)."""
    diff2 = sqnorm(x - y, keepdim=False)
    den = ((1.0 - c * sqnorm(x, keepdim=False)) * (1.0 - c * sqnorm(y, keepdim=False))).clamp_min(MIN_NORM)
    arg = 1.0 + 2.0 * c * diff2 / den
    return torch.acosh(arg.clamp_min(1.0)) / math.sqrt(c)


def pairwise_distance(q: Tensor, x: Tensor, c: float = 1.0) -> Tensor:
    """[m, r] x [n, r] -> [m, n] geodesic distances (matmul form, used by block scans)."""
    q2 = sqnorm(q)  # [m, 1]
    x2 = sqnorm(x).transpose(0, 1)  # [1, n]
    diff2 = (q2 + x2 - 2.0 * (q @ x.transpose(0, 1))).clamp_min(0.0)
    den = ((1.0 - c * q2) * (1.0 - c * x2)).clamp_min(MIN_NORM)
    return torch.acosh((1.0 + 2.0 * c * diff2 / den).clamp_min(1.0)) / math.sqrt(c)


def transport_to_origin(x: Tensor, v: Tensor, c: float = 1.0) -> Tensor:
    """Parallel transport of v in T_x to T_0: (lambda_x / lambda_0) v."""
    return v * conformal_factor(x, c) / 2.0


def transport_from_origin(x: Tensor, v0: Tensor, c: float = 1.0) -> Tensor:
    """Parallel transport of v0 in T_0 to T_x: (lambda_0 / lambda_x) v0."""
    return v0 * 2.0 / conformal_factor(x, c)


def origin_tangent_norm(v0: Tensor) -> Tensor:
    """Riemannian norm of a vector in T_0 (lambda_0 = 2 for every c)."""
    return 2.0 * v0.norm(dim=-1)


def place(directions: Tensor, radius: Tensor | float) -> Tensor:
    """Embed unit directions at given Euclidean radii (hierarchy level -> radial coordinate)."""
    d = directions / safe_norm(directions)
    r = radius if isinstance(radius, Tensor) else torch.tensor(radius, dtype=d.dtype)
    if r.ndim == 1:
        r = r[:, None]
    return d * r.to(d.dtype)
