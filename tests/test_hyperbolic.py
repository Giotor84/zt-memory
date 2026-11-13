import torch

from zt_memory import hyperbolic as H

torch.manual_seed(0)


def _points(n: int, r: int = 8, max_norm: float = 0.85) -> torch.Tensor:
    x = torch.randn(n, r, dtype=torch.float64)
    return x / x.norm(dim=-1, keepdim=True) * torch.rand(n, 1, dtype=torch.float64) * max_norm


def test_exp_log_are_inverse():
    x, y = _points(64), _points(64)
    assert torch.allclose(H.expmap(x, H.logmap(x, y)), y, atol=1e-8)
    assert torch.allclose(H.expmap0(H.logmap0(y)), y, atol=1e-10)


def test_distance_properties():
    x, y = _points(64), _points(64)
    d = H.distance(x, y)
    assert torch.allclose(d, H.distance(y, x), atol=1e-10)
    assert torch.allclose(H.distance(x, x), torch.zeros(64, dtype=torch.float64), atol=1e-6)
    mobius = 2 * H.artanh(H.mobius_add(-x, y).norm(dim=-1))
    assert torch.allclose(d, mobius, atol=1e-7)


def test_pairwise_matches_elementwise():
    q, x = _points(5), _points(7)
    pw = H.pairwise_distance(q, x)
    ew = H.distance(q[:, None, :], x[None, :, :])
    assert torch.allclose(pw, ew, atol=1e-6)


def test_logmap_riemannian_norm_is_distance():
    x, y = _points(32), _points(32)
    riem = H.conformal_factor(x)[:, 0] * H.logmap(x, y).norm(dim=-1)
    assert torch.allclose(riem, H.distance(x, y), atol=1e-7)


def test_transport_to_origin_is_isometry():
    x = _points(32)
    v = torch.randn(32, 8, dtype=torch.float64)
    riem_x = H.conformal_factor(x)[:, 0] * v.norm(dim=-1)
    v0 = H.transport_to_origin(x, v)
    assert torch.allclose(H.origin_tangent_norm(v0), riem_x, atol=1e-9)
    assert torch.allclose(H.transport_from_origin(x, v0), v, atol=1e-12)
