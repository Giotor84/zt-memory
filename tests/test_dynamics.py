import torch

from zt_memory import hyperbolic as H
from zt_memory.dynamics import current_energy, free_evolution, newtonian_step, surprise_force, velocity_norm

D = torch.float64


def _iterate(v, e, forces, beta, gamma):
    for f in forces:
        v = beta * v + (1 - beta) * f
        e = gamma * e + velocity_norm(v)
    return v, e


def test_free_evolution_matches_iteration():
    v0 = torch.randn(4, 6, dtype=D)
    e0 = torch.rand(4, dtype=D)
    for beta, gamma in [(0.8, 0.95), (0.9, 0.9), (0.5, 0.6)]:
        b = torch.full((4,), beta, dtype=D)
        g = torch.full((4,), gamma, dtype=D)
        k = 7
        v_it, e_it = v0.clone(), e0.clone()
        for _ in range(k):
            v_it = beta * v_it
            e_it = gamma * e_it + velocity_norm(v_it)
        v_cf, e_cf = free_evolution(v0, e0, torch.full((4,), k), b, g)
        assert torch.allclose(v_cf, v_it, atol=1e-12)
        assert torch.allclose(e_cf, e_it, atol=1e-10)


def test_newtonian_step_with_gap_matches_iteration():
    v0 = torch.randn(3, 5, dtype=D)
    e0 = torch.rand(3, dtype=D)
    f = torch.randn(3, 5, dtype=D)
    b = torch.full((3,), 0.85, dtype=D)
    g = torch.full((3,), 0.92, dtype=D)
    last_t, now = torch.full((3,), 10), 14  # 3 free steps + 1 forced
    zeros = torch.zeros_like(f)
    v_it, e_it = _iterate(v0, e0, [zeros, zeros, zeros, f], 0.85, 0.92)
    v, e = newtonian_step(v0, e0, last_t, now, f, b, g)
    assert torch.allclose(v, v_it, atol=1e-12)
    assert torch.allclose(e, e_it, atol=1e-10)


def test_recurring_pattern_accumulates_momentum_noise_cancels():
    b = torch.full((1,), 0.9, dtype=D)
    g = torch.full((1,), 0.95, dtype=D)
    f = torch.zeros(1, 4, dtype=D)
    f[0, 0] = 1.0
    v_rec, e_rec = torch.zeros(1, 4, dtype=D), torch.zeros(1, dtype=D)
    v_alt, e_alt = v_rec.clone(), e_rec.clone()
    for t in range(1, 30):
        v_rec, e_rec = newtonian_step(v_rec, e_rec, torch.tensor([t - 1]), t, f, b, g)
        v_alt, e_alt = newtonian_step(v_alt, e_alt, torch.tensor([t - 1]), t, f * (-1) ** t, b, g)
    assert velocity_norm(v_rec) > 5 * velocity_norm(v_alt)
    assert e_rec > 3 * e_alt


def test_cold_nodes_decay_and_friction_orders_decay():
    v = torch.randn(2, 4, dtype=D)
    e = torch.tensor([5.0, 5.0], dtype=D)
    b = torch.tensor([0.5, 0.95], dtype=D)
    g = torch.tensor([0.6, 0.995], dtype=D)  # EPHEMERAL vs SPEC
    t0 = torch.zeros(2, dtype=torch.int64)
    e_at = {k: current_energy(v, e, t0, k, b, g) for k in (1, 5, 200, 2000)}
    # Residual momentum keeps feeding energy for ~1/(1-beta) ticks (inertia), then friction wins.
    assert e_at[5][1] > e_at[1][1]
    assert e_at[2000][1] < e_at[200][1] < e_at[5][1] * 2
    assert (e_at[2000] < e_at[5]).all()
    assert e_at[200][0] < 1e-6  # high-friction ephemeral node is gone
    assert e_at[200][1] > 1e3 * e_at[200][0]


def test_surprise_force_norm_equals_geodesic_distance():
    x = torch.randn(8, 6, dtype=D) * 0.2
    z = torch.randn(8, 6, dtype=D) * 0.2
    f0, s = surprise_force(x, z)
    assert torch.allclose(velocity_norm(f0), s, atol=1e-8)
    assert torch.allclose(s, H.distance(x, z), atol=1e-12)
