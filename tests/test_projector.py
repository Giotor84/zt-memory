import pytest
import torch

from zt_memory.projector import ProjectorStateError, SubspaceProjector


def _low_rank(n: int, d: int = 64, r: int = 8, noise: float = 0.01, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    basis = torch.linalg.qr(torch.randn(d, r, generator=g))[0]
    z = torch.randn(n, r, generator=g) * torch.linspace(5, 1, r)
    return z @ basis.T + noise * torch.randn(n, d, generator=g) + 3.0


def _proj_matrix(p: SubspaceProjector) -> torch.Tensor:
    b = p._components[p.drop_top : p.n_target]
    return b.T @ b


def test_svd_captures_low_rank_signal():
    x = _low_rank(512)
    p = SubspaceProjector(8)
    p.fit(x)
    assert p.explained_variance_ratio().sum() > 0.99


def test_incremental_equals_batch():
    x = _low_rank(600)
    full = SubspaceProjector(8)
    full.fit(x)
    inc = SubspaceProjector(8)
    for i in range(0, 600, 100):
        inc.partial_fit(x[i : i + 100])
    assert torch.allclose(_proj_matrix(full), _proj_matrix(inc), atol=1e-4)
    assert torch.allclose(full._mean, inc._mean, atol=1e-10)


def test_freeze_save_load(tmp_path):
    x = _low_rank(256)
    p = SubspaceProjector(8, drop_top=1)
    p.fit(x)
    p.freeze()
    with pytest.raises(ProjectorStateError):
        p.partial_fit(x)
    p.save(tmp_path / "p.pt")
    q = SubspaceProjector.load(tmp_path / "p.pt")
    assert q.frozen and q.fingerprint == p.fingerprint
    assert torch.allclose(p.transform(x[:10]), q.transform(x[:10]))
    assert p.transform(x[:3]).shape == (3, 8)


def test_freeze_requires_enough_components():
    p = SubspaceProjector(16)
    p.partial_fit(_low_rank(8, d=64))
    with pytest.raises(ProjectorStateError):
        p.freeze()
