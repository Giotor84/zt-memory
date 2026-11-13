import numpy as np
import pytest

from zt_memory import NodeKind, Requirement

REQS = [
    Requirement("REST API exposing CRUD endpoints for invoices with pagination", key="api"),
    Requirement("Persist invoices in PostgreSQL through a repository layer with connection pooling", key="db"),
    Requirement("JWT authentication middleware protecting every endpoint", key="auth"),
]


def test_seed_requirements_idempotent_and_updatable(tmp_path, make_manager):
    mgr = make_manager(repo_root=tmp_path / "empty")
    ids = mgr.seed_requirements(REQS)
    assert len(set(ids)) == 3 and mgr.project_mode() == "greenfield"
    assert mgr.count_by_kind()[NodeKind.SPEC] == 3
    assert mgr.seed_requirements(REQS) == ids
    pos_before = mgr.positions([ids[2]])
    new = mgr.seed_requirements([Requirement("OAuth2 device flow login with refresh tokens", key="auth")])
    assert new == [ids[2]]
    assert mgr.record(ids[2]).payload.startswith("OAuth2")
    assert not np.allclose(mgr.positions([ids[2]]), pos_before)
    assert np.isclose(np.linalg.norm(mgr.positions([ids[2]])), mgr.radius_of(NodeKind.SPEC), atol=1e-5)


def test_generated_artifact_binds_and_heats_specs(tmp_path, make_manager):
    root = tmp_path / "empty"
    mgr = make_manager(repo_root=root)
    ids = mgr.seed_requirements(REQS)
    e_before = mgr.energies([ids[1]])[0]
    art = root / "app/repository.py"
    art.parent.mkdir(parents=True)
    art.write_text("class InvoiceRepository:\n    def __init__(self, pool):\n        self.pool = pool\n\n"
                   "    def list_invoices(self, page: int):\n        return []\n")
    rep = mgr.register_generated_artifact("app/repository.py", [ids[1]])
    assert rep.chunks_added >= 1
    edges = mgr.neighbors([ids[1]])
    file_id = mgr.record_by_key("file:app/repository.py").id
    assert any(e.kind == "implements" and e.dst == file_id for e in edges)
    assert mgr.energies([ids[1]])[0] > e_before * mgr.config.physics[NodeKind.SPEC].gamma ** 2


def test_decisions_and_ephemeral_pruning(tmp_path, make_manager):
    mgr = make_manager(repo_root=tmp_path / "empty")
    ids = mgr.seed_requirements(REQS)
    dec = mgr.record_decision("Use hexagonal architecture: domain core independent of FastAPI", related_ids=ids[:2])
    assert {e.dst for e in mgr.neighbors([dec]) if e.src == dec} == set(ids[:2])
    tmp = mgr.add_ephemeral("scratch: try uvicorn workers=4")
    assert mgr.prune(energy_floor=1e-3) == 0  # still hot
    for _ in range(60):
        mgr.reinforce([ids[0]], mgr.positions([ids[0]])[0], [1.0])
    assert mgr.prune(energy_floor=1e-3) == 1
    assert mgr.record(tmp) is None and mgr.record(dec) is not None
    with pytest.raises(ValueError):
        mgr.prune(kinds=frozenset({NodeKind.CHUNK}))
