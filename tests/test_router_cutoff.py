import pytest

from zt_memory import GeometricRouter, NodeKind, RouterConfig


@pytest.fixture
def brown(repo, make_manager):
    mgr = make_manager()
    mgr.sync()
    return mgr


def test_cutoff_stops_before_budget_but_keeps_minimum(brown):
    q = "database pool acquire connection"
    full = GeometricRouter(brown, RouterConfig(cutoff_alpha=0.0, reinforce=False)).route_to_compute_engine(q, token_budget=20_000)
    cut = GeometricRouter(brown, RouterConfig(cutoff_alpha=0.9, min_blocks=3, structural_headers=0, reinforce=False)) \
        .route_to_compute_engine(q, token_budget=20_000)
    assert len(cut.blocks) < len(full.blocks) and cut.token_estimate < full.token_estimate
    assert sum(len(b.node_ids) for b in cut.blocks) >= 3  # guaranteed minimum (blocks may merge)
    best = max(full.blocks, key=lambda b: b.score)
    assert any(b.node_ids == best.node_ids for b in cut.blocks)  # the top block always survives


def test_structural_headers_survive_the_cut(brown):
    cfg = RouterConfig(cutoff_alpha=0.99, min_blocks=1, structural_headers=3, reinforce=False)
    ctx = GeometricRouter(brown, cfg).route_to_compute_engine("handle request route health", token_budget=20_000)
    code_files = {b.path for b in ctx.blocks if b.kind == NodeKind.CHUNK}
    headers = {b.path for b in ctx.blocks if b.kind == NodeKind.FILE}
    assert headers and headers <= code_files


def test_ranked_files_is_a_unique_ranking(brown):
    ranked = GeometricRouter(brown, RouterConfig(reinforce=False)).ranked_files("slugify title")
    paths = [p for p, _ in ranked]
    assert paths[0] == "src/util/strings.py" and len(paths) == len(set(paths))
    assert [s for _, s in ranked] == sorted((s for _, s in ranked), reverse=True)


def test_recency_list_proposes_hot_chunks(brown, repo):
    path = repo / "src/util/strings.py"
    for i in range(4):  # make one unrelated file hot
        path.write_text(path.read_text().replace(f"# rev {i}", "") + f"\n# rev {i + 1}\n")
        brown.sync()
    q = "database pool acquire connection"
    plain = dict(GeometricRouter(brown, RouterConfig(reinforce=False, energy_weight=0.0)).ranked_files(q))
    hot = dict(GeometricRouter(brown, RouterConfig(reinforce=False, energy_weight=0.0, recency_weight=2.0)).ranked_files(q))
    assert "src/util/strings.py" in hot
    assert hot["src/util/strings.py"] / max(hot.values()) > plain.get("src/util/strings.py", 0.0) / max(plain.values())
