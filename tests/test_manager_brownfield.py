import numpy as np
import pytest

from zt_memory import IncompatibleStoreError, NodeKind
from zt_memory.config import MEMORY_KINDS


def _chunk(mgr, rel, symbol):
    return mgr.record_by_key(f"chunk:{rel}::{symbol}")


def test_initial_ingest_builds_hierarchy(repo, make_manager):
    mgr = make_manager()
    rep = mgr.ingest_repository()
    counts = mgr.count_by_kind()
    assert rep.files_seen == 5 and counts[NodeKind.FILE] == 5
    assert counts[NodeKind.DIR] == 3  # src, src/app, src/util
    assert counts[NodeKind.CHUNK] == rep.chunks_added > 0
    assert mgr.project_mode() == "brownfield"
    rec = _chunk(mgr, "src/app/config.py", "parse_json_config")
    assert rec is not None and mgr.read_chunk(rec).startswith("def parse_json_config")
    # hierarchy is radial: chunk at 0.90, file at 0.80, dir at 0.55, root at 0
    file_rec = mgr.record(rec.parent_id)
    norms = np.linalg.norm(mgr.positions([rec.id, file_rec.id, file_rec.parent_id, mgr.root_id]), axis=1)
    assert np.allclose(norms, [0.90, 0.80, 0.55, 0.0], atol=1e-5)


def test_noop_resync_has_no_surprise(repo, make_manager):
    mgr = make_manager()
    first = mgr.ingest_repository()
    again = mgr.ingest_repository()
    assert again.chunks_added == again.chunks_updated == again.chunks_removed == 0
    assert again.chunks_unchanged == first.chunks_added
    assert again.surprise_events == 0


def test_modified_symbol_gets_surprise_and_energy(repo, make_manager):
    mgr = make_manager()
    mgr.ingest_repository()
    for _ in range(3):  # let initial-ingest energy cool down a little
        mgr.reinforce([mgr.root_id], np.ones(32, np.float32), [0.0])
    target = repo / "src/app/config.py"
    target.write_text(target.read_text().replace('startswith("postgres")', 'startswith(("postgres", "mysql", "sqlite"))'))
    rep = mgr.ingest_files(["src/app/config.py"])
    assert rep.chunks_updated == 1 and rep.chunks_added == 0
    assert rep.changed_symbols == ["~src/app/config.py::validate_config"]
    assert rep.surprise_max > 0
    changed = _chunk(mgr, "src/app/config.py", "validate_config").id
    untouched = _chunk(mgr, "src/app/config.py", "parse_json_config").id
    e = mgr.energies([changed, untouched])
    assert e[0] > e[1]


def test_line_shift_updates_spans_without_events(repo, make_manager):
    mgr = make_manager()
    mgr.ingest_repository()
    before = _chunk(mgr, "src/util/strings.py", "camel_to_snake")
    target = repo / "src/util/strings.py"
    text = target.read_text()
    target.write_text(text.replace("def slugify_title", "def slugify_title", 1).replace(
        "def camel_to_snake", "\n\n\ndef camel_to_snake", 1))
    rep = mgr.ingest_files([target])  # absolute paths are accepted
    after = _chunk(mgr, "src/util/strings.py", "camel_to_snake")
    assert rep.chunks_updated == 0 and rep.chunks_added == 0
    assert after.id == before.id and after.start_line == before.start_line + 3
    assert mgr.read_chunk(after) is not None


def test_removed_symbol_and_file(repo, make_manager):
    mgr = make_manager()
    mgr.ingest_repository()
    target = repo / "src/app/http_server.py"
    text = target.read_text()
    target.write_text(text[: text.index("def render_template")])
    rep = mgr.ingest_files(["src/app/http_server.py"])
    assert rep.chunks_removed == 1 and _chunk(mgr, "src/app/http_server.py", "render_template") is None

    (repo / "src/util/strings.py").unlink()
    rep = mgr.ingest_repository()
    assert rep.files_removed == 1
    assert mgr.record_by_key("file:src/util/strings.py") is None
    assert mgr.record_by_key("dir:src/util") is None  # empty dir pruned
    assert mgr.record_by_key("dir:src/app") is not None
    ids, _ = mgr.search(mgr.embed_query("slugify title url"), 50, MEMORY_KINDS)
    recs = mgr.records([i for i in ids[0] if i >= 0])
    assert all(r.path != "src/util/strings.py" for r in recs.values())


def test_momentum_accumulates_on_recurring_edits(repo, make_manager):
    mgr = make_manager()
    mgr.ingest_repository()
    path = repo / "src/app/db.py"
    for i in range(5):
        path.write_text(path.read_text().replace(f"size: int = {4 + i}", f"size: int = {5 + i}"))
        mgr.ingest_files(["src/app/db.py"])
    hot = _chunk(mgr, "src/app/db.py", "DatabasePool.__init__").id
    cold = _chunk(mgr, "src/util/strings.py", "slugify_title").id
    e_hot, e_cold = mgr.energies([hot, cold])
    assert e_hot > e_cold
    top = [i for i, _ in mgr.hottest(3, frozenset({NodeKind.CHUNK}))]
    assert hot in top


def test_persistence_roundtrip(repo, make_manager, tmp_path):
    mgr = make_manager()
    mgr.ingest_repository()
    ids = [r for r in mgr.meta.ids_by_kinds([NodeKind.CHUNK])]
    energies = mgr.energies(ids)
    counts = mgr.count_by_kind()
    clock = mgr.clock
    mgr.close()
    again = make_manager()
    assert again.clock == clock and again.count_by_kind() == counts
    assert np.allclose(again.energies(ids), energies, rtol=1e-5)
    assert again.ingest_repository().surprise_events == 0


def test_incompatible_store_rejected(repo, make_manager):
    first = make_manager()
    first.ingest_repository()
    first.close()
    with pytest.raises(IncompatibleStoreError):
        make_manager(rank=16)


def test_tiny_ram_budget_pages_to_ssd(repo, make_manager):
    # capacity ~32 entries for ~25 nodes plus repeated syncs -> forced paging
    mgr = make_manager(ram_budget_bytes=32 * (32 * 4 + 126))
    for i in range(6):
        p = repo / f"src/gen/mod_{i}.py"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n\n".join(f"def generated_{i}_{j}(x):\n    return x * {j}" for j in range(8)) + "\n")
    mgr.ingest_repository()
    st = mgr.ws.stats()
    assert st.resident <= st.capacity == 32
    assert st.evictions > 0
    ids = mgr.meta.ids_by_kinds([NodeKind.CHUNK])
    e_live = mgr.energies(ids)
    mgr.flush()
    mgr.ws.clear()
    assert np.allclose(mgr.energies(ids), e_live, rtol=1e-5)


def test_hnsw_backend_agrees_with_scan(repo, make_manager, tmp_path):
    pytest.importorskip("hnswlib")
    scan = make_manager()
    scan.ingest_repository()
    scan.close()
    queries = ["parse json configuration", "database connection pool", "render html template", "slug title"]
    results = {}
    for backend in ("hnsw", "scan"):
        mgr = make_manager(index_backend=backend)
        results[backend] = [mgr.search(mgr.embed_query(q), 3, MEMORY_KINDS)[0][0][0] for q in queries]
        mgr.close()
    assert results["hnsw"] == results["scan"]


def test_inserting_a_symbol_does_not_rekey_neighbours(repo, make_manager):
    mgr = make_manager()
    mgr.ingest_repository()
    path = repo / "src/app/config.py"
    ids_before = {r.symbol: r.id for r in mgr.meta.chunks_of_file("src/app/config.py")}
    text = path.read_text()
    path.write_text(text.replace("def parse_json_config", "def load_env(name: str) -> str:\n    return name\n\n\ndef parse_json_config", 1))
    rep = mgr.ingest_files(["src/app/config.py"])
    assert (rep.chunks_added, rep.chunks_updated, rep.chunks_removed) == (1, 0, 0)
    ids_after = {r.symbol: r.id for r in mgr.meta.chunks_of_file("src/app/config.py")}
    assert {k: ids_after[k] for k in ids_before} == ids_before
    # reordering symbols is free as well
    t = path.read_text()
    a, b = t.index("def parse_json_config"), t.index("def validate_config")
    path.write_text(t[:a] + t[b:].rstrip("\n") + "\n\n\n" + t[a:b].rstrip("\n") + "\n")
    rep = mgr.ingest_files(["src/app/config.py"])
    assert (rep.chunks_added, rep.chunks_updated, rep.chunks_removed, rep.surprise_events) == (0, 0, 0, 0)


def test_new_stores_use_uncentered_svd(repo, make_manager):
    mgr = make_manager()
    mgr.sync()
    assert mgr.projector.frozen and not mgr.projector.center
