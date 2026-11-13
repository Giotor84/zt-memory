import re

import pytest

from zt_memory import GeometricRouter, HashingEncoder, NodeKind, Requirement, RouterConfig, route_to_compute_engine
from zt_memory.engine_client import parse_generated_files, write_generated_files


@pytest.fixture
def brown(repo, make_manager):
    mgr = make_manager()
    mgr.ingest_repository()
    return mgr


def test_brownfield_routing_returns_exact_relevant_spans(brown):
    ctx = route_to_compute_engine("fix parse json config when the file is missing", brown)
    assert ctx.mode == "brownfield"
    code = [b for b in ctx.blocks if b.kind == NodeKind.CHUNK]
    assert code and any("def parse_json_config" in b.text for b in code)
    best = max(code, key=lambda b: b.score)
    assert "parse_json_config" in best.title
    prompt = ctx.render(line_numbers=True)
    assert "MODE: BROWNFIELD" in prompt and "unified diffs" in prompt and "src/app/config.py" in prompt
    msgs = ctx.to_chat_messages()
    assert [m["role"] for m in msgs] == ["system", "user"]


def test_token_budget_is_respected(brown):
    router = GeometricRouter(brown, RouterConfig(template_overhead_tokens=64))
    ctx = router.route_to_compute_engine("database connection pool acquire release", token_budget=300)
    assert ctx.blocks and ctx.token_estimate <= 300
    big = router.route_to_compute_engine("database connection pool acquire release", token_budget=20_000)
    assert len(big.blocks) >= len(ctx.blocks)


def test_routing_reinforces_selected_nodes(brown):
    rec = brown.record_by_key("chunk:src/app/db.py::DatabasePool.acquire_connection")
    e0 = brown.energies([rec.id])[0]
    ctx = route_to_compute_engine("database pool acquire connection", brown)
    assert rec.id in ctx.node_ids
    e1 = brown.energies([rec.id])[0]
    gamma = brown.config.physics[NodeKind.CHUNK].gamma
    assert e1 > e0 * gamma  # without reinforcement it would only have decayed


def test_expansion_adds_seed_file_headers(brown):
    router = GeometricRouter(brown)
    cand = router._collect_candidates(brown.embed_query("render html template"))
    kinds = {r.kind for r in brown.records(cand).values()}
    assert NodeKind.FILE in kinds and NodeKind.CHUNK in kinds


def test_stale_file_is_resynced_before_injection(brown, repo):
    target = repo / "src/app/config.py"
    target.write_text("# header comment\n" + target.read_text().replace("Parse a JSON", "Parse a strict JSON"))
    ctx = route_to_compute_engine("parse json config", brown)
    block = next(b for b in ctx.blocks if b.kind == NodeKind.CHUNK and "parse_json_config" in b.title)
    assert "Parse a strict JSON" in block.text
    assert target.read_text().splitlines()[block.start_line - 1].startswith("def parse_json_config")


def test_greenfield_loop(tmp_path, make_manager):
    root = tmp_path / "proj"
    mgr = make_manager(repo_root=root)
    ids = mgr.seed_requirements([
        Requirement("Invoice REST API with pagination and filtering", key="api"),
        Requirement("PostgreSQL invoice repository with connection pooling", key="db"),
    ])
    ctx = route_to_compute_engine("scaffold the invoice repository and database pooling", mgr)
    assert ctx.mode == "greenfield"
    assert any(b.kind == NodeKind.SPEC for b in ctx.blocks)
    assert "path=<relative/path>" in ctx.render()

    model_output = (
        "Here you go.\n```python path=app/db/repository.py\nclass InvoiceRepository:\n"
        "    def __init__(self, pool):\n        self.pool = pool\n```\n"
        "```python path=app/api/routes.py\ndef list_invoices(page: int = 1):\n    return []\n```\n"
    )
    files = parse_generated_files(model_output)
    assert [f.path for f in files] == ["app/db/repository.py", "app/api/routes.py"]
    written = write_generated_files(files, root)
    mgr.register_generated_artifact(written[0], [ids[1]])
    mgr.register_generated_artifact(written[1], [ids[0]])
    ctx2 = route_to_compute_engine("add filtering to the invoice repository", mgr)
    assert set(ctx2.file_map) == {"app/db/repository.py", "app/api/routes.py"}
    assert any(b.path == "app/db/repository.py" for b in ctx2.blocks)


def test_generated_paths_cannot_escape_root(tmp_path):
    files = parse_generated_files("```python path=../evil.py\nx = 1\n```\n")
    with pytest.raises(ValueError):
        write_generated_files(files, tmp_path)


HYBRID_FILES = {
    "src/app/errors.py": '''
        def classify_failure(code: str) -> str:
            """Map low-level failure codes onto user-facing categories."""
            if code == "ZX_HANDSHAKE_9931":
                return "tls"
            return "unknown"
    ''',
}


class CodeBlindEncoder(HashingEncoder):
    """Simulates a semantic embedding that cannot resolve rare literal codes (the failure mode
    hybrid search must cover): tokens with digits or ALLCAPS_SNAKE shape are invisible to it."""

    @property
    def fingerprint(self) -> str:
        return "code-blind:" + super().fingerprint

    def _tokens(self, text):
        return super()._tokens(re.sub(r"\b[A-Z][A-Z0-9]*_[A-Z0-9_]+\b", " ", text))


@pytest.fixture
def hybrid(repo, make_manager):
    from conftest import write_repo

    write_repo(repo, HYBRID_FILES)
    mgr = make_manager(encoder=CodeBlindEncoder(dim=512))
    mgr.ingest_repository()
    return mgr


def _rank_of(ctx, symbol):
    ordered = sorted((b for b in ctx.blocks if b.kind == NodeKind.CHUNK), key=lambda b: -b.score)
    return next((i for i, b in enumerate(ordered) if f"[{symbol}]" in b.title), None)


def test_hybrid_fusion_surfaces_exact_error_codes(hybrid):
    query = "users report ZX_HANDSHAKE_9931 after upgrading"
    fused = GeometricRouter(hybrid).route_to_compute_engine(query, token_budget=4000)
    geo_only = GeometricRouter(hybrid, RouterConfig(lexical_weight=0.0)).route_to_compute_engine(query, token_budget=4000)
    r_fused, r_geo = _rank_of(fused, "classify_failure"), _rank_of(geo_only, "classify_failure")
    assert r_fused == 0
    assert r_geo is None or r_geo > r_fused
    block = next(b for b in fused.blocks if "classify_failure" in b.title)
    assert block.lex_rank == 0 and block.geo_rank is not None


def test_lexical_hits_bring_structural_context(hybrid):
    router = GeometricRouter(hybrid)
    hits = hybrid.lexical_search("ZX_HANDSHAKE_9931", 4)
    cand = router._collect_candidates(hybrid.embed_query("ZX_HANDSHAKE_9931"), hits)
    recs = hybrid.records(cand)
    assert any(r.kind == NodeKind.FILE and r.path == "src/app/errors.py" for r in recs.values())


def test_multilanguage_repo_routes_qualified_symbols(repo, make_manager):
    pytest.importorskip("tree_sitter_typescript")
    from conftest import write_repo

    write_repo(repo, {
        "web/cart.ts": '''
            export class CartService {
              applyCoupon(code: string): number {
                return code.length;
              }

              checkout(): void {}
            }
        ''',
    })
    mgr = make_manager()
    mgr.ingest_repository()
    ctx = route_to_compute_engine("bug in CartService.applyCoupon with empty coupon codes", mgr)
    best = max((b for b in ctx.blocks if b.kind == NodeKind.CHUNK), key=lambda b: b.score)
    assert "[CartService.applyCoupon]" in best.title and best.path == "web/cart.ts"
