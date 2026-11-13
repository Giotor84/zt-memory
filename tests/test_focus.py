"""Focused context: stub filter, clone removal (MMR among chunks only), bodies only where they matter."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from zt_memory.cli import main
from zt_memory.config import NodeKind, RouterConfig
from zt_memory.project import ProjectConfig
from zt_memory.router import GeometricRouter, _Scored

PRICING = '''VAT = 0.22


def apply_discount(price: float, percent: float) -> float:
    """Price after a percentage discount (0-100)."""
    total = price - price * percent
    total = round(total, 2)
    return total


def with_vat(price: float) -> float:
    """Price including VAT."""
    if price < 0:
        raise ValueError("negative price")
    gross = price * (1 + VAT)
    cents = round(gross * 100)
    if cents % 5:
        cents += 5 - cents % 5
    result = cents / 100
    assert result >= price
    return result


def discount_label(percent: float) -> str:
    """Human readable discount."""
    if percent <= 0:
        return "no discount"
    if percent >= 100:
        return "free"
    whole = int(percent)
    if whole == percent:
        text = f"-{whole}%"
    else:
        text = f"-{percent:.1f}%"
    return text.replace(".", ",")
'''
CLONE = PRICING.split("def with_vat")[0].replace("VAT = 0.22", "RATE = 0.22")  # same apply_discount, other module
STUB = "def apply_discount(price: float, percent: float) -> float: ...\ndef with_vat(price: float) -> float: ...\n"
TESTS = """from shop.pricing import apply_discount


def test_apply_discount():
    \"\"\"A percentage discount on round prices, on cents and at the edges.\"\"\"
    assert apply_discount(200, 10) == 180
    assert apply_discount(100, 0) == 100
    assert apply_discount(100, 100) == 0
    assert apply_discount(19.99, 50) == 9.99
    assert apply_discount(0, 30) == 0
"""
QUERY = "apply_discount returns a wrong price for a percentage discount"


@pytest.fixture
def shop(tmp_path, capsys):
    root = tmp_path / "shop_repo"
    (root / "shop").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "shop/__init__.py").write_text("")
    (root / "shop/pricing.py").write_text(PRICING)
    (root / "shop/pricing.pyi").write_text(STUB)
    (root / "shop/legacy_pricing.py").write_text(CLONE)
    (root / "shop/labels.pyi").write_text("def only_in_stub(x: int) -> str: ...\n")
    (root / "tests/test_pricing.py").write_text(TESTS)
    assert main(["init", "--repo", str(root), "--rank", "32"]) == 0
    cfg = ProjectConfig.load(root)
    cfg.memory.update(calibration_samples=128)
    cfg.save(overwrite=True)
    assert main(["sync", "--repo", str(root), "--quiet"]) == 0
    capsys.readouterr()
    return root


def _route(root, query=QUERY, **changes):
    pc = ProjectConfig.load(root)
    with pc.open_manager() as mgr:
        rc = replace(RouterConfig(reinforce=False), **changes)
        return GeometricRouter(mgr, rc).route_to_compute_engine(query, token_budget=4000)


def test_default_is_unchanged(shop):
    ctx = _route(shop)
    assert all(b.full for b in ctx.blocks)
    assert any(b.path == "shop/pricing.pyi" for b in ctx.blocks)  # today the stub can take a slot


def test_dedup_drops_the_stub_and_the_clone(shop):
    ctx = _route(shop, dedup=True, diversity_lambda=0.3)
    paths = [b.path for b in ctx.blocks if b.kind == NodeKind.CHUNK]
    assert "shop/pricing.pyi" not in paths  # the implementation is a candidate: the stub never goes in
    bodies = [b for b in ctx.blocks if b.full and "apply_discount" in b.text and "def apply_discount" in b.text]
    assert len(bodies) == 1  # one apply_discount body, not the copy in legacy_pricing.py
    assert all(b.full for b in ctx.blocks if not (b.path or "").endswith(".pyi"))  # B renders everything whole


def test_lone_stub_is_only_a_signature(shop):
    ctx = _route(shop, "only_in_stub label function", dedup=True)
    stub = [b for b in ctx.blocks if b.path == "shop/labels.pyi"]
    assert stub and not any(b.full for b in stub)


def test_focus_keeps_bodies_for_the_top_and_signatures_for_the_rest(shop):
    whole = _route(shop, dedup=True, diversity_lambda=0.3)
    focus = _route(shop, dedup=True, diversity_lambda=0.3, focus_blocks=1, focus_ratio=0.99)
    full = [b for b in focus.blocks if b.kind == NodeKind.CHUNK and b.full and not b.symbols[0].startswith("<")]  # one-line module headers stay whole: cheaper than a signature
    sigs = [b for b in focus.blocks if b.kind == NodeKind.CHUNK and not b.full]
    assert len(full) == 1 and "def apply_discount" in full[0].text and sigs
    tests = [b for b in focus.blocks if (b.path or "").startswith("tests/") and not b.symbols[0].startswith("<")]
    assert tests and not any(b.full for b in tests)  # tests never take a focus slot
    assert focus.token_estimate < whole.token_estimate
    text = focus.render_context()
    assert "signatures only" in text and "def with_vat(price: float) -> float:  — Price including VAT." in text
    assert "cents += 5 - cents % 5" not in text  # with_vat's body is not sent


def test_reinvestment_adds_signatures_within_budget(shop):
    plain = _route(shop, dedup=True, diversity_lambda=0.3, focus_blocks=1, focus_ratio=0.99)
    more = _route(shop, dedup=True, diversity_lambda=0.3, focus_blocks=1, focus_ratio=0.99, signature_extra=20)
    sig = lambda c: {(b.path, b.start_line) for b in c.blocks if not b.full}  # noqa: E731
    assert sig(plain) <= sig(more) and more.token_estimate <= 4000
    assert not any(b.path == "shop/pricing.pyi" for b in more.blocks)  # reinvesting never brings the stub back


def test_mmr_compares_chunks_with_chunks_only():
    """A file node and a chunk with the same direction: the chunk is not pushed down by the file (other level)."""
    v = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    kinds = [NodeKind.FILE, NodeKind.CHUNK, NodeKind.CHUNK]
    records = {i: SimpleNamespace(kind=k, path=f"f{i}.py", symbol=f"s{i}") for i, k in enumerate(kinds)}
    sc = _Scored("brownfield", np.zeros(2), np.arange(3), records, np.array([1.0, 0.9, 0.8]), np.zeros(3), np.zeros(3),
                 np.zeros(3), np.arange(3), -np.ones(3, dtype=np.int64), v)
    router = GeometricRouter.__new__(GeometricRouter)
    router.config = RouterConfig(diversity_lambda=0.5, dedup=True)
    assert router._mmr_order(sc, [0, 1, 2], 1.0, chosen=[]) == [0, 1, 2]
    sc.vectors = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])  # chunk 2 is a clone of chunk 1
    assert router._mmr_order(sc, [0, 1, 2], 1.0, chosen=[]) == [0, 1]
