from __future__ import annotations

from zt_memory.config import MemoryConfig, RouterConfig
from zt_memory.encoders import HashingEncoder
from zt_memory.harmony import extract_call_sites
from zt_memory.manager import NewtonianMemoryManager
from zt_memory.router import GeometricRouter


def test_extract_call_sites_ignores_builtins_and_finds_domain_calls():
    code = """def process_order(order_id, items):
    if len(items) == 0:
        raise ValueError("empty")
    subtotal = compute_subtotal(items)
    return apply_vat_rate(subtotal, rate=0.22)
"""
    sites = extract_call_sites(code, start_line=10)
    names = [s.callee for s in sites]
    assert "len" not in names and "ValueError" not in names
    assert names == ["compute_subtotal", "apply_vat_rate"]
    assert sites[0].line == 13
    assert sites[1].line == 14


def test_octave_separation_and_impedance_matched_resonance(tmp_path):
    repo = tmp_path / "repo"
    (repo / "src" / "pkg").mkdir(parents=True)
    (repo / "tests").mkdir(parents=True)

    (repo / "src" / "pkg" / "core.py").write_text(
        '''def define_model(kw_only=False, default=None):
    """Define a structured model with kw_only handling."""
    return _build_attrs(kw_only=kw_only, default=default)


def _build_attrs(kw_only=False, default=None):
    return {"kw_only": kw_only, "default": default}
''',
        encoding="utf-8",
    )
    (repo / "src" / "pkg" / "api.py").write_text(
        '''from .core import define_model

def create_endpoint_schema(spec):
    """Build endpoint schema using define_model."""
    return define_model(kw_only=True, default=spec)
''',
        encoding="utf-8",
    )
    test_lines = ["from src.pkg.core import define_model"]
    for i in range(8):
        test_lines.append(
            f'''def test_define_model_kw_only_case_{i}():
    """Test define_model kw_only default sentinel behavior case {i}."""
    res = define_model(kw_only=True, default={i})
    assert res["kw_only"] is True
'''
        )
    (repo / "tests" / "test_core.py").write_text("\n\n".join(test_lines), encoding="utf-8")

    cfg = MemoryConfig(repo_root=repo, storage_dir=tmp_path / "store", rank=16, calibration_samples=32)
    with NewtonianMemoryManager(cfg, HashingEncoder(dim=64)) as mgr:
        mgr.sync()
        rc = RouterConfig(dedup=True, diversity_lambda=0.15, focus_blocks=2, signature_extra=8)
        ctx = GeometricRouter(mgr, rc).route_to_compute_engine(
            "define_model kw_only default sentinel behavior", token_budget=500
        )
        loc = ctx.render_locations(max_tokens=300)
        assert "src/pkg/core.py" in loc.files
        assert loc.text.splitlines()[0].startswith("- src/pkg/")
        assert ctx.harmonic_field is not None
        assert any(cs.caller_path == "src/pkg/api.py" and cs.target_symbol == "define_model" for cs in ctx.harmonic_field.call_slices)
        assert any(rt.path == "tests/test_core.py" and rt.coupled_to == "define_model" for rt in ctx.harmonic_field.resonant_tests)
