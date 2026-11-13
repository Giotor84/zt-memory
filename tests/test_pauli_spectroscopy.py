"""Unit tests for Branch 3 & 4: Pauli Exclusion Principle (`pauli.py`) and Structural Spectroscopy (`spectroscopy.py`)."""
from __future__ import annotations

from pathlib import Path

from zt_memory.chunking import chunk_source
from zt_memory.cli import main
from zt_memory.mcp_server import ZtTools
from zt_memory.pauli import PauliSessionCache
from zt_memory.project import ProjectConfig
from zt_memory.spectroscopy import (
    extract_spectroscopic_entries,
    render_spectroscopic_toc,
    spectroscopic_outlines_for_span,
)


SAMPLE_MD = """# Architettura Pipeline ACME

Testo introduttivo della pipeline.

```python
# Questo commento in un blocco di codice NON e' un titolo Markdown
def helper():
    return 42
```

## 1. Ingestione Sentenze (Tier A)
Dettagli sull'estrazione dei PDF e anonimizzazione.
Riga aggiuntiva per superare la soglia minima.

## 2. Motore SageMaker (Tier B)
Chiamate a Bedrock con MiniMax M2.5 e Qwen3-235B.
Riga aggiuntiva per il secondo blocco.
"""

SAMPLE_JS = """// Dashboard Controller
export class OrientaApp {
    constructor(config) {
        this.config = config;
        this.filters = {};
    }

    renderTable(rows) {
        const html = rows.map(r => `<tr><td>${r.id}</td></tr>`).join("");
        return html;
    }
}

export async function fetchMetrics(endpoint) {
    const res = await fetch(endpoint);
    return await res.json();
}

const computePolarityScore = (cred, deb) => {
    const total = (cred || 0) + (deb || 0);
    return total > 0 ? cred / total : 0.5;
};
"""

SAMPLE_HTML = """<!DOCTYPE html>
<html>
<head>
<style>
  body { background: #003087; color: #fff; }
  .accent { color: #F47920; }
</style>
</head>
<body>
  <div id="header-bar">Header</div>
  <section id="tab-overview">
    <h1>Overview</h1>
  </section>
  <div id="modal-detail">Modal</div>
  <script>
    function initDashboard(data) {
        console.log("init", data.length);
        return true;
    }

    const applyMandanteFilter = (mandante) => {
        return mandante ? mandante.toUpperCase() : "ALL";
    };
  </script>
</body>
</html>
"""


def test_markdown_spectroscopy_ignores_fenced_comments():
    entries = extract_spectroscopic_entries("docs/STATE.md", SAMPLE_MD)
    symbols = [e.symbol for e in entries]
    assert symbols == [
        "§# Architettura Pipeline ACME",
        "§## 1. Ingestione Sentenze (Tier A)",
        "§## 2. Motore SageMaker (Tier B)",
    ]
    chunks = chunk_source("docs/STATE.md", SAMPLE_MD, min_lines=3, max_lines=60)
    assert any(c.symbol.startswith("§## 1. Ingestione Sentenze") for c in chunks)
    assert not any(c.symbol.startswith("<block:") for c in chunks)


def test_javascript_stdlib_spectroscopy():
    entries = extract_spectroscopic_entries("app/LEGAL_APP.js", SAMPLE_JS)
    symbols = [e.symbol for e in entries]
    assert "OrientaApp" in symbols
    assert "OrientaApp.renderTable" in symbols
    assert "fetchMetrics" in symbols
    assert "computePolarityScore" in symbols

    # Verify on-the-fly spectroscopic outline resolution for legacy <block:...> spans
    outlines = spectroscopic_outlines_for_span("app/LEGAL_APP.js", SAMPLE_JS, 1)
    assert any("OrientaApp" in o.signature for o in outlines)
    assert any("fetchMetrics" in o.signature for o in outlines)


def test_html_dashboard_spectroscopy_extracts_style_dom_and_script_functions():
    entries = extract_spectroscopic_entries("dash/preview.html", SAMPLE_HTML)
    symbols = [e.symbol for e in entries]
    assert "style#1" in symbols
    assert "dom#header-bar" in symbols
    assert "dom#tab-overview" in symbols
    assert "dom#modal-detail" in symbols
    assert "script.initDashboard" in symbols
    assert "script.applyMandanteFilter" in symbols

    toc = render_spectroscopic_toc("dash/preview.html", SAMPLE_HTML)
    assert "SPECTROSCOPIC TOC" in toc
    assert "script.initDashboard" in toc
    assert "dom#tab-overview" in toc


def test_pauli_exclusion_unchanged_partial_and_wavefront_delta(tmp_path: Path):
    cache = PauliSessionCache(tmp_path / "pauli_session.json")
    lines = [f"def fn_{i}(x: int) -> int:\n    return x + {i}\n" for i in range(1, 25)]
    text_v1 = "\n".join(lines)
    total = len(text_v1.splitlines())

    # Turn 1: Initial read -> full emission
    r1 = cache.read_with_refraction("src/mod.py", text_v1, 1, total, total, lang="python")
    assert r1.status == "full"
    assert r1.lines_saved == 0

    # Turn 2: Exact re-read of unchanged file -> Pauli Exclusion Stub (-95% lines)
    r2 = cache.read_with_refraction("src/mod.py", text_v1, 1, total, total, lang="python")
    assert r2.status == "unchanged"
    assert "[PAULI_EXCLUSION: UNCHANGED]" in r2.rendered_block
    assert r2.lines_saved >= total - 4

    # Turn 3: Forced full read with force_full=True bypasses Pauli Exclusion
    r3 = cache.read_with_refraction("src/mod.py", text_v1, 1, total, total, lang="python", force_full=True)
    assert r3.status == "full"

    # Turn 4: Mid-session edit on disk -> Pauli Wavefront Delta (unified diff instead of full dump)
    mod_lines = text_v1.splitlines()
    mod_lines[10] = "    return x * 999  # mid-session bugfix"
    text_v2 = "\n".join(mod_lines)
    r4 = cache.read_with_refraction("src/mod.py", text_v2, 1, total, total, lang="python")
    assert r4.status == "delta"
    assert "[PAULI_WAVEFRONT_DELTA" in r4.rendered_block
    assert "+    return x * 999  # mid-session bugfix" in r4.rendered_block
    assert r4.lines_saved > 20


def test_mcp_read_spans_toc_and_pauli_focus_promotion(tmp_path: Path, capsys):
    root = tmp_path / "pauli_repo"
    (root / "src").mkdir(parents=True)
    code_a = "\n".join(
        [
            "def compute_alpha(price: float, rate: float) -> float:",
            '    """Alpha pricing calculation."""',
            *[f"    step_{i} = price * (1.0 + rate * 0.{i})" for i in range(1, 18)],
            "    return step_17",
            "",
            "def compute_beta(price: float, rate: float) -> float:",
            '    """Beta fallback pricing calculation."""',
            *[f"    val_{i} = price - (rate * {i})" for i in range(1, 18)],
            "    return val_17",
        ]
    )
    (root / "src/engine.py").write_text(code_a, encoding="utf-8")
    (root / "src/dashboard.html").write_text(SAMPLE_HTML, encoding="utf-8")

    assert main(["init", "--repo", str(root), "--rank", "32"]) == 0
    cfg = ProjectConfig.load(root)
    cfg.memory.update(calibration_samples=64)
    cfg.router.update(preset="api", focus_blocks=1, focus_ratio=0.99)
    cfg.save(overwrite=True)
    assert main(["sync", "--repo", str(root), "--quiet"]) == 0
    capsys.readouterr()

    tools = ZtTools(ProjectConfig.load(root), default_mode="focus", read_span=True)
    try:
        # 1. Test `path:toc` over MCP `zt_read_span`
        toc_out = tools.read_spans(["src/dashboard.html:toc"])
        assert "SPECTROSCOPIC TOC" in toc_out
        assert "script.initDashboard" in toc_out

        # 2. Test Pauli Exclusion on `zt_read_span` across turns
        span1 = tools.read_spans(["src/engine.py:L1-20"])
        assert "step_17 = price" in span1
        span2 = tools.read_spans(["src/engine.py:L1-20"])
        assert "[PAULI_EXCLUSION: UNCHANGED]" in span2
        assert "pauli refracted:" in span2

        # 3. Test Pauli refraction in `query_context(mode="focus")`
        tools._pauli.reset()
        q1 = tools.query_context("compute_beta fallback pricing calculation", mode="focus")
        assert "val_17 = price" in q1  # compute_beta body is shown in full on turn 1
        q2 = tools.query_context("compute_beta fallback pricing calculation", mode="focus")
        assert "[PAULI_IN_CONTEXT @ turn #1]" in q2  # compute_beta is refracted on turn 2!
        assert "val_17 = price" not in q2  # duplicate full body suppressed on turn 2!
    finally:
        tools.close()
