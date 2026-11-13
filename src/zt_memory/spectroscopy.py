"""Harmonic Structural Spectroscopy for Non-Python & Monolithic Files (.md, .html, .js, .ts).

Solves Finding C from the 152-session Forensic Token-Waste Study:
- Monolithic Markdown state/KB files (`project_baseline_state.md`, `KNOWLEDGE_BASE.md`),
  JavaScript apps (`LEGAL_APP.js`), and self-contained HTML dashboards (`preview_report.html`,
  `template.html`) were previously chunked as anonymous `<block:hash>` spans when `tree-sitter` was absent,
  penalized by `nonsymbol_chunk_weight = 0.55`, and stripped of `# Heading` lines by `outline.py`.

This module provides zero-dependency stdlib spectroscopic extractors that decompose:
1. Markdown (`.md`, `.markdown`, `.rst`) into nested section heading trees (`§ Heading`).
2. JavaScript / TypeScript (`.js`, `.jsx`, `.mjs`, `.cjs`, `.ts`, `.tsx`) into functions, classes, methods,
   and arrow-function bindings (`function render`, `Class.method`, `const handler = (...) =>`).
3. HTML Dashboards (`.html`, `.htm`) into DOM landmark containers (`dom#id`), `<style>` blocks, and
   embedded `<script>` functions with exact line numbers in the parent HTML file.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Sequence

from .outline import Outline
from .syntax import SymbolNode

_MD_EXTENSIONS = frozenset({".md", ".markdown", ".mdown", ".mkd"})
_HTML_EXTENSIONS = frozenset({".html", ".htm", ".xhtml"})
_JS_TS_EXTENSIONS = frozenset({".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx", ".mts", ".cts"})
_SPECTROSCOPIC_EXTENSIONS = _MD_EXTENSIONS | _HTML_EXTENSIONS | _JS_TS_EXTENSIONS

_MD_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")

# JS/TS stdlib spectroscopic patterns
_JS_CLASS_RE = re.compile(
    r"^\s*(?:export\s+(?:default\s+)?)?(?:abstract\s+)?class\s+([A-Za-z_$][\w$]*)"
)
_JS_FUNC_RE = re.compile(
    r"^\s*(?:export\s+(?:default\s+)?)?(?:async\s+)?function\s*\*?\s*([A-Za-z_$][\w$]*)\s*[<(]"
)
_JS_ARROW_OR_FN_VAR_RE = re.compile(
    r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*(?::\s*[^=]+)?=\s*"
    r"(?:async\s+)?(?:function\b|\([^)]*\)\s*=>|[A-Za-z_$][\w$]*\s*=>)"
)
_JS_OBJ_ASSIGN_FN_RE = re.compile(
    r"^\s*(?:window|globalThis|self|[A-Za-z_$][\w$]*)\s*\.\s*([A-Za-z_$][\w$]*)\s*=\s*"
    r"(?:async\s+)?(?:function\b|\([^)]*\)\s*=>)"
)
_JS_METHOD_RE = re.compile(
    r"^\s*(?:(?:public|private|protected|static|async|readonly|override|get|set)\s+)*"
    r"([A-Za-z_$][\w$]*)\s*\([^;]*\)\s*(?::\s*[^{;]+)?\{"
)
_JS_CONTROL_KEYWORDS = frozenset({
    "if", "for", "while", "switch", "catch", "with", "return", "sizeof", "typeof", "delete", "await", "yield"
})

# HTML landmark patterns
_HTML_SCRIPT_OPEN_RE = re.compile(r"<script\b[^>]*>", re.IGNORECASE)
_HTML_SCRIPT_CLOSE_RE = re.compile(r"</script\s*>", re.IGNORECASE)
_HTML_STYLE_OPEN_RE = re.compile(r"<style\b[^>]*>", re.IGNORECASE)
_HTML_STYLE_CLOSE_RE = re.compile(r"</style\s*>", re.IGNORECASE)
_HTML_ID_LANDMARK_RE = re.compile(
    r"<(section|main|article|header|footer|nav|aside|dialog|template|form|table|div)\b[^>]*\bid=[\"']([^\"']+)[\"'][^>]*>",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SpectroscopicEntry:
    symbol: str
    start_line: int
    end_line: int
    sig_line: int
    signature: str
    kind: str  # "md_section", "js_fn", "js_class", "html_dom", "html_script", "html_style"


def is_spectroscopic_path(rel_path: str | None) -> bool:
    if not rel_path:
        return False
    return PurePosixPath(rel_path).suffix.lower() in _SPECTROSCOPIC_EXTENSIONS


def is_markdown_path(rel_path: str | None) -> bool:
    if not rel_path:
        return False
    return PurePosixPath(rel_path).suffix.lower() in _MD_EXTENSIONS


# ------------------------------------------------------------------------------------------
# 1. Markdown Structural Spectroscopy
# ------------------------------------------------------------------------------------------
def _clean_md_title(raw: str) -> str:
    clean = re.sub(r"`([^`]+)`", r"\1", raw)
    clean = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", clean)
    clean = " ".join(clean.split()).strip()
    return clean[:90] if len(clean) > 90 else clean


def markdown_entries(lines: Sequence[str]) -> list[SpectroscopicEntry]:
    """Extract hierarchical Markdown headings outside fenced code blocks."""
    headings: list[tuple[int, int, str, str]] = []  # (line_1based, level, title, raw_line)
    in_fence = False
    fence_marker = ""
    for idx, line in enumerate(lines, start=1):
        m_fence = _MD_FENCE_RE.match(line)
        if m_fence:
            marker = m_fence.group(1)[:3]
            if not in_fence:
                in_fence = True
                fence_marker = marker
            elif line.lstrip().startswith(fence_marker):
                in_fence = False
                fence_marker = ""
            continue
        if in_fence:
            continue
        m_head = _MD_HEADING_RE.match(line)
        if m_head:
            level = len(m_head.group(1))
            title = _clean_md_title(m_head.group(2))
            if title:
                headings.append((idx, level, title, line.strip()))

    if not headings:
        return []

    total = len(lines)
    entries: list[SpectroscopicEntry] = []
    for i, (line_no, level, title, raw_line) in enumerate(headings):
        # Section ends right before the next heading of any level (for flat non-overlapping chunks)
        end_line = headings[i + 1][0] - 1 if i + 1 < len(headings) else total
        while end_line > line_no and not lines[end_line - 1].strip():
            end_line -= 1
        sym = f"§{'#' * level} {title}"
        entries.append(
            SpectroscopicEntry(
                symbol=sym,
                start_line=line_no,
                end_line=max(line_no, end_line),
                sig_line=line_no,
                signature=raw_line[:180],
                kind="md_section",
            )
        )
    return entries


def markdown_symbols(text: str) -> list[SymbolNode] | None:
    lines = text.splitlines()
    entries = markdown_entries(lines)
    if not entries:
        return None
    return [
        SymbolNode(
            name=e.symbol,
            start=e.start_line,
            end=e.end_line,
            sig_line=e.sig_line,
            members=[],
            container=False,
        )
        for e in entries
    ]


# ------------------------------------------------------------------------------------------
# 2. JavaScript / TypeScript Stdlib Spectroscopy (when tree-sitter is absent)
# ------------------------------------------------------------------------------------------
def _find_brace_end(lines: Sequence[str], start_idx: int, max_scan: int = 800) -> int:
    """Given 0-based `start_idx`, track `{` and `}` outside simple string literals to find the closing line (1-based)."""
    depth = 0
    opened = False
    in_str: str | None = None
    limit = min(len(lines), start_idx + max_scan)
    for k in range(start_idx, limit):
        ln = lines[k]
        i = 0
        n = len(ln)
        while i < n:
            ch = ln[i]
            if in_str:
                if ch == "\\":
                    i += 2
                    continue
                if ch == in_str:
                    in_str = None
                i += 1
                continue
            if ch in ("'", '"', "`"):
                in_str = ch
                i += 1
                continue
            if ch == "/" and i + 1 < n and ln[i + 1] == "/":
                break  # line comment
            if ch == "{":
                depth += 1
                opened = True
            elif ch == "}":
                depth -= 1
                if opened and depth <= 0:
                    return k + 1
            i += 1
        if not opened and k > start_idx + 4 and ln.rstrip().endswith(";"):
            return k + 1
    return min(len(lines), start_idx + 25)


def js_entries(lines: Sequence[str], *, line_offset: int = 0, prefix: str = "") -> list[SpectroscopicEntry]:
    """Extract JS/TS classes, functions, arrow functions, and methods from `lines`."""
    out: list[SpectroscopicEntry] = []
    n = len(lines)
    i = 0
    while i < n:
        ln = lines[i]
        stripped = ln.strip()
        if not stripped or stripped.startswith(("//", "/*", "*")):
            i += 1
            continue

        m_cls = _JS_CLASS_RE.match(ln)
        if m_cls:
            cls_name = m_cls.group(1)
            end_1 = _find_brace_end(lines, i, max_scan=1500)
            qual_cls = f"{prefix}{cls_name}"
            out.append(
                SpectroscopicEntry(
                    symbol=qual_cls,
                    start_line=line_offset + i + 1,
                    end_line=line_offset + end_1,
                    sig_line=line_offset + i + 1,
                    signature=stripped.rstrip("{ ").strip()[:180],
                    kind="js_class",
                )
            )
            # Scan class body for methods
            k = i + 1
            while k < min(n, end_1 - 1):
                m_meth = _JS_METHOD_RE.match(lines[k])
                if m_meth:
                    m_name = m_meth.group(1)
                    if m_name not in _JS_CONTROL_KEYWORDS:
                        m_end = min(end_1, _find_brace_end(lines, k, max_scan=400))
                        out.append(
                            SpectroscopicEntry(
                                symbol=f"{qual_cls}.{m_name}",
                                start_line=line_offset + k + 1,
                                end_line=line_offset + m_end,
                                sig_line=line_offset + k + 1,
                                signature=lines[k].strip().rstrip("{ ").strip()[:180],
                                kind="js_fn",
                            )
                        )
                        k = max(k + 1, m_end)
                        continue
                k += 1
            i = max(i + 1, end_1)
            continue

        m_fn = _JS_FUNC_RE.match(ln) or _JS_ARROW_OR_FN_VAR_RE.match(ln) or _JS_OBJ_ASSIGN_FN_RE.match(ln)
        if m_fn:
            fn_name = m_fn.group(1)
            if fn_name not in _JS_CONTROL_KEYWORDS:
                end_1 = _find_brace_end(lines, i, max_scan=600)
                qual_fn = f"{prefix}{fn_name}"
                out.append(
                    SpectroscopicEntry(
                        symbol=qual_fn,
                        start_line=line_offset + i + 1,
                        end_line=line_offset + end_1,
                        sig_line=line_offset + i + 1,
                        signature=stripped.rstrip("{ ").strip()[:180],
                        kind="js_fn",
                    )
                )
                i = max(i + 1, end_1)
                continue

        i += 1
    return out


def js_spectroscopic_symbols(text: str) -> list[SymbolNode] | None:
    lines = text.splitlines()
    entries = js_entries(lines)
    if not entries:
        return None
    top_nodes: list[SymbolNode] = []
    by_class: dict[str, SymbolNode] = {}
    for e in entries:
        if e.kind == "js_class":
            node = SymbolNode(e.symbol, e.start_line, e.end_line, e.sig_line, [], container=True)
            by_class[e.symbol] = node
            top_nodes.append(node)
        elif "." in e.symbol:
            cls_name, _, meth_name = e.symbol.rpartition(".")
            parent = by_class.get(cls_name)
            if parent is not None:
                parent.members.append(SymbolNode(meth_name, e.start_line, e.end_line, e.sig_line))
            else:
                top_nodes.append(SymbolNode(e.symbol, e.start_line, e.end_line, e.sig_line))
        else:
            top_nodes.append(SymbolNode(e.symbol, e.start_line, e.end_line, e.sig_line))
    return top_nodes


# ------------------------------------------------------------------------------------------
# 3. HTML Self-Contained Dashboard Spectroscopy
# ------------------------------------------------------------------------------------------
def html_entries(lines: Sequence[str]) -> list[SpectroscopicEntry]:
    """Extract `<style>`, DOM `#id` landmarks, and embedded `<script>` JS functions from an HTML file."""
    entries: list[SpectroscopicEntry] = []
    n = len(lines)
    i = 0
    script_idx = 0
    style_idx = 0
    while i < n:
        ln = lines[i]
        if _HTML_STYLE_OPEN_RE.search(ln):
            style_idx += 1
            start_1 = i + 1
            end_1 = start_1
            while end_1 <= n and not _HTML_STYLE_CLOSE_RE.search(lines[end_1 - 1]):
                end_1 += 1
            end_1 = min(n, end_1)
            entries.append(
                SpectroscopicEntry(
                    symbol=f"style#{style_idx}",
                    start_line=start_1,
                    end_line=end_1,
                    sig_line=start_1,
                    signature=f"<style> block #{style_idx} ({end_1 - start_1 + 1} lines)",
                    kind="html_style",
                )
            )
            i = end_1
            continue

        if _HTML_SCRIPT_OPEN_RE.search(ln):
            script_idx += 1
            start_1 = i + 1
            end_1 = start_1
            while end_1 <= n and not _HTML_SCRIPT_CLOSE_RE.search(lines[end_1 - 1]):
                end_1 += 1
            end_1 = min(n, end_1)
            inner_lines = lines[start_1: max(start_1, end_1 - 1)]
            js_in_script = js_entries(inner_lines, line_offset=start_1, prefix="script.")
            if js_in_script:
                entries.extend(js_in_script)
            else:
                entries.append(
                    SpectroscopicEntry(
                        symbol=f"script#{script_idx}",
                        start_line=start_1,
                        end_line=end_1,
                        sig_line=start_1,
                        signature=f"<script> block #{script_idx} ({end_1 - start_1 + 1} lines)",
                        kind="html_script",
                    )
                )
            i = end_1
            continue

        m_dom = _HTML_ID_LANDMARK_RE.search(ln)
        if m_dom:
            tag = m_dom.group(1).lower()
            dom_id = m_dom.group(2).strip()
            if dom_id and (tag != "div" or any(k in dom_id.lower() for k in ("tab", "panel", "view", "modal", "page", "section", "card", "chart", "table", "app", "root", "main", "header", "filter", "drawer"))):
                entries.append(
                    SpectroscopicEntry(
                        symbol=f"dom#{dom_id}",
                        start_line=i + 1,
                        end_line=min(n, i + 45),
                        sig_line=i + 1,
                        signature=ln.strip()[:180],
                        kind="html_dom",
                    )
                )
        i += 1
    return entries


def html_spectroscopic_symbols(text: str) -> list[SymbolNode] | None:
    lines = text.splitlines()
    entries = html_entries(lines)
    if not entries:
        return None
    entries.sort(key=lambda e: (e.start_line, e.end_line))
    # Clamp any `html_dom` landmark's end_line so it never overlaps a subsequent landmark or `<script>`/`<style>`
    clamped: list[SymbolNode] = []
    for idx, e in enumerate(entries):
        end_line = e.end_line
        if e.kind == "html_dom" and idx + 1 < len(entries):
            end_line = min(end_line, max(e.start_line, entries[idx + 1].start_line - 1))
        clamped.append(
            SymbolNode(
                name=e.symbol,
                start=e.start_line,
                end=end_line,
                sig_line=e.sig_line,
                members=[],
                container=False,
            )
        )
    return clamped


# ------------------------------------------------------------------------------------------
# Unified Spectroscopic API (Chunking + On-the-Fly Outline Resolution + File TOC)
# ------------------------------------------------------------------------------------------
def extract_spectroscopic_entries(rel_path: str, text: str, *, start_line: int = 1) -> list[SpectroscopicEntry]:
    suffix = PurePosixPath(rel_path).suffix.lower()
    lines = text.splitlines()
    if suffix in _MD_EXTENSIONS:
        raw = markdown_entries(lines)
    elif suffix in _HTML_EXTENSIONS:
        raw = html_entries(lines)
    elif suffix in _JS_TS_EXTENSIONS:
        raw = js_entries(lines)
    else:
        return []
    if start_line == 1:
        return raw
    off = start_line - 1
    return [
        SpectroscopicEntry(
            symbol=e.symbol,
            start_line=e.start_line + off,
            end_line=e.end_line + off,
            sig_line=e.sig_line + off,
            signature=e.signature,
            kind=e.kind,
        )
        for e in raw
    ]


def extract_spectroscopic_symbols(rel_path: str, text: str) -> tuple[list[SymbolNode] | None, tuple[str, ...]]:
    """Return `(SymbolNode list, comment_prefixes)` for `.md`, `.html`, `.js`/`.ts` when tree-sitter is absent."""
    suffix = PurePosixPath(rel_path).suffix.lower()
    if suffix in _MD_EXTENSIONS:
        return markdown_symbols(text), ("<!--",)
    if suffix in _HTML_EXTENSIONS:
        return html_spectroscopic_symbols(text), ("<!--", "//")
    if suffix in _JS_TS_EXTENSIONS:
        return js_spectroscopic_symbols(text), ("//", "/*", "*", "*/")
    return None, ()


def spectroscopic_outlines_for_span(rel_path: str | None, text: str, start_line: int) -> list[Outline]:
    """Extract structural `Outline` items from a chunk of `.md`, `.html`, or `.js`/`.ts` even if the chunk was
    stored in an older index as `<block:hash>`."""
    if not rel_path or not is_spectroscopic_path(rel_path):
        return []
    entries = extract_spectroscopic_entries(rel_path, text, start_line=start_line)
    if not entries:
        return []
    out: list[Outline] = []
    for e in entries[:6]:
        sig = e.signature if e.kind == "md_section" else f"[{e.symbol}] {e.signature}"
        out.append(Outline(line=e.sig_line, signature=sig, doc=None))
    return out


def render_spectroscopic_toc(rel_path: str, text: str, *, max_entries: int = 80) -> str:
    """Render a compact structural Table of Contents (`path:toc`) for any `.md`, `.html`, `.js`, or `.py` file."""
    from .chunking import chunk_python

    lines = text.splitlines()
    total = len(lines)
    suffix = PurePosixPath(rel_path).suffix.lower()
    rows: list[str] = []

    if suffix in (".py", ".pyi"):
        chunks = chunk_python(text) or []
        for c in chunks[:max_entries]:
            if c.symbol.startswith(("<header", "<gap")):
                continue
            first_ln = c.text.splitlines()[0].strip() if c.text else c.symbol
            rows.append(f"  L{c.start_line:>4}-L{c.end_line:<4}  {c.symbol:<34}  {first_ln[:90]}")
    else:
        entries = extract_spectroscopic_entries(rel_path, text, start_line=1)
        for e in entries[:max_entries]:
            rows.append(f"  L{e.start_line:>4}-L{e.end_line:<4}  {e.symbol:<34}  {e.signature[:90]}")

    if not rows:
        return f"### {rel_path} [SPECTROSCOPIC TOC: {total} lines, no structural landmarks detected]"
    more = f"\n  ... (+more entries beyond {max_entries})" if len(rows) >= max_entries else ""
    return (
        f"### {rel_path} [SPECTROSCOPIC TOC: {len(rows)} structural sections across {total} lines]\n"
        + "\n".join(rows)
        + more
    )
