"""Semantic chunking with line-shift-invariant identities.

A chunk's identity is its *symbol path*, never its line numbers:

    <header>                        leading imports/constants of the file
    Class                           class header (signature, docstring, fields before the first member)
    Class.method                    member; nested containers recurse (Outer.Inner.method)
    impl Display for P.fmt          Rust impl members, Go methods as Type.Method, C++ ns.Class.method
    name~1a2b3c                     later duplicates (overloads, property setters): signature hash
    Class.<gap:9f8e7d6c>            non-symbol code between members, anchored to its first line's content
    fn#<blk:0a1b2c3d>               continuation blocks of an oversized symbol, content-anchored

Spans never include surrounding blank lines and absorb directly-preceding comments/decorators,
so inserting/removing lines, blank lines or whole symbols elsewhere leaves every other chunk's
(symbol, content_hash) unchanged; only its line span moves. Reordering symbols is also free.

Backends: tree-sitter (Python, JS/TS/TSX, Go, Rust, Java, C, C++) -> stdlib `ast` (Python) ->
content-anchored blocks (any text).
"""
from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from typing import Literal

from .syntax import LanguageSpec, SymbolNode, available_languages, language_for_path, python_ast_symbols, treesitter_symbols

CHUNKER_VERSION = "2"
Backend = Literal["auto", "treesitter", "ast", "blocks"]


@dataclass(frozen=True, slots=True)
class CodeChunk:
    symbol: str
    start_line: int  # 1-based inclusive
    end_line: int  # inclusive
    text: str
    content_hash: str


def content_hash(text: str) -> str:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()


def _short_hash(text: str, n: int = 8) -> str:
    return hashlib.blake2b(" ".join(text.split()).encode("utf-8"), digest_size=8).hexdigest()[:n]


def slice_lines(lines: list[str], start: int, end: int) -> str:
    return "\n".join(lines[start - 1 : end])


def embedding_text(rel_path: str, chunk: CodeChunk) -> str:
    """Exact text the encoder sees for a chunk (ingest and calibration must agree)."""
    return f"{rel_path}\n{chunk.symbol}\n{chunk.text}"


def chunker_fingerprint() -> str:
    return f"v{CHUNKER_VERSION}:{','.join(available_languages())}"


# ------------------------------------------------------------------------------------------
# primitives
# ------------------------------------------------------------------------------------------
def _blank(line: str) -> bool:
    return not line.strip()


def _has_content(lines: list[str], start: int, end: int) -> bool:
    return any(ch.isalnum() for ln in lines[start - 1 : end] for ch in ln)


def _trim(lines: list[str], start: int, end: int) -> tuple[int, int]:
    while start <= end and _blank(lines[start - 1]):
        start += 1
    while end >= start and _blank(lines[end - 1]):
        end -= 1
    return start, end


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip())


def _is_leading(line: str, prefixes: tuple[str, ...]) -> bool:
    s = line.strip()
    return bool(s) and not s.startswith("#!") and s.startswith(prefixes)


def _signature(lines: list[str], sig_line: int) -> str:
    buf: list[str] = []
    for ln in lines[sig_line - 1 : sig_line + 4]:
        buf.append(ln.strip())
        if "{" in ln or ln.rstrip().endswith((":", ";")):
            break
    return " ".join(" ".join(buf).split())


def anchored_blocks(lines: list[str], start: int, end: int, *, min_lines: int, max_lines: int) -> list[tuple[int, int]]:
    """Content-defined split of [start, end]: cut before a line at the region's base indentation that
    follows a blank line, once the block has >= min_lines; forced cut at max_lines. Boundaries depend
    only on local content, so an insertion re-chunks only its own block (barring forced cuts)."""
    start, end = _trim(lines, start, end)
    if start > end:
        return []
    base = min((_indent(lines[i - 1]) for i in range(start, end + 1) if not _blank(lines[i - 1])), default=0)
    blocks: list[tuple[int, int]] = []
    cur = start
    for i in range(start + 1, end + 1):
        size = i - cur
        natural = (size >= min_lines and _blank(lines[i - 2]) and not _blank(lines[i - 1]) and _indent(lines[i - 1]) <= base)
        if natural or size >= max_lines:
            s, e = _trim(lines, cur, i - 1)
            if s <= e:
                blocks.append((s, e))
            cur = i
    s, e = _trim(lines, cur, end)
    if s <= e:
        blocks.append((s, e))
    return blocks


def _first_content_line(lines: list[str], start: int, end: int) -> str:
    return next((lines[i - 1] for i in range(start, end + 1) if not _blank(lines[i - 1])), "")


class _Namer:
    """Per-scope unique names: first occurrence keeps the plain name, later duplicates get suffixes."""

    def __init__(self) -> None:
        self._used: set[str] = set()

    def take(self, name: str, disambiguator: str) -> str:
        if name not in self._used:
            self._used.add(name)
            return name
        candidate = f"{name}~{_short_hash(disambiguator, 6)}"
        k = 2
        while candidate in self._used:
            candidate = f"{name}~{_short_hash(disambiguator, 6)}@{k}"
            k += 1
        self._used.add(candidate)
        return candidate


# ------------------------------------------------------------------------------------------
# assembly (backend-independent)
# ------------------------------------------------------------------------------------------
@dataclass
class _Ctx:
    lines: list[str]
    prefixes: tuple[str, ...]
    max_lines: int
    min_lines: int
    out: list[tuple[str, int, int]]


def _place(ctx: _Ctx, syms: list[SymbolNode], lo: int, hi: int) -> list[tuple[SymbolNode, int, int]]:
    placed: list[tuple[SymbolNode, int, int]] = []
    prev = lo - 1
    for s in sorted(syms, key=lambda s: (s.start, s.end)):
        start, end = max(s.start, prev + 1), min(s.end, hi)
        if start > end:
            continue  # shares lines with the previous symbol: content stays there
        while start - 1 > prev and _is_leading(ctx.lines[start - 2], ctx.prefixes):
            start -= 1
        placed.append((s, start, end))
        prev = end
    return placed


def _emit_span(ctx: _Ctx, qual: str, start: int, end: int, namer: _Namer | None = None) -> None:
    if end - start + 1 <= ctx.max_lines:
        ctx.out.append((qual, start, end))
        return
    blocks = anchored_blocks(ctx.lines, start, end, min_lines=ctx.min_lines, max_lines=ctx.max_lines)
    local = namer or _Namer()
    for i, (s, e) in enumerate(blocks):
        if i == 0:
            ctx.out.append((qual, s, e))
        else:
            anchor = _short_hash(_first_content_line(ctx.lines, s, e))
            ctx.out.append((local.take(f"{qual}#<blk:{anchor}>", f"{s}"), s, e))


def _emit_scope(ctx: _Ctx, placed: list[tuple[SymbolNode, int, int]], prefix: str, gap_lo: int, hi: int, *, top: bool) -> None:
    namer = _Namer()
    for s, start, end in placed:
        qual = prefix + namer.take(s.name, _signature(ctx.lines, s.sig_line))
        members = _place(ctx, s.members, s.sig_line + 1, end) if s.container else []
        if members and members[0][1] > start:
            _emit_span(ctx, qual, *_trim(ctx.lines, start, members[0][1] - 1))
            _emit_scope(ctx, members, qual + ".", members[0][1], end, top=False)
        else:
            _emit_span(ctx, qual, start, end)

    gap_namer = _Namer()
    cursor = gap_lo
    bounds = [(st, en) for _, st, en in placed if en >= gap_lo] + [(hi + 1, hi + 1)]
    for st, en in bounds:
        gs, ge = _trim(ctx.lines, cursor, min(st - 1, hi))
        if gs <= ge and _has_content(ctx.lines, gs, ge):
            if top and cursor == gap_lo and (not placed or ge < placed[0][1]):
                _emit_span(ctx, "<header>", gs, ge, gap_namer)
            else:
                for bs, be in anchored_blocks(ctx.lines, gs, ge, min_lines=ctx.min_lines, max_lines=ctx.max_lines):
                    if _has_content(ctx.lines, bs, be):
                        anchor = _short_hash(_first_content_line(ctx.lines, bs, be))
                        ctx.out.append((gap_namer.take(f"{prefix}<gap:{anchor}>", f"{bs}"), bs, be))
        cursor = max(cursor, en + 1)


def assemble_chunks(lines: list[str], symbols: list[SymbolNode], prefixes: tuple[str, ...], *, max_lines: int, min_lines: int) -> list[CodeChunk]:
    ctx = _Ctx(lines, prefixes, max_lines, min_lines, [])
    _emit_scope(ctx, _place(ctx, symbols, 1, len(lines)), "", 1, len(lines), top=True)
    ctx.out.sort(key=lambda t: (t[1], t[2], t[0]))
    chunks: list[CodeChunk] = []
    for sym, s, e in ctx.out:
        text = slice_lines(lines, s, e)
        chunks.append(CodeChunk(sym, s, e, text, content_hash(text)))
    return chunks


def chunk_blocks(text: str, *, max_lines: int, min_lines: int) -> list[CodeChunk]:
    lines = text.splitlines()
    namer = _Namer()
    chunks: list[CodeChunk] = []
    for s, e in anchored_blocks(lines, 1, len(lines), min_lines=min_lines, max_lines=max_lines):
        body = slice_lines(lines, s, e)
        sym = namer.take(f"<block:{_short_hash(_first_content_line(lines, s, e))}>", f"{s}")
        chunks.append(CodeChunk(sym, s, e, body, content_hash(body)))
    return chunks


def extract_symbols(rel_path: str, text: str, backend: Backend = "auto") -> tuple[list[SymbolNode] | None, LanguageSpec | None]:
    spec = language_for_path(rel_path)
    if spec is None or backend == "blocks":
        return None, spec
    if backend in ("auto", "treesitter"):
        syms = treesitter_symbols(spec, text)
        if syms is not None or backend == "treesitter":
            return syms, spec
    if spec.name == "python" and backend in ("auto", "ast"):
        return python_ast_symbols(text), spec
    return None, spec


def chunk_source(rel_path: str, text: str, *, max_lines: int = 120, min_lines: int = 12, backend: Backend = "auto") -> list[CodeChunk]:
    syms, spec = extract_symbols(rel_path, text, backend)
    if not syms and backend == "auto":
        from .spectroscopy import extract_spectroscopic_symbols

        spec_syms, prefixes = extract_spectroscopic_symbols(rel_path, text)
        if spec_syms:
            return assemble_chunks(text.splitlines(), spec_syms, prefixes, max_lines=max_lines, min_lines=min_lines)
    if syms is None or spec is None:
        return chunk_blocks(text, max_lines=max_lines, min_lines=min_lines)
    return assemble_chunks(text.splitlines(), syms, spec.comment_prefixes, max_lines=max_lines, min_lines=min_lines)


def chunk_python(text: str, *, max_lines: int = 120, min_lines: int = 12) -> list[CodeChunk] | None:
    """Python chunks via the best available backend; None on unparsable source."""
    syms, spec = extract_symbols("x.py", text)
    if syms is None or spec is None:
        return None
    return assemble_chunks(text.splitlines(), syms, spec.comment_prefixes, max_lines=max_lines, min_lines=min_lines)
