"""Signature and first docstring line of the symbols in a code span: where to look, without the code.

Used by the `locations` mode of the MCP tool: an agent that can read files itself (Claude Code, Cursor) needs
to know which file and lines to open, not the bodies. A body pasted into the conversation is paid again on
every later turn of the agent; a location list is ~10x smaller.

Language-agnostic heuristics on the chunk text (no reparsing): the header starts at the line that defines the
symbol's short name and ends at the first `:` (Python) or `{` / `=>` / `;` (brace languages) with brackets
balanced, at most 12 lines. The doc is the first line of a docstring right after the header, else the comment
block right above the definition.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

_KEYWORDS = r"(?:async\s+def|def|class|func|fn|function|interface|struct|enum|trait|type|impl|record|object|module|macro_rules!)"
_MAX_SIGNATURE = 200
_MAX_DOC = 160
_MAX_HEADER_LINES = 12  # long parameter lists, one per line; the signature is clipped to _MAX_SIGNATURE anyway
_DOC_QUOTES = ('"""', "'''", 'r"""', "r'''", 'b"""', 'u"""')
_COMMENT = re.compile(r"^\s*(?:///?|#(?!\[)|/\*\*?|\*/|\*|--)\s?")


@dataclass(frozen=True)
class Outline:
    line: int  # 1-based line of the definition in the file
    signature: str
    doc: str | None = None


def short_name(symbol: str) -> str | None:
    """`Class.method~1a2b` -> `method`; non-symbol chunks (`<header>`, `<gap:...>`) -> None."""
    if not symbol or symbol.startswith("<"):
        return None
    return symbol.split("~", 1)[0].rsplit(".", 1)[-1] or None


def outline(text: str, start_line: int, symbols: Sequence[str]) -> list[Outline]:
    """One entry per symbol whose definition is found in `text` (in symbol order), else the first code line."""
    lines = text.splitlines()
    out: list[Outline] = []
    seen: set[int] = set()
    for sym in symbols:
        name = short_name(sym)
        i = _find_definition(lines, name) if name else None
        if i is None or i in seen:
            continue
        seen.add(i)
        signature, end = _header(lines, i)
        out.append(Outline(start_line + i, signature, _doc_after(lines, end) or _doc_before(lines, i)))
    if not out:
        i = next((k for k, ln in enumerate(lines) if ln.strip() and not ln.lstrip().startswith("@") and not _COMMENT.match(ln)), None)
        if i is not None:
            out.append(Outline(start_line + i, _clip(lines[i].strip(), _MAX_SIGNATURE)))
    return out


def _find_definition(lines: Sequence[str], name: str) -> int | None:
    n = re.escape(name)
    patterns = (
        re.compile(rf"\b{_KEYWORDS}\s+\*?{n}\b"),                              # def f / class C / fn f / func F
        re.compile(rf"\bfunc\s*\([^)]*\)\s*{n}\b"),                           # Go method: func (r *T) Name(
        re.compile(rf"^\s*(?:export\s+)?(?:default\s+)?(?:const|let|var)\s+{n}\b"),  # const f = (...) =>
        re.compile(rf"^\s*(?:(?:public|private|protected|static|async|readonly|override|get|set)\s+)*{n}\s*[(<=:]"),  # methods, fields
        re.compile(rf"\b{n}\s*\("),                                           # C/Java: int name(
    )
    for pat in patterns:
        for i, line in enumerate(lines):
            if pat.search(line) and not _COMMENT.match(line):
                return i
    return None


def _header(lines: Sequence[str], i: int) -> tuple[str, int]:
    """The definition line plus its continuation lines (open brackets, trailing comma), cut before the body."""
    python = lines[i].lstrip().startswith(("def ", "async def ", "class "))
    parts: list[str] = []
    depth = 0
    end = i
    for k in range(i, min(len(lines), i + _MAX_HEADER_LINES)):
        line = re.sub(r"\s+#.*$", "", lines[k]) if python else lines[k]
        end = k
        brace = None if python else _brace_cut(line, depth)
        if brace is not None:
            parts.append(line[:brace].strip())
            break
        parts.append(line.strip())
        depth += sum(line.count(c) for c in "([") - sum(line.count(c) for c in ")]")
        tail = line.rstrip()
        if depth > 0 or tail.endswith((",", "(", "[", "->", "\\")):
            continue
        nxt = next((ln.strip() for ln in lines[k + 1:k + 2]), "")
        if not python and nxt.startswith("{"):
            continue  # brace on the next line
        break
    signature = re.sub(r"\s+", " ", " ".join(p for p in parts if p)).replace("( ", "(").replace(" )", ")")
    return _clip(signature.rstrip("{ ").strip(), _MAX_SIGNATURE), end


def _brace_cut(line: str, depth: int) -> int | None:
    """Index of the `{` that opens the body (outside brackets), if any."""
    d = depth
    for idx, ch in enumerate(line):
        if ch in "([":
            d += 1
        elif ch in ")]":
            d -= 1
        elif ch == "{" and d <= 0:
            return idx
    return None


def _doc_after(lines: Sequence[str], end: int) -> str | None:
    k = end + 1
    while k < len(lines) and not lines[k].strip():
        k += 1
    if k >= len(lines):
        return None
    s = lines[k].strip()
    quote = next((q for q in _DOC_QUOTES if s.startswith(q)), None)
    if quote is None:
        return None
    closing = quote[-3:]
    first = s[len(quote):]
    if closing in first:  # one-line docstring
        return _clip(first.split(closing, 1)[0].strip(), _MAX_DOC) or None
    paragraph: list[str] = [first.strip()] if first.strip() else []
    for line in lines[k + 1:k + 8]:  # first paragraph: up to a blank line or the closing quotes
        text = line.strip()
        done = closing in text
        text = text.split(closing, 1)[0].strip()
        if not text and paragraph:
            break
        if text:
            paragraph.append(text)
        if done or len(" ".join(paragraph)) > _MAX_DOC:
            break
    return _clip(" ".join(paragraph), _MAX_DOC) or None


def _doc_before(lines: Sequence[str], i: int) -> str | None:
    k = i - 1
    while k >= 0 and (lines[k].lstrip().startswith(("@", "#[")) or not lines[k].strip()):  # decorators, attributes
        k -= 1
    block: list[str] = []
    while k >= 0 and _COMMENT.match(lines[k]):
        text = _COMMENT.sub("", lines[k], count=1).strip().rstrip("*/").strip()
        if text:
            block.append(text)
        k -= 1
    return _clip(block[-1], _MAX_DOC) if block else None  # top line of the block: usually the summary


def _clip(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"
