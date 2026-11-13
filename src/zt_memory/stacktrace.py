"""Code locations named in error output: tracebacks, test failures, compiler and linter messages.

A pasted traceback says exactly where the program was when it failed. The router resolves those
locations to indexed symbols and pins them at the top of the context, ahead of any ranking.

Supported shapes (one regex each, order of appearance kept):
  Python   File "/abs/or/rel/path.py", line 412, in func
  generic  path/to/file.ext:LINE[:COL]   (pytest, mypy, ruff, tsc, eslint, go, rustc "-->", node "at f (...)")
Paths may be absolute, relative, Windows-style, or point to an installed copy of the package
(site-packages/pkg/mod.py): they are matched to indexed files by their longest common path suffix.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Iterable, Sequence

_EXT = r"(?:py|pyi|js|jsx|mjs|cjs|ts|tsx|go|rs|java|kt|kts|c|h|cc|cpp|hpp|cs|rb|php|swift|scala)"
_PY_FRAME = re.compile(r'File "(?P<path>[^"\n]+)", line (?P<line>\d+)(?:, in (?P<func>[^\s,]+))?')
_LOCATION = re.compile(
    rf"(?<![\w/\\.~@+-])(?P<path>(?:[A-Za-z]:)?[\w./\\@+~-]*?[\w@+~-]+\.{_EXT}):(?P<line>\d+)(?::\d+)?(?:: in (?P<func>[\w<>.]+))?"
)


@dataclass(frozen=True)
class Location:
    path: str  # as written in the text, separators normalized to "/"
    line: int
    func: str | None = None


def parse_locations(text: str) -> list[Location]:
    """Every file:line the text names, innermost frame first, without duplicates.

    Python prints tracebacks outermost first ("most recent call last"), so Python locations are reversed;
    other toolchains (Go, Node, Java, compilers) already print the failing location first."""
    found: list[tuple[int, Location]] = []
    for m in _PY_FRAME.finditer(text):
        found.append((m.start(), Location(_norm(m.group("path")), int(m.group("line")), m.group("func"))))
    for m in _LOCATION.finditer(text):
        found.append((m.start(), Location(_norm(m.group("path")), int(m.group("line")), m.group("func"))))
    found.sort(key=lambda t: t[0])
    py = [loc for _, loc in found if loc.path.endswith((".py", ".pyi"))]
    other = [loc for _, loc in found if not loc.path.endswith((".py", ".pyi"))]
    out: dict[tuple[str, int], Location] = {}
    for loc in [*reversed(py), *other]:
        out.setdefault((loc.path, loc.line), loc)
    return list(out.values())


def _norm(path: str) -> str:
    p = re.sub(r"/{2,}", "/", path.strip().strip("'\"").replace("\\", "/"))
    while p.startswith("./"):
        p = p[2:]
    return p


class PathResolver:
    """Maps a path as written in error output to an indexed repository path, or None.

    A match needs the whole indexed path to be a suffix of the written one (absolute paths into the repo),
    or at least two trailing components in common (installed copies: site-packages/pkg/mod.py ->
    src/pkg/mod.py), or a bare file name that is unique in the repository. Ties are refused, not guessed."""

    def __init__(self, indexed_paths: Iterable[str]) -> None:
        self._paths = set(indexed_paths)
        self._by_name: dict[str, list[tuple[str, ...]]] = defaultdict(list)
        for p in self._paths:
            parts = PurePosixPath(p).parts
            self._by_name[parts[-1].lower()].append(parts)

    def resolve(self, written: str) -> str | None:
        if written in self._paths:
            return written
        parts = tuple(x for x in written.split("/") if x not in ("", "."))
        if not parts:
            return None
        candidates = self._by_name.get(parts[-1].lower(), [])
        best, best_n, tie = None, 0, False
        for cand in candidates:
            n = _common_suffix(parts, cand)
            whole = n == len(cand)
            if not whole and n < 2 and not (len(parts) == 1 and len(candidates) == 1):
                continue
            score = n + (0.5 if whole else 0.0)
            if score > best_n:
                best, best_n, tie = "/".join(cand), score, False
            elif score == best_n:
                tie = True
        return None if tie else best


def _common_suffix(a: Sequence[str], b: Sequence[str]) -> int:
    n = 0
    while n < min(len(a), len(b)) and a[-1 - n].lower() == b[-1 - n].lower():
        n += 1
    return n
