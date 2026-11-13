"""Calibration corpus for the SVD basis when the project has too little text (greenfield)."""
from __future__ import annotations

import random
import sysconfig
from pathlib import Path

from .chunking import chunk_python

_SKIP_PARTS = frozenset({"test", "tests", "idlelib", "site-packages", "dist-packages", "lib2to3", "turtledemo"})


def stdlib_calibration_corpus(n: int, *, max_lines: int = 80, seed: int = 0) -> list[str]:
    """Deterministic sample of Python stdlib chunks (Python-biased: prefer project text when available)."""
    if n <= 0:
        return []
    root = Path(sysconfig.get_paths()["stdlib"])
    files = sorted(p for p in root.rglob("*.py") if not (_SKIP_PARTS & set(p.relative_to(root).parts)))
    random.Random(seed).shuffle(files)
    out: list[str] = []
    for path in files:
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        chunks = chunk_python(text, max_lines=max_lines) or []
        for ch in chunks:
            if ch.text.strip():
                out.append(ch.text)
                if len(out) >= n:
                    return out
    return out
