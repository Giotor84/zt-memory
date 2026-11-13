"""Source file discovery shared by the manager, the watcher and the calibration tools.

Inside a git work tree the file list is `git ls-files --cached --others --exclude-standard`
(exact .gitignore semantics, untracked-but-not-ignored files included); elsewhere a pruned walk.
Hidden path components and configured excluded dirs are always skipped (this also hides `.zt/`).
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path, PurePosixPath
from typing import Callable, Iterable

from .config import MemoryConfig


def is_indexable(rel: str, config: MemoryConfig) -> bool:
    p = PurePosixPath(rel)
    if any(part.startswith(".") or part in config.excluded_dirs for part in p.parts[:-1]):
        return False
    if config.exclude_globs and is_excluded(rel, config):
        return False
    return p.suffix.lower() in config.code_extensions or p.name in config.code_filenames


def exclude_patterns(config: MemoryConfig) -> list[str]:
    return [g.strip() for g in config.exclude_globs.split(",") if g.strip()]


def is_excluded(rel: str, config: MemoryConfig) -> bool:
    """PurePosixPath.match semantics plus prefix directory globbing (`dir/*` or `dir/**`), case-insensitive."""
    norm = rel.replace("\\", "/").lower().lstrip("/")
    path = PurePosixPath(norm)
    for raw in exclude_patterns(config):
        g = raw.replace("\\", "/").lower().lstrip("/")
        if path.match(g):
            return True
        if g.endswith("/**") and (norm == g[:-3] or norm.startswith(g[:-2])):
            return True
        if g.endswith("/*") and norm.startswith(g[:-1]):
            return True
    return False


def git_files(root: Path, timeout_s: float = 30.0) -> list[str] | None:
    """Repo-relative paths under `root` from git, or None when git/work tree is unavailable."""
    if shutil.which("git") is None:
        return None
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
            capture_output=True, timeout=timeout_s, check=True,
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return None
    paths = [p for p in out.decode("utf-8", "surrogateescape").split("\0") if p]
    return [p for p in paths if (root / p).is_file()]  # drops deleted-but-tracked entries and submodules


def walk_files(root: Path, excluded_dirs: Iterable[str]) -> list[str]:
    excluded = set(excluded_dirs)
    out: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in excluded and not d.startswith("."))
        for name in sorted(filenames):
            out.append(Path(dirpath, name).relative_to(root).as_posix())
    return out


def discover(config: MemoryConfig, *, predicate: Callable[[str], bool] | None = None) -> list[str]:
    root = config.repo_root
    if not root.is_dir():
        return []
    files = git_files(root) if config.respect_gitignore else None
    if files is None:
        files = walk_files(root, config.excluded_dirs)
    keep = predicate or (lambda rel: is_indexable(rel, config))
    return sorted(rel for rel in files if keep(rel))
