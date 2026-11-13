"""Reading line ranges of project files on request: the NEED round of `zt code` and the `zt_read_span` MCP tool.

One rule for both: a span is readable only if its file is one zt would index. That keeps out
- paths outside the project (absolute, `..`, or a symlink that resolves elsewhere);
- files matched by `[memory] exclude_globs` (the files an agent must never read), checked on the path as written
  and on the resolved path;
- hidden files and folders (`.git`, `.zt`, `.env`), the excluded folders (`node_modules`, `.venv`, ...), files
  without a code/text extension and files over `max_file_bytes`;
- inside a git work tree with `respect_gitignore`, files git ignores.
A refusal never says more than the reason: no content, no size, not even whether an excluded file exists.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath

from .config import DEFAULT_CODE_EXTENSIONS, DEFAULT_CODE_FILENAMES, DEFAULT_EXCLUDED_DIRS, MemoryConfig
from .discovery import is_excluded

MAX_SPAN_LINES = 300
MAX_TOTAL_LINES = 600

_SPAN = re.compile(r"^\s*`?(?P<path>[^`]+?):L?(?P<start>\d+)(?:\s*-\s*L?(?P<end>\d+))?(?P<force>!)?`?\s*$")
_TOC_SPAN = re.compile(r"^\s*`?(?P<path>[^`]+?):(?P<mode>toc|outline)`?\s*$", re.IGNORECASE)


class SpanRefused(ValueError):
    """The span cannot be read; the message is safe to show to the model."""


@dataclass(frozen=True)
class SpanPolicy:
    """Which files of `root` may be read. `config` None = the defaults, with no exclude_globs."""

    root: Path
    config: MemoryConfig | None = None

    def resolve(self, raw: str) -> tuple[str, Path]:
        """(repo-relative POSIX path, absolute path) of a readable file, else SpanRefused."""
        text = raw.strip().replace("\\", "/")
        while text.startswith("./"):
            text = text[2:]
        rel = PurePosixPath(text)
        if not text or rel.is_absolute() or PureWindowsPath(raw.strip()).drive or ".." in rel.parts:
            raise SpanRefused("outside the project: give a path relative to the project root")
        root = Path(self.root).resolve()
        target = (root / rel).resolve()
        if target != root and root not in target.parents:
            raise SpanRefused("outside the project")
        real = target.relative_to(root).as_posix()
        for candidate in dict.fromkeys((rel.as_posix(), real)):
            self._check(candidate)
        if not target.is_file():
            raise SpanRefused("no such file in the project")
        if target.stat().st_size > (self.config.max_file_bytes if self.config else 1_000_000):
            raise SpanRefused("file too large to read through zt")
        if self._git_ignored(root, real):
            raise SpanRefused("ignored by git: not part of the project's code")
        return real, target

    def _check(self, rel: str) -> None:
        cfg = self.config
        p = PurePosixPath(rel)
        if cfg is not None and cfg.exclude_globs and is_excluded(rel, cfg):
            raise SpanRefused("excluded by the project configuration: zt never reads this file")
        excluded_dirs = cfg.excluded_dirs if cfg else DEFAULT_EXCLUDED_DIRS
        if any(part.startswith(".") for part in p.parts) or any(part in excluded_dirs for part in p.parts[:-1]):
            raise SpanRefused("hidden or excluded folder: not part of the indexed code")
        extensions = cfg.code_extensions if cfg else DEFAULT_CODE_EXTENSIONS
        filenames = cfg.code_filenames if cfg else DEFAULT_CODE_FILENAMES
        if p.suffix.lower() not in extensions and p.name not in filenames:
            raise SpanRefused("not a code or text file zt indexes")

    def _git_ignored(self, root: Path, rel: str) -> bool:
        if self.config is not None and not self.config.respect_gitignore:
            return False
        if shutil.which("git") is None:
            return False
        try:
            rc = subprocess.run(["git", "-C", str(root), "check-ignore", "-q", "--", rel], capture_output=True,
                                timeout=10).returncode
        except (subprocess.SubprocessError, OSError):
            return False
        return rc == 0  # 1 = not ignored, 128 = not a work tree

    def read(self, raw_path: str, start: int, end: int, *, max_lines: int = MAX_SPAN_LINES) -> tuple[str, str, int, int, int]:
        """(relative path, text, first line, last line, lines in file); at most `max_lines` lines."""
        rel, target = self.resolve(raw_path)
        try:
            lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as exc:
            raise SpanRefused(f"cannot read the file: {exc.strerror or exc}") from None
        if start > end:
            start, end = end, start
        start = max(1, start)
        if start > len(lines):
            raise SpanRefused(f"the file has only {len(lines)} lines")
        end = min(len(lines), end, start + max(1, max_lines) - 1)
        return rel, "\n".join(lines[start - 1:end]), start, end, len(lines)


def parse_span(text: str) -> tuple[str, int, int]:
    """`path:L10-40`, `path:10-40` or `path:L12` (one line) -> (path, start, end)."""
    m = _SPAN.match(text)
    if not m:
        raise SpanRefused(f"cannot read {text!r}: write spans as path:L<start>-<end>")
    start = int(m.group("start"))
    return m.group("path").strip(), start, int(m.group("end") or start)


def parse_span_ex(text: str) -> tuple[str, int, int, bool, bool]:
    """Extended span parser: returns `(path, start, end, force_full, is_toc)`.
    Supports `path:L10-40`, `path:L10-40!` (force_full), and `path:toc` / `path:outline`."""
    m_toc = _TOC_SPAN.match(text)
    if m_toc:
        return m_toc.group("path").strip(), 1, 1, False, True
    m = _SPAN.match(text)
    if not m:
        raise SpanRefused(f"cannot read {text!r}: write spans as path:L<start>-<end>, path:L<start>-<end>!, or path:toc")
    start = int(m.group("start"))
    end = int(m.group("end") or start)
    force = bool(m.group("force"))
    return m.group("path").strip(), start, end, force, False
