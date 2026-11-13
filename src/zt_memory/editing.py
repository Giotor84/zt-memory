"""Edit protocol between the compute engine and the repository: SEARCH/REPLACE blocks.

Why not unified diffs: local models routinely get hunk headers and line numbers wrong. A SEARCH block
anchors on exact text the model has just seen in the routed context, which is verifiable before writing.

    path/to/file.py
    <<<<<<< SEARCH
    exact current lines
    =======
    replacement lines
    >>>>>>> REPLACE

Application is all-or-nothing: every block is resolved in memory first (exact match, then trailing-
whitespace-insensitive, then indentation-insensitive with re-indentation, always requiring a unique match;
a section wrapped in its own ``` fence, a habit of chat models, is retried without it);
only if all succeed are files written, after a backup that `undo_last` can restore.
"""
from __future__ import annotations

import difflib
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Sequence

SEARCH_REPLACE_CONTRACT = (
    "Reply with edit blocks only, in exactly this format (one block per change):\n\n"
    "path/relative/to/repo.ext\n"
    "<<<<<<< SEARCH\n"
    "exact lines copied from the code context, with their indentation\n"
    "=======\n"
    "the replacement lines\n"
    ">>>>>>> REPLACE\n\n"
    "Rules: the SEARCH text must match the current file exactly and occur once in it - copy it from the context "
    "above and include a few unchanged surrounding lines to make it unique. Keep blocks small. To create a new "
    "file use an empty SEARCH section. Blocks for the same file are applied in order. Do not use line numbers "
    "and do not put ``` fences inside the SEARCH or REPLACE sections. "
    "At most one short sentence before the first block; nothing after the last block."
)

_SEARCH = re.compile(r"^\s*<{5,9} ?SEARCH\s*$")
_DIVIDER = re.compile(r"^\s*={5,9}\s*$")
_REPLACE = re.compile(r"^\s*>{5,9} ?REPLACE\s*$")
_FENCE = re.compile(r"^\s*```")
_FENCE_OPEN = re.compile(r"^\s*```[\w+#.-]*\s*$")
_FENCE_CLOSE = re.compile(r"^\s*```\s*$")
_LINE_NO = re.compile(r"^\s*\d+ \| ?")
_FORBIDDEN_DIRS = frozenset({".git", ".zt"})
_PROSE_SUFFIXES = frozenset({".md", ".markdown", ".rst", ".txt", ".mdx"})  # fences may be real content there


class EditError(ValueError):
    pass


@dataclass(frozen=True)
class EditBlock:
    path: str
    search: str
    replace: str
    index: int  # position in the model's answer (for feedback)


@dataclass
class BlockFailure:
    block: EditBlock
    reason: str


@dataclass
class EditPlan:
    """Resolved edits, not yet written."""

    new_contents: dict[str, str] = field(default_factory=dict)
    original: dict[str, str | None] = field(default_factory=dict)  # None = file did not exist
    failures: list[BlockFailure] = field(default_factory=list)
    match_modes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failures and bool(self.new_contents)

    def diff(self) -> str:
        out: list[str] = []
        for path, new in self.new_contents.items():
            old = self.original.get(path) or ""
            out.extend(difflib.unified_diff(old.splitlines(keepends=True), new.splitlines(keepends=True),
                                            fromfile=f"a/{path}", tofile=f"b/{path}"))
        return "".join(out)


@dataclass(frozen=True)
class AppliedEdit:
    changed: tuple[str, ...]
    created: tuple[str, ...]
    backup_id: str


# ------------------------------------------------------------------------------------------
# parsing
# ------------------------------------------------------------------------------------------
def _clean_path(line: str) -> str:
    p = line.strip()
    for prefix in ("File:", "file:", "Path:", "path:", "Filename:", "#"):
        if p.startswith(prefix):
            p = p[len(prefix):].strip()
    return p.strip("`*:'\" ").replace("\\", "/")


def _strip_line_numbers(lines: list[str]) -> list[str]:
    """Models sometimes copy the 'NN | ' gutter of a numbered context: drop it when every line has it."""
    if lines and all(_LINE_NO.match(ln) for ln in lines if ln.strip()):
        return [_LINE_NO.sub("", ln, count=1) for ln in lines]
    return lines


def parse_edit_blocks(text: str) -> list[EditBlock]:
    lines = text.replace("\r\n", "\n").split("\n")
    blocks: list[EditBlock] = []
    i = 0
    last_path: str | None = None
    while i < len(lines):
        if not _SEARCH.match(lines[i]):
            i += 1
            continue
        path = None
        for j in range(i - 1, max(i - 4, -1), -1):  # nearest preceding non-fence, non-empty line
            cand = lines[j].strip()
            if not cand or _FENCE.match(cand):
                continue
            if not _REPLACE.match(cand):
                path = _clean_path(cand)
            break
        path = path or last_path  # consecutive blocks for one file may omit the path
        if not path:
            raise EditError(f"edit block {len(blocks) + 1}: no file path before '<<<<<<< SEARCH'")
        search: list[str] = []
        replace: list[str] = []
        k = i + 1
        while k < len(lines) and not _DIVIDER.match(lines[k]):
            search.append(lines[k])
            k += 1
        if k >= len(lines):
            raise EditError(f"edit block {len(blocks) + 1} ({path}): missing '======='")
        k += 1
        while k < len(lines) and not _REPLACE.match(lines[k]):
            replace.append(lines[k])
            k += 1
        if k >= len(lines):
            raise EditError(f"edit block {len(blocks) + 1} ({path}): missing '>>>>>>> REPLACE'")
        blocks.append(EditBlock(path, "\n".join(_strip_line_numbers(search)), "\n".join(_strip_line_numbers(replace)), len(blocks) + 1))
        last_path = path
        i = k + 1
    return blocks


# ------------------------------------------------------------------------------------------
# resolution
# ------------------------------------------------------------------------------------------
def _safe_rel(repo_root: Path, path: str) -> str:
    rel = PurePosixPath(path)
    if not path or rel.is_absolute() or ".." in rel.parts or re.match(r"^[A-Za-z]:", path):
        raise EditError(f"unsafe path {path!r}")
    if rel.parts and rel.parts[0] in _FORBIDDEN_DIRS:
        raise EditError(f"refusing to edit inside {rel.parts[0]}/")
    target = (repo_root / rel).resolve()
    if repo_root.resolve() not in target.parents:
        raise EditError(f"path escapes the repository: {path!r}")
    return str(rel)


def _indent(s: str) -> str:
    return s[: len(s) - len(s.lstrip())]


def _find(lines: list[str], search: list[str], key) -> list[int]:  # noqa: ANN001
    n = len(search)
    want = [key(s) for s in search]
    return [i for i in range(len(lines) - n + 1) if [key(x) for x in lines[i : i + n]] == want]


def _unfence(section: list[str]) -> list[str] | None:
    """The section without its own outer ```lang ... ``` pair, or None when it is not wrapped in one."""
    body = list(section)
    while body and not body[0].strip():
        body.pop(0)
    while body and not body[-1].strip():
        body.pop()
    if len(body) >= 2 and _FENCE_OPEN.match(body[0]) and _FENCE_CLOSE.match(body[-1]):
        return body[1:-1]
    return None


def _apply_block(text: str, block: EditBlock) -> tuple[str, str]:
    """Return (new_text, match_mode). Works on lines; preserves the file's newline style and final newline."""
    search = block.search.split("\n")
    replace = block.replace.split("\n") if block.replace else []
    try:
        return _apply_lines(text, search, replace)
    except EditError as exc:
        unfenced = _unfence(search)
        if unfenced is None or "not found" not in str(exc):
            raise
        unfenced_replace = _unfence(replace)
        new, mode = _apply_lines(text, unfenced, replace if unfenced_replace is None else unfenced_replace)
        return new, "unfenced+" + mode


def _apply_lines(text: str, search: list[str], replace: list[str]) -> tuple[str, str]:
    newline = "\r\n" if "\r\n" in text else "\n"
    trailing = text.endswith(("\n", "\r\n"))
    lines = text.splitlines()
    search = list(search)
    while search and not search[-1].strip():
        search.pop()  # tolerate a trailing blank line in SEARCH
    while search and not search[0].strip():
        search.pop(0)
    if not search:
        raise EditError("empty SEARCH on an existing file (only allowed to create files)")
    for mode, key in (("exact", lambda s: s), ("rstrip", lambda s: s.rstrip()), ("indent", lambda s: s.strip())):
        hits = _find(lines, search, key)
        if len(hits) > 1:
            raise EditError(f"SEARCH matches {len(hits)} places ({mode}): add surrounding lines to make it unique")
        if len(hits) == 1:
            i = hits[0]
            if mode == "indent":  # re-indent the replacement by the observed offset
                have, wrote = _indent(lines[i]), _indent(search[0])
                fixed = []
                for r in replace:
                    if r.startswith(wrote):
                        fixed.append(have + r[len(wrote):])
                    else:
                        fixed.append(have + r.lstrip() if r.strip() else r)
                replace = fixed
            out = lines[:i] + replace + lines[i + len(search):]
            return newline.join(out) + (newline if trailing or not out else ""), mode
    raise EditError("SEARCH text not found in the file (it must be copied exactly from the current content)")


def plan_edits(repo_root: Path, blocks: Sequence[EditBlock], *, encoding: str = "utf-8") -> EditPlan:
    plan = EditPlan()
    root = Path(repo_root)
    for block in blocks:
        try:
            rel = _safe_rel(root, block.path)
            if rel not in plan.new_contents:
                p = root / rel
                if p.exists():
                    if not p.is_file():
                        raise EditError(f"{rel} is not a regular file")
                    try:
                        current = p.read_bytes().decode(encoding)
                    except UnicodeDecodeError as exc:
                        raise EditError(f"{rel} is not {encoding} text") from exc
                    plan.original[rel] = current
                else:
                    plan.original[rel] = None
                    current = None
            else:
                current = plan.new_contents[rel]
            if current is None:
                if block.search.strip():
                    raise EditError(f"{rel} does not exist (use an empty SEARCH to create it)")
                body = block.replace
                unfenced = _unfence(body.split("\n"))
                if unfenced is not None and PurePosixPath(rel).suffix.lower() not in _PROSE_SUFFIXES:
                    body = "\n".join(unfenced)  # a new source file wrapped in a chat fence
                plan.new_contents[rel] = body + ("\n" if body and not body.endswith("\n") else "")
                plan.match_modes.append("create")
            elif not block.search.strip() and not current.strip():
                plan.new_contents[rel] = block.replace + "\n"
                plan.match_modes.append("fill-empty")
            else:
                new, mode = _apply_block(current, block)
                plan.new_contents[rel] = new
                plan.match_modes.append(mode)
        except EditError as exc:
            plan.failures.append(BlockFailure(block, str(exc)))
    return plan


# ------------------------------------------------------------------------------------------
# writing, backup, undo
# ------------------------------------------------------------------------------------------
def _backups_dir(repo_root: Path) -> Path:
    return Path(repo_root) / ".zt" / "backups"


def write_plan(repo_root: Path, plan: EditPlan, *, label: str = "") -> AppliedEdit:
    if not plan.ok:
        raise EditError("refusing to write a plan with failures or no changes")
    root = Path(repo_root)
    backup_id = time.strftime("%Y%m%d-%H%M%S") + f"-{os.getpid()}"
    bdir = _backups_dir(root) / backup_id
    (bdir / "files").mkdir(parents=True, exist_ok=True)
    manifest = {"label": label, "files": []}
    for rel, old in plan.original.items():
        if rel not in plan.new_contents:
            continue
        if old is not None:
            dst = bdir / "files" / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes((root / rel).read_bytes())
        manifest["files"].append({"path": rel, "existed": old is not None})
    (bdir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    changed, created = [], []
    for rel, new in plan.new_contents.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".zt-tmp")
        tmp.write_bytes(new.encode("utf-8"))  # bytes: newline style already resolved per file
        os.replace(tmp, target)
        (created if plan.original.get(rel) is None else changed).append(rel)
    return AppliedEdit(tuple(changed), tuple(created), backup_id)


def undo_last(repo_root: Path) -> tuple[str, list[str]] | None:
    """Restore the newest not-yet-undone backup. Returns (backup_id, restored paths) or None."""
    root = Path(repo_root)
    bdir = _backups_dir(root)
    if not bdir.is_dir():
        return None
    candidates = sorted(d for d in bdir.iterdir() if d.is_dir() and not d.name.endswith(".undone") and (d / "manifest.json").exists())
    if not candidates:
        return None
    last = candidates[-1]
    manifest = json.loads((last / "manifest.json").read_text(encoding="utf-8"))
    restored: list[str] = []
    for entry in manifest["files"]:
        rel = entry["path"]
        target = root / rel
        if entry["existed"]:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((last / "files" / rel).read_bytes())
        elif target.exists():
            target.unlink()
        restored.append(rel)
    last.rename(last.with_name(last.name + ".undone"))
    return last.name, restored


def failure_feedback(plan: EditPlan, repo_root: Path, *, max_lines: int = 60) -> str:
    """Message for the model's retry: what failed and the file's real current text around the intended spot."""
    parts = ["Some edit blocks could not be applied, so nothing was changed:"]
    for f in plan.failures:
        parts.append(f"- block {f.block.index} ({f.block.path}): {f.reason}")
        p = Path(repo_root) / f.block.path
        if p.is_file():
            lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
            first = next((s.strip() for s in f.block.search.split("\n") if s.strip() and not _FENCE.match(s)), "")
            anchor = next((i for i, ln in enumerate(lines) if first and first[:40] in ln), None)
            lo = max(0, (anchor or 0) - 10)
            excerpt = "\n".join(lines[lo : lo + max_lines])
            parts.append(f"  Current content of {f.block.path} (excerpt):\n```\n{excerpt}\n```")
    parts.append("Resend ALL edit blocks (including the ones that were fine), with SEARCH copied exactly from the current content.")
    return "\n".join(parts)
