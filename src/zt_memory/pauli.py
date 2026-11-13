"""Pauli Exclusion Principle & Session Phase Refraction for `zt-memory`.

Solves Finding A from the 152-session Forensic Token-Waste Study:
- 54.6% of all raw tool token waste (7.94M tokens, ~14.5B compounded KV-cache tokens) came from
  repeated file reads within the same coding session (e.g. 75.1% re-read churn in one private project;
  one adapter module read 182 times, the pipeline module 111 times).

Physics Formulation (Pauli Exclusion Principle):
No two identical informational fermions `(path, start_line, end_line, sha256)` may occupy the same
session context window simultaneously:
1. Stationary State (Delta H = 0, disk `sha256` unchanged since earlier turn in this session):
   - In `zt_read_span`: refracts the duplicate span to a 3-line `[PAULI_EXCLUSION: UNCHANGED]` anchor
     (~30 tokens instead of ~2,000 tokens, -98% token waste), unless forced with a trailing `!` (`path:L10-50!`).
   - In `zt_query_context(mode="focus")`: refracts already-delivered full bodies to a 1-line Pauli resonance
     anchor and transfers their freed token budget to promote the next highest-scoring unseen symbols into full focus!
2. Perturbed State (Delta H != 0, file was edited mid-session):
   - When the agent re-reads a region it previously read in the same session after editing the file,
     emits a `[PAULI_WAVEFRONT_DELTA]` containing the unified diff (`+-3` lines) against the turn `#t`
     snapshot plus the updated spectroscopic symbol line numbers (`L_new`), then updates the Pauli state.
3. Partial Overlap Refraction:
   - If the agent previously read `L1-100` (unchanged) and now requests `L80-160`, the overlapping
     prefix `L80-100` is collapsed to a 1-line note and only the new lines `L101-160` are emitted.
"""
from __future__ import annotations

import difflib
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .spectroscopy import extract_spectroscopic_entries

DEFAULT_SESSION_TTL_S = 7200.0  # 2 hours of inactivity resets the session exclusion cache
MAX_CACHED_SPANS = 250


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


@dataclass
class PauliSpanState:
    path: str
    start_line: int
    end_line: int
    sha256: str
    turn: int
    timestamp: float
    lines: list[str]


@dataclass(frozen=True)
class PauliReadOutcome:
    status: str  # "full", "unchanged", "delta", "partial_overlap"
    rendered_block: str
    lines_emitted: int
    lines_saved: int


class PauliSessionCache:
    """In-memory and `.zt/pauli_session.json`-persisted Pauli Exclusion state for the active coding session."""

    def __init__(self, state_file: Path | None = None, *, ttl_s: float = DEFAULT_SESSION_TTL_S) -> None:
        self.state_file = state_file
        self.ttl_s = ttl_s
        self.turn: int = 0
        self.last_active: float = 0.0
        self.spans: list[PauliSpanState] = []
        if state_file is not None:
            self._load()

    def _load(self) -> None:
        if self.state_file is None or not self.state_file.is_file():
            return
        try:
            raw = json.loads(self.state_file.read_text(encoding="utf-8"))
            now = time.time()
            last = float(raw.get("last_active", 0.0))
            if self.ttl_s > 0 and last > 0 and (now - last) > self.ttl_s:
                # Session expired: start fresh
                return
            self.turn = int(raw.get("turn", 0))
            self.last_active = last
            loaded: list[PauliSpanState] = []
            for item in raw.get("spans", []):
                loaded.append(
                    PauliSpanState(
                        path=str(item["path"]),
                        start_line=int(item["start_line"]),
                        end_line=int(item["end_line"]),
                        sha256=str(item["sha256"]),
                        turn=int(item.get("turn", 1)),
                        timestamp=float(item.get("timestamp", now)),
                        lines=list(item.get("lines", [])),
                    )
                )
            self.spans = loaded[-MAX_CACHED_SPANS:]
        except (OSError, ValueError, KeyError, TypeError):
            self.spans = []

    def save(self) -> None:
        if self.state_file is None:
            return
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "turn": self.turn,
                "last_active": self.last_active,
                "spans": [
                    {
                        "path": s.path,
                        "start_line": s.start_line,
                        "end_line": s.end_line,
                        "sha256": s.sha256,
                        "turn": s.turn,
                        "timestamp": s.timestamp,
                        "lines": s.lines[:400],
                    }
                    for s in self.spans[-MAX_CACHED_SPANS:]
                ],
            }
            tmp = self.state_file.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            tmp.replace(self.state_file)
        except OSError:
            pass

    def reset(self) -> None:
        self.turn = 0
        self.last_active = time.time()
        self.spans.clear()
        if self.state_file is not None and self.state_file.is_file():
            try:
                self.state_file.unlink()
            except OSError:
                pass

    def _bump_turn(self) -> int:
        now = time.time()
        if self.ttl_s > 0 and self.last_active > 0 and (now - self.last_active) > self.ttl_s:
            self.spans.clear()
            self.turn = 0
        self.turn += 1
        self.last_active = now
        return self.turn

    def find_covering_unchanged(self, path: str, start_line: int, end_line: int, text: str) -> PauliSpanState | None:
        """Return a prior state if `(path, start_line, end_line)` was already delivered in full with identical content."""
        req_lines = text.splitlines()
        req_hash = sha256_text(text)
        for s in reversed(self.spans):
            if s.path != path:
                continue
            if s.start_line == start_line and s.end_line == end_line and s.sha256 == req_hash:
                return s
            if s.start_line <= start_line and end_line <= s.end_line and s.lines:
                offset_lo = start_line - s.start_line
                offset_hi = offset_lo + len(req_lines)
                if 0 <= offset_lo <= offset_hi <= len(s.lines):
                    if s.lines[offset_lo:offset_hi] == req_lines:
                        return s
        return None

    def record_span(self, path: str, start_line: int, end_line: int, text: str, *, turn: int | None = None) -> None:
        t = turn if turn is not None else self._bump_turn()
        now = time.time()
        self.last_active = now
        digest = sha256_text(text)
        lines = text.splitlines()
        # Replace any existing state with the exact same (path, start_line, end_line)
        self.spans = [
            s for s in self.spans
            if not (s.path == path and s.start_line == start_line and s.end_line == end_line)
        ]
        self.spans.append(PauliSpanState(path, start_line, end_line, digest, t, now, lines))
        if len(self.spans) > MAX_CACHED_SPANS:
            self.spans = self.spans[-MAX_CACHED_SPANS:]

    def read_with_refraction(
        self,
        rel_path: str,
        text: str,
        first: int,
        last: int,
        total_lines: int,
        *,
        lang: str = "",
        line_numbers: bool = False,
        cut_note: str = "",
        force_full: bool = False,
    ) -> PauliReadOutcome:
        """Apply Pauli Exclusion / Wavefront Delta to a requested file span."""
        t = self._bump_turn()
        cur_lines = text.splitlines()
        n_lines = len(cur_lines)

        def fmt_lines(lines_slice: Sequence[str], start_no: int) -> str:
            if not line_numbers:
                return "\n".join(lines_slice)
            width = len(str(start_no + len(lines_slice) - 1))
            return "\n".join(f"{no:>{width}} | {ln}" for no, ln in enumerate(lines_slice, start=start_no))

        if not force_full:
            # 1. Check for exact or sub-span unchanged match (Stationary State: Delta H = 0)
            cov = self.find_covering_unchanged(rel_path, first, last, text)
            if cov is not None and n_lines >= 6:
                age_s = max(0, int(time.time() - cov.timestamp))
                first_anchor = cur_lines[0].strip()[:90] if cur_lines else ""
                last_anchor = cur_lines[-1].strip()[:90] if cur_lines else ""
                spec_entries = extract_spectroscopic_entries(rel_path, text, start_line=first)
                sym_summary = (
                    "\n> Landmarks: " + ", ".join(f"`{e.symbol}` (L{e.start_line}-{e.end_line})" for e in spec_entries[:5])
                    if spec_entries else ""
                )
                block = (
                    f"### {rel_path}  L{first}-{last} (of {total_lines}) [PAULI_EXCLUSION: UNCHANGED]{cut_note}\n"
                    f"> State `sha256={cov.sha256[:10]}` already in session context (turn #{cov.turn}, {age_s}s ago). "
                    f"Pass `{rel_path}:L{first}-{last}!` to force re-emission.\n"
                    f"> Boundary anchors: L{first} `{first_anchor}` ... L{last} `{last_anchor}`"
                    f"{sym_summary}"
                )
                return PauliReadOutcome("unchanged", block, lines_emitted=3, lines_saved=max(0, n_lines - 3))

            # 2. Check for perturbed state on the same file region (file edited mid-session: Delta H != 0, IoU >= 0.70)
            best_prior: PauliSpanState | None = None
            best_iou = 0.0
            for s in reversed(self.spans):
                if s.path != rel_path or not s.lines:
                    continue
                inter_lo = max(first, s.start_line)
                inter_hi = min(last, s.end_line)
                if inter_hi >= inter_lo:
                    inter = inter_hi - inter_lo + 1
                    union = max(last, s.end_line) - min(first, s.start_line) + 1
                    iou = inter / max(1, union)
                    if iou > best_iou:
                        best_iou = iou
                        best_prior = s

            if best_prior is not None and best_iou >= 0.70 and n_lines >= 12:
                diff_lines = list(
                    difflib.unified_diff(
                        best_prior.lines,
                        cur_lines,
                        fromfile=f"{rel_path}:L{best_prior.start_line}-{best_prior.end_line}@turn{best_prior.turn}",
                        tofile=f"{rel_path}:L{first}-{last}@now",
                        n=3,
                        lineterm="",
                    )
                )
                if 0 < len(diff_lines) <= max(10, int(n_lines * 0.55)):
                    self.record_span(rel_path, first, last, text, turn=t)
                    spec_entries = extract_spectroscopic_entries(rel_path, text, start_line=first)
                    sym_map = (
                        "\n> Updated landmarks: " + ", ".join(f"`{e.symbol}` -> L{e.start_line}-{e.end_line}" for e in spec_entries[:6])
                        if spec_entries else ""
                    )
                    diff_body = "\n".join(diff_lines)
                    block = (
                        f"### {rel_path}  L{first}-{last} (of {total_lines}) [PAULI_WAVEFRONT_DELTA vs turn #{best_prior.turn}]{cut_note}"
                        f"{sym_map}\n```diff\n{diff_body}\n```"
                    )
                    return PauliReadOutcome(
                        "delta",
                        block,
                        lines_emitted=len(diff_lines) + 2,
                        lines_saved=max(0, n_lines - len(diff_lines) - 2),
                    )

            # 3. Check for Partial Overlap Refraction (prefix or suffix overlap >= 12 unchanged lines)
            for s in reversed(self.spans):
                if s.path != rel_path or not s.lines:
                    continue
                inter_lo = max(first, s.start_line)
                inter_hi = min(last, s.end_line)
                overlap_len = inter_hi - inter_lo + 1
                if overlap_len >= 12 and (inter_lo == first or inter_hi == last) and overlap_len < n_lines:
                    prior_slice = s.lines[inter_lo - s.start_line : inter_hi - s.start_line + 1]
                    cur_slice = cur_lines[inter_lo - first : inter_hi - first + 1]
                    if prior_slice == cur_slice:
                        if inter_lo == first:
                            new_first, new_last = inter_hi + 1, last
                            new_slice = cur_lines[new_first - first :]
                        else:
                            new_first, new_last = first, inter_lo - 1
                            new_slice = cur_lines[: new_last - first + 1]
                        if new_slice:
                            self.record_span(rel_path, first, last, text, turn=t)
                            body = fmt_lines(new_slice, new_first)
                            block = (
                                f"### {rel_path}  L{first}-{last} (of {total_lines}) [PAULI_PARTIAL_REFRACTION]{cut_note}\n"
                                f"> Lines L{inter_lo}-{inter_hi} ({overlap_len} lines) unchanged from turn #{s.turn}; "
                                f"emitting only unseen L{new_first}-{new_last}:\n"
                                f"```{lang}\n{body}\n```"
                            )
                            return PauliReadOutcome(
                                "partial_overlap",
                                block,
                                lines_emitted=len(new_slice) + 2,
                                lines_saved=overlap_len,
                            )

        # 3. Brand new span (or forced full read): record and emit in full
        self.record_span(rel_path, first, last, text, turn=t)
        body = fmt_lines(cur_lines, first)
        block = f"### {rel_path}  L{first}-{last} (of {total_lines}){cut_note}\n```{lang}\n{body}\n```"
        return PauliReadOutcome("full", block, lines_emitted=n_lines, lines_saved=0)
