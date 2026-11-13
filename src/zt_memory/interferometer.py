"""Dissipative Interferometer: Destructive Test-Output Interference & Newtonian Shockwave Feedback.

Physical & Harmonic Formulation:
1. Destructive Interference of the Stationary Wave (psi_bg -> 0):
   - Standard test runners (pytest, unittest, cargo test) emit high-entropy stationary background noise:
     session headers, plugin listings, progress dots, passing test names, external `site-packages` frames,
     and repeated warnings.
   - In an interferometer, the reference wave cancels the stationary background in anti-phase:
     when all tests pass (Delta psi == 0), output collapses to a single harmonic lock line (~12 tokens).

2. Constructive Interference at Symmetry-Breaking Faults (Delta psi != 0):
   - When a failure occurs, only the symmetry-breaking wave packet survives:
     * Failing test node ID (`path::test_name`)
     * Innermost repository fault frame (`repo_path:line in symbol`) + 1-line fault code
     * Condensed `E   <ExcType>: <delta>` assertion lines (deduplicated across parametrized runs)
   - External library frames (`site-packages`, `_pytest`, `pluggy`, `unittest`) are destructively canceled.

3. Newtonian Shockwave Feedback (F_shock -> v_t -> E_t) with Call-Edge Transmission:
   - A test failure is a physical surprise event in the Poincaré ball.
   - The interferometer resolves every repository frame in the failure wave packet to its `CHUNK` node in
     `NewtonianMemoryManager`. Moreover, when a test fails on an assertion (`assert fn(x) == y` or
     `DID NOT RAISE`) where the production function `fn` already returned and is not in the traceback,
     the shockwave extracts the AST call sites in the failing test window and transmits the Newtonian
     impulse across the call edge directly into the production `CHUNK` node `fn`!
"""
from __future__ import annotations

import math
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from .config import NodeKind
from .harmony import extract_call_sites, is_test_octave
from .outline import short_name
from .stacktrace import PathResolver

if TYPE_CHECKING:
    from .manager import NewtonianMemoryManager

_SUMMARY_RE = re.compile(
    r"=+\s*((?:\d+\s+(?:failed|passed|skipped|deselected|xfailed|xpassed|warnings?|errors?)[,\s]*)+)"
    r"(?:in\s+([\d.]+s))?\s*=+",
    re.IGNORECASE,
)
_FAIL_HEADER_RE = re.compile(r"^_{3,}\s+(.+?)\s+_{3,}$")
_SHORT_FAIL_RE = re.compile(r"^(?:FAILED|ERROR)\s+([^\s]+)(?:\s+-\s+(.+))?$")
_FRAME_LINE_RE = re.compile(r"^([^\s:][^:]*\.py):(\d+):\s*(?:in\s+([A-Za-z0-9_<>.]+))?")
_EXTERNAL_MARKERS = ("site-packages", "_pytest", "pluggy", "importlib", "unittest\\case.py", "unittest/case.py", ".venv")


@dataclass(frozen=True, slots=True)
class FaultWavePacket:
    test_node: str
    exc_summary: str
    repo_frames: tuple[tuple[str, int, str, str], ...]  # (rel_path, line, func, code_line)
    assertion_lines: tuple[str, ...]
    repeat_count: int = 1


@dataclass(frozen=True, slots=True)
class InterferometerReport:
    passed: bool
    summary_line: str
    duration_s: float
    raw_tokens: int
    compact_tokens: int
    compression_ratio: float
    compact_output: str
    faults: tuple[FaultWavePacket, ...] = ()
    heated_nodes: tuple[tuple[int, str, float], ...] = ()  # (node_id, "path::symbol", new_energy)


def _estimate_tokens(text: str, chars_per_token: float = 3.2) -> int:
    return max(1, math.ceil(len(text) / chars_per_token)) if text else 0


def _is_external_frame(path_str: str) -> bool:
    norm = path_str.replace("\\", "/").lower()
    return any(m in norm for m in _EXTERNAL_MARKERS)


def compress_pytest_output(
    raw_output: str,
    *,
    exit_code: int = 0,
    repo_files: Sequence[str] = (),
    max_tokens: int = 220,
) -> InterferometerReport:
    """Apply destructive interference to raw pytest stdout/stderr."""
    raw_tokens = _estimate_tokens(raw_output)
    lines = raw_output.splitlines()

    summary_match = None
    for ln in reversed(lines):
        m = _SUMMARY_RE.search(ln)
        if m:
            summary_match = m
            break

    status_counts = summary_match.group(1).strip().rstrip(",") if summary_match else ("passed" if exit_code == 0 else "failed")
    duration_str = summary_match.group(2) if (summary_match and summary_match.group(2)) else "0.00s"
    try:
        duration_s = float(duration_str.rstrip("s"))
    except ValueError:
        duration_s = 0.0

    passed = exit_code == 0 and "failed" not in status_counts.lower() and "error" not in status_counts.lower()

    if passed:
        compact = f"PASS [{status_counts} in {duration_str}] -- stationary wave canceled (0 faults)"
        c_tokens = _estimate_tokens(compact)
        ratio = 1.0 - (c_tokens / max(raw_tokens, 1))
        return InterferometerReport(
            passed=True,
            summary_line=compact,
            duration_s=duration_s,
            raw_tokens=raw_tokens,
            compact_tokens=c_tokens,
            compression_ratio=max(0.0, ratio),
            compact_output=compact,
        )

    resolver = PathResolver(repo_files) if repo_files else None

    sections: list[tuple[str, list[str]]] = []
    current_title: str | None = None
    current_buf: list[str] = []
    short_failures: dict[str, str] = {}

    for ln in lines:
        sm = _SHORT_FAIL_RE.match(ln.strip())
        if sm:
            short_failures[sm.group(1)] = (sm.group(2) or "").strip()
            continue
        hm = _FAIL_HEADER_RE.match(ln.strip())
        if hm:
            if current_title is not None:
                sections.append((current_title, current_buf))
            current_title = hm.group(1).strip()
            current_buf = []
            continue
        if current_title is not None:
            if ln.startswith("====") and ("short test summary" in ln.lower() or _SUMMARY_RE.search(ln)):
                sections.append((current_title, current_buf))
                current_title = None
                current_buf = []
            else:
                current_buf.append(ln)
    if current_title is not None and current_buf:
        sections.append((current_title, current_buf))

    grouped_faults: dict[tuple[str, str], FaultWavePacket] = {}
    for title, buf in sections:
        repo_frames: list[tuple[str, int, str, str]] = []
        assertions: list[str] = []
        last_code_line = ""
        for bln in buf:
            stripped = bln.strip()
            if bln.startswith("E   "):
                err_line = " ".join(bln[4:].strip().split())
                if len(err_line) > 120:
                    err_line = err_line[:117] + "..."
                if err_line and err_line not in assertions:
                    assertions.append(err_line)
            elif stripped.startswith(">"):
                last_code_line = " ".join(stripped[1:].strip().split())
                if len(last_code_line) > 96:
                    last_code_line = last_code_line[:93] + "..."
            else:
                fm = _FRAME_LINE_RE.match(stripped)
                if fm:
                    fpath, fline_s, ffunc = fm.group(1), fm.group(2), fm.group(3) or "?"
                    if _is_external_frame(fpath):
                        continue
                    fline = int(fline_s)
                    rel = resolver.resolve(fpath) if resolver is not None else fpath.replace("\\", "/")
                    if rel is None:
                        if not _is_external_frame(fpath):
                            rel = fpath.replace("\\", "/")
                        else:
                            continue
                    repo_frames.append((rel, fline, ffunc, last_code_line))

        kept_frames = tuple(repo_frames[-2:])
        exc_summary = assertions[-1] if assertions else short_failures.get(title, "Test failure")
        sig_key = (
            f"{kept_frames[-1][0]}:{kept_frames[-1][1]}" if kept_frames else title,
            exc_summary[:80],
        )
        if sig_key in grouped_faults:
            prev = grouped_faults[sig_key]
            grouped_faults[sig_key] = FaultWavePacket(
                test_node=prev.test_node,
                exc_summary=prev.exc_summary,
                repo_frames=prev.repo_frames,
                assertion_lines=prev.assertion_lines,
                repeat_count=prev.repeat_count + 1,
            )
        else:
            grouped_faults[sig_key] = FaultWavePacket(
                test_node=title,
                exc_summary=exc_summary,
                repo_frames=kept_frames,
                assertion_lines=tuple(assertions[:3]),
                repeat_count=1,
            )

    if not grouped_faults and short_failures:
        for tnode, msg in list(short_failures.items())[:4]:
            grouped_faults[(tnode, msg)] = FaultWavePacket(
                test_node=tnode,
                exc_summary=msg or "failed",
                repo_frames=(),
                assertion_lines=(msg,) if msg else (),
            )

    fault_list = tuple(grouped_faults.values())
    out_lines = [f"FAIL [{status_counts} in {duration_str}] -- {len(fault_list)} distinct fault wave(s):"]
    used = _estimate_tokens(out_lines[0])
    for fw in fault_list:
        mult = f" (x{fw.repeat_count})" if fw.repeat_count > 1 else ""
        block_lines = [f"- [!] {fw.test_node}{mult}: {fw.exc_summary}"]
        for rel, fline, ffunc, code_ln in fw.repo_frames:
            code_part = f" -> `{code_ln}`" if code_ln else ""
            block_lines.append(f"    at {rel}:{fline} in {ffunc}{code_part}")
        for aln in fw.assertion_lines[:-1]:
            block_lines.append(f"    E: {aln}")
        block_text = "\n".join(block_lines)
        cost = _estimate_tokens(block_text)
        if len(out_lines) > 1 and used + cost > max_tokens:
            out_lines.append(f"- ... ({len(fault_list) - len(out_lines) + 1} more fault waves suppressed by budget)")
            break
        out_lines.append(block_text)
        used += cost

    compact = "\n".join(out_lines)
    c_tokens = _estimate_tokens(compact)
    ratio = 1.0 - (c_tokens / max(raw_tokens, 1))
    return InterferometerReport(
        passed=False,
        summary_line=out_lines[0],
        duration_s=duration_s,
        raw_tokens=raw_tokens,
        compact_tokens=c_tokens,
        compression_ratio=max(0.0, ratio),
        compact_output=compact,
        faults=fault_list,
    )


def inject_failure_shockwave(
    manager: "NewtonianMemoryManager",
    report: InterferometerReport,
    *,
    shock_scale: float = 2.5,
) -> InterferometerReport:
    """Inject Newtonian surprise force into the CHUNK nodes where fault waves peaked, plus production symbols
    called at the failing test site (when a test fails on an assertion after the production call returned)."""
    if report.passed or not report.faults:
        return report

    with manager.lock:
        resolver = PathResolver(manager.file_paths())
        target_ids: list[int] = []
        weights: list[float] = []
        prod_chunks = [r for r in manager.meta.by_kind(NodeKind.CHUNK) if r.path and not is_test_octave(r.path)]
        by_short_sym: dict[str, list[int]] = {}
        for r in prod_chunks:
            sname = short_name(r.symbol or "")
            if sname and not sname.startswith("<"):
                by_short_sym.setdefault(sname, []).append(r.id)

        for fw in report.faults:
            for idx, (rel, fline, _, code_ln) in enumerate(fw.repo_frames):
                resolved = resolver.resolve(rel) or rel
                chunks = [
                    c for c in manager.meta.chunks_of_file(resolved)
                    if c.start_line is not None and c.end_line is not None and c.start_line <= fline <= c.end_line
                ]
                if chunks:
                    innermost = min(chunks, key=lambda c: (c.end_line - c.start_line, c.start_line))
                    w = (1.5 if idx == len(fw.repo_frames) - 1 else 0.8) * math.sqrt(fw.repeat_count)
                    target_ids.append(innermost.id)
                    weights.append(w)

                # If the frame is in a test file, transmit the shockwave across AST calls in the failing test window
                if is_test_octave(resolved):
                    window_text = manager.read_lines(resolved, max(1, fline - 14), fline + 4) or code_ln
                    if window_text:
                        for site in extract_call_sites(window_text, start_line=max(1, fline - 14)):
                            callee_ids = by_short_sym.get(site.callee, [])
                            if callee_ids and len(callee_ids) <= 4:
                                imp = math.sqrt(len(callee_ids))
                                for cid in callee_ids:
                                    target_ids.append(cid)
                                    weights.append(1.8 * math.sqrt(fw.repeat_count) / imp)

        if not target_ids:
            return report

        epicenter = manager.positions([target_ids[-1]])[0]
        manager.reinforce(target_ids, epicenter, weights, scale=shock_scale)
        uniq_ids = list(dict.fromkeys(target_ids))
        recs = manager.records(uniq_ids)
        new_energies = manager.energies(uniq_ids)
        # Prioritize showing heated production symbols first
        heated_pairs = [
            (nid, f"{recs[nid].path}::{recs[nid].symbol}", round(float(e), 4), is_test_octave(recs[nid].path))
            for nid, e in zip(uniq_ids, new_energies)
            if nid in recs
        ]
        heated_pairs.sort(key=lambda t: (t[3], -t[2]))
        heated = tuple((nid, sym, e) for nid, sym, e, _ in heated_pairs)
        if heated:
            heat_line = "  heated in memory: " + ", ".join(f"{sym} (E={e:.2f})" for _, sym, e in heated[:3])
            compact_with_heat = report.compact_output + "\n" + heat_line
        else:
            compact_with_heat = report.compact_output

        return InterferometerReport(
            passed=report.passed,
            summary_line=report.summary_line,
            duration_s=report.duration_s,
            raw_tokens=report.raw_tokens,
            compact_tokens=_estimate_tokens(compact_with_heat),
            compression_ratio=1.0 - (_estimate_tokens(compact_with_heat) / max(report.raw_tokens, 1)),
            compact_output=compact_with_heat,
            faults=report.faults,
            heated_nodes=heated,
        )


def run_interferometer_verify(
    repo_root: Path,
    targets: Sequence[str] = (),
    *,
    manager: "NewtonianMemoryManager | None" = None,
    python_exe: str | None = None,
    timeout_s: float = 60.0,
    max_tokens: int = 220,
) -> InterferometerReport:
    """Execute pytest on `targets` (or resonant tests from `manager`), cancel background wave, and inject shockwave."""
    py = python_exe or sys.executable
    cmd = [py, "-m", "pytest", "--tb=short", "-q", "--no-header", *targets]
    proc = subprocess.run(
        cmd,
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    raw = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    repo_files = manager.file_paths() if manager is not None else ()
    rep = compress_pytest_output(raw, exit_code=proc.returncode, repo_files=repo_files, max_tokens=max_tokens)
    if manager is not None and not rep.passed:
        rep = inject_failure_shockwave(manager, rep)
    return rep
