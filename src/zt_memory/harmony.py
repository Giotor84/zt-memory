"""Harmonic Resonance & Impedance-Matched Call Coupling in the Poincaré Ball.

Physical & Harmonic Formulation:
1. Octave Separation (Fundamental f_0 vs Upper Octave 2*f_0):
   - Production symbols live on the fundamental band (Omega_0).
   - Verification/test symbols (`tests/`, `test_*.py`) live on the first harmonic octave (Omega_1 = 2*Omega_0).
   - In compact routing (`locations` / `focus`), the fundamental band is protected from destructive crowding
     by the test octave: production symbols always anchor the dynamic cutoff and receive primary bandwidth,
     while the upper octave is compressed into a resonant verification band.

2. Impedance-Matched Coupled Oscillators (Call-Site Micro-Slicing):
   - Unweighted static call graphs explode on high-degree utility hubs ("acoustic short-circuit").
   - When a fundamental symbol `u` is excited by query `q` with amplitude `s_u`, energy transmission to a
     coupled caller `v` (verified via AST call-site extraction at line `L_call`) is governed by:
       * Acoustic Impedance of target `u`: Z_u = sqrt(1 + deg_in(u))
       * Phase Consonance in Poincaré/SVD space: C(v, q) = 0.5 * (1 + cos(theta_{v, q}))
       * Newtonian Thermal Gain: G(v) = 1 + 0.25 * log(1 + E_v)
       * Resonance Amplitude: R(u -> v) = (s_u / Z_u) * C(v, q) * G(v)
   - Instead of loading the entire caller chunk (100-300 tokens), only the 1-line resonant call-site
     micro-slice (`path:L_call [caller -> target]  call_expr`) is emitted (~14 tokens).
"""
from __future__ import annotations

import ast
import math
import re
import textwrap
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import numpy as np

from .config import NodeKind
from .discovery import is_excluded
from .outline import short_name

if TYPE_CHECKING:
    from .manager import NewtonianMemoryManager
    from .storage import NodeRecord

_PYTHON_BUILTINS = frozenset({
    "abs", "all", "any", "ascii", "bin", "bool", "breakpoint", "bytearray", "bytes",
    "callable", "chr", "classmethod", "compile", "complex", "delattr", "dict", "dir",
    "divmod", "enumerate", "eval", "exec", "filter", "float", "format", "frozenset",
    "getattr", "globals", "hasattr", "hash", "help", "hex", "id", "input", "int",
    "isinstance", "issubclass", "iter", "len", "list", "locals", "map", "max",
    "memoryview", "min", "next", "object", "oct", "open", "ord", "pow", "print",
    "property", "range", "repr", "reversed", "round", "set", "setattr", "slice",
    "sorted", "staticmethod", "str", "sum", "super", "tuple", "type", "vars", "zip",
    "ValueError", "TypeError", "KeyError", "IndexError", "AttributeError", "RuntimeError",
    "Exception", "BaseException", "StopIteration", "NotImplementedError", "OSError",
    "append", "extend", "get", "items", "keys", "values", "strip", "split", "join",
    "replace", "format", "lower", "upper", "startswith", "endswith", "pop", "update",
})

_CALL_RE = re.compile(r"(?:\b([A-Za-z_][A-Za-z0-9_]*)\s*\(|^@(?:[A-Za-z_][A-Za-z0-9_]*\.)*([A-Za-z_][A-Za-z0-9_]*))")


@dataclass(frozen=True, slots=True)
class CallSite:
    callee: str
    line: int  # 1-based file line number
    snippet: str


@dataclass(frozen=True, slots=True)
class ResonantCallSlice:
    caller_id: int
    caller_path: str
    caller_symbol: str
    target_symbol: str
    call_line: int
    snippet: str
    resonance: float


@dataclass(frozen=True, slots=True)
class ResonantTestTarget:
    node_id: int
    path: str
    symbol: str
    pytest_target: str
    start_line: int
    end_line: int
    resonance: float
    coupled_to: str | None = None


@dataclass(frozen=True, slots=True)
class HarmonicField:
    call_slices: tuple[ResonantCallSlice, ...] = ()
    resonant_tests: tuple[ResonantTestTarget, ...] = ()


def is_test_octave(path: str | None) -> bool:
    """True when `path` belongs to the upper verification octave (2*f_0)."""
    if not path:
        return False
    parts = [p.lower() for p in path.replace("\\", "/").split("/") if p]
    if any(p in ("test", "tests", "testing", "__tests__", "spec", "specs") for p in parts[:-1]):
        return True
    name = parts[-1] if parts else ""
    return name.startswith("test_") or name.endswith(("_test.py", "_tests.py", ".test.ts", ".spec.ts", ".test.js", "_test.go"))


def extract_call_sites(text: str, start_line: int = 1, *, target_name: str | None = None) -> list[CallSite]:
    """Extract function/method call sites from a chunk's source text with 1-based file line numbers."""
    lines = text.splitlines()
    if not lines:
        return []
    sites: list[CallSite] = []
    seen: set[tuple[str, int]] = set()
    try:
        tree = ast.parse(textwrap.dedent(text))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                for dec in node.decorator_list:
                    dname = dec.id if isinstance(dec, ast.Name) else (dec.attr if isinstance(dec, ast.Attribute) else None)
                    if dname and dname not in _PYTHON_BUILTINS and (target_name is None or dname == target_name):
                        rel_lineno = getattr(dec, "lineno", 1)
                        abs_line = start_line + rel_lineno - 1
                        if (dname, abs_line) not in seen:
                            seen.add((dname, abs_line))
                            idx = max(0, min(len(lines) - 1, rel_lineno - 1))
                            sites.append(CallSite(dname, abs_line, " ".join(lines[idx].strip().split())[:88]))
            if isinstance(node, ast.Call):
                name: str | None = None
                if isinstance(node.func, ast.Name):
                    name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    name = node.func.attr
                if not name or name in _PYTHON_BUILTINS:
                    continue
                if target_name is not None and name != target_name:
                    continue
                rel_lineno = getattr(node, "lineno", 1)
                abs_line = start_line + rel_lineno - 1
                if (name, abs_line) in seen:
                    continue
                seen.add((name, abs_line))
                idx = max(0, min(len(lines) - 1, rel_lineno - 1))
                snippet = " ".join(lines[idx].strip().split())
                if len(snippet) > 88:
                    snippet = snippet[:85] + "..."
                sites.append(CallSite(name, abs_line, snippet))
        return sites
    except SyntaxError:
        pass

    # Fallback line-by-line regex when chunk is a partial syntax block or non-Python
    for idx, raw in enumerate(lines):
        stripped = raw.strip()
        if not stripped or stripped.startswith(("#", "//", "def ", "class ", "async def ")):
            continue
        abs_line = start_line + idx
        for m in _CALL_RE.finditer(stripped):
            name = m.group(1) or m.group(2)
            if name in _PYTHON_BUILTINS or (target_name is not None and name != target_name):
                continue
            if (name, abs_line) in seen:
                continue
            seen.add((name, abs_line))
            snippet = " ".join(stripped.split())
            if len(snippet) > 88:
                snippet = snippet[:85] + "..."
            sites.append(CallSite(name, abs_line, snippet))
    return sites


def _to_pytest_target(path: str, symbol: str) -> str:
    clean = symbol.split("#")[0].split("~")[0]
    parts = [p for p in clean.split(".") if p and not p.startswith("<")]
    return f"{path}::{'::'.join(parts)}" if parts else path


def compute_harmonic_resonance(
    manager: "NewtonianMemoryManager",
    z_q: np.ndarray,
    fundamental_records: Sequence[tuple["NodeRecord", float]],
    *,
    candidate_test_records: Sequence[tuple["NodeRecord", float]] = (),
    max_fundamental_drivers: int = 3,
    max_call_slices: int = 3,
    max_tests: int = 2,
) -> HarmonicField:
    """Couple excited fundamental symbols (Omega_0) to callers (Omega_0) and resonant tests (Omega_1)."""
    if not fundamental_records:
        return HarmonicField()

    q_norm = np.linalg.norm(z_q)
    q_hat = z_q / max(float(q_norm), 1e-12)
    fundamental_ids = {rec.id for rec, _ in fundamental_records}

    raw_callers: list[tuple[int, "NodeRecord", "NodeRecord", float, CallSite, float]] = []
    # (caller_id, caller_rec, driver_rec, driver_score, call_site, impedance)
    coupled_tests: dict[int, tuple["NodeRecord", float, str | None]] = {}

    drivers = [
        (rec, score)
        for rec, score in fundamental_records
        if rec.kind == NodeKind.CHUNK and rec.symbol and not rec.symbol.startswith("<")
    ][:max_fundamental_drivers]

    for driver_rec, driver_score in drivers:
        sym_name = short_name(driver_rec.symbol or "")
        if not sym_name or len(sym_name) < 2 or sym_name in _PYTHON_BUILTINS:
            continue
        # Fast lexical lookup of all chunks mentioning `sym_name`
        lex_hits = manager.lexical_search(sym_name, 32, frozenset({NodeKind.CHUNK}))
        cand_ids = [h.node_id for h in lex_hits if h.node_id != driver_rec.id]
        if not cand_ids:
            continue
        cand_recs = manager.records(cand_ids)
        verified_ForDriver: list[tuple["NodeRecord", CallSite]] = []
        for cid in cand_ids:
            crec = cand_recs.get(cid)
            if crec is None or crec.kind != NodeKind.CHUNK or not crec.path or crec.start_line is None or is_excluded(crec.path, manager.config):
                continue
            if (crec.symbol or "<").startswith("<") or "<gap:" in (crec.symbol or "") or "<block:" in (crec.symbol or ""):
                continue
            # Avoid self-calls inside the exact same symbol
            if crec.path == driver_rec.path and (crec.symbol or "").split("#")[0] == (driver_rec.symbol or "").split("#")[0]:
                continue
            text = manager.read_chunk(crec)
            if not text:
                continue
            sites = extract_call_sites(text, crec.start_line, target_name=sym_name)
            if sites:
                verified_ForDriver.append((crec, sites[0]))

        # Acoustic impedance Z_u = sqrt(1 + in_degree(u)) damps ubiquitous utility hubs
        in_degree = len(verified_ForDriver)
        impedance = math.sqrt(1.0 + in_degree)

        for crec, site in verified_ForDriver:
            if is_test_octave(crec.path):
                prev = coupled_tests.get(crec.id)
                base_amp = float(driver_score) / impedance
                if prev is None or base_amp > prev[1]:
                    coupled_tests[crec.id] = (crec, base_amp, sym_name)
            else:
                raw_callers.append((crec.id, crec, driver_rec, float(driver_score), site, impedance))

    # Also include candidate test chunks from the upper octave that resonated directly with the query
    for trec, tscore in candidate_test_records:
        if trec.kind == NodeKind.CHUNK and trec.path and trec.start_line is not None and trec.end_line is not None:
            if trec.id not in coupled_tests:
                coupled_tests[trec.id] = (trec, float(tscore) * 0.75, None)

    # Batch-evaluate phase consonance C(v, q) and Newtonian energy gain G(v) in Poincaré ball
    all_eval_ids = list({c[0] for c in raw_callers} | set(coupled_tests.keys()))
    consonance_map: dict[int, float] = {}
    energy_gain_map: dict[int, float] = {}
    if all_eval_ids:
        arr_ids = np.array(all_eval_ids, dtype=np.int64)
        pos = manager.positions(arr_ids)
        norms = np.maximum(np.linalg.norm(pos, axis=1, keepdims=True), 1e-12)
        unit_pos = pos / norms
        cos_theta = unit_pos @ q_hat
        energies = manager.energies(arr_ids)
        for idx, nid in enumerate(all_eval_ids):
            consonance_map[nid] = float(0.5 * (1.0 + np.clip(cos_theta[idx], -1.0, 1.0)))
            energy_gain_map[nid] = float(1.0 + 0.25 * math.log1p(max(0.0, float(energies[idx]))))

    # Rank production call-site micro-slices
    call_slices: list[ResonantCallSlice] = []
    seen_callers: set[tuple[str, str]] = set()
    for cid, crec, drec, dscore, site, impedance in raw_callers:
        if cid in fundamental_ids:
            # Already shown as a primary fundamental symbol; still valuable if it's a distinct file caller,
            # but prioritize unseen callers first
            scale = 0.65
        else:
            scale = 1.0
        res = scale * (dscore / impedance) * consonance_map.get(cid, 0.5) * energy_gain_map.get(cid, 1.0)
        key = (crec.path or "", (crec.symbol or "").split("#")[0])
        if key in seen_callers:
            continue
        seen_callers.add(key)
        call_slices.append(
            ResonantCallSlice(
                caller_id=cid,
                caller_path=crec.path or "",
                caller_symbol=crec.symbol or "?",
                target_symbol=short_name(drec.symbol or "?"),
                call_line=site.line,
                snippet=site.snippet,
                resonance=res,
            )
        )
    call_slices.sort(key=lambda s: -s.resonance)

    # Rank upper-octave resonant tests
    test_targets: list[ResonantTestTarget] = []
    seen_test_syms: set[str] = set()
    for tid, (trec, base_amp, coupled_sym) in coupled_tests.items():
        if not trec.path or trec.start_line is None or trec.end_line is None:
            continue
        pytest_id = _to_pytest_target(trec.path, trec.symbol or "")
        if pytest_id in seen_test_syms:
            continue
        seen_test_syms.add(pytest_id)
        harmonic_boost = 1.35 if coupled_sym is not None else 1.0
        res = base_amp * harmonic_boost * consonance_map.get(tid, 0.5) * energy_gain_map.get(tid, 1.0)
        test_targets.append(
            ResonantTestTarget(
                node_id=tid,
                path=trec.path,
                symbol=trec.symbol or "?",
                pytest_target=pytest_id,
                start_line=trec.start_line,
                end_line=trec.end_line,
                resonance=res,
                coupled_to=coupled_sym,
            )
        )
    test_targets.sort(key=lambda t: -t.resonance)

    return HarmonicField(
        call_slices=tuple(call_slices[:max_call_slices]),
        resonant_tests=tuple(test_targets[:max_tests]),
    )
