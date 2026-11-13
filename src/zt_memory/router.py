"""Stage 2: geometric routing from the SSD memory to the VRAM compute engine.

Pipeline for a query q:
  1. q -> hidden state -> SVD subspace -> point z_q on the ball (CHUNK level).
  2. Seeds: exact/approx k-NN by Poincaré distance among content nodes. Within one node level the
     ranking equals cosine similarity of the embeddings; the radius only separates levels (file vs symbol).
  3. Lexical tier: FTS5/BM25 over code-aware tokens finds exact identifiers, error codes and
     quoted literals that the compressed geometry cannot resolve; its hits join the candidates.
  4. Expansion from the top seeds of both lists: explicit edges (spec -> implementing code, decisions)
     and the header (imports, constants) of their files.
  5. Fusion (Reciprocal Rank Fusion): rel = w_g/(k+1+rank_geo) + w_l/(k+1+rank_lex), with w_l
     boosted for code-like queries; score = kind_weight * rel/max(rel) * (1 + w_E * norm. log-energy).
  6. Locations named in error output in the query (tracebacks, compiler messages) are resolved to the
     symbols at those lines and pinned ahead of the ranked candidates (stacktrace.py).
  7. Materialize exact text (code from disk with hash check, specs from metadata),
     octave-separated harmonic packing under the token budget, merge overlapping spans, reinforce selected nodes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import PurePosixPath
from typing import Callable, Literal, NamedTuple, Sequence

import numpy as np
import torch

from . import hyperbolic as H
from .discovery import is_excluded
from .config import TEXT_KINDS, Mode, NodeKind, RouterConfig
from .harmony import HarmonicField, compute_harmonic_resonance
from .lexical import LexicalHit, build_query
from .manager import NewtonianMemoryManager
from .outline import outline, short_name
from .pauli import PauliSessionCache
from .spectroscopy import is_spectroscopic_path, spectroscopic_outlines_for_span
from .stacktrace import PathResolver, parse_locations
from .storage import NodeRecord

SEED_KINDS = frozenset({NodeKind.CHUNK, NodeKind.SPEC, NodeKind.DECISION, NodeKind.EPHEMERAL})

_LANG = {
    ".py": "python", ".pyi": "python", ".js": "javascript", ".jsx": "jsx", ".ts": "typescript", ".tsx": "tsx",
    ".go": "go", ".rs": "rust", ".java": "java", ".kt": "kotlin", ".c": "c", ".h": "c", ".cpp": "cpp",
    ".cc": "cpp", ".hpp": "cpp", ".cs": "csharp", ".rb": "ruby", ".php": "php", ".swift": "swift",
    ".sh": "bash", ".sql": "sql", ".yaml": "yaml", ".yml": "yaml", ".toml": "toml", ".json": "json",
    ".md": "markdown", ".html": "html", ".htm": "html", ".css": "css",
}

DEFAULT_SYSTEM_PROMPT = (
    "You are the compute engine of a navigator-guided coding system. The context below was selected "
    "geometrically from the project memory and contains exact source spans with their line ranges. "
    "Treat it as the relevant subset of the project; if something essential is missing, say exactly what."
)

_CONTRACT = {
    "brownfield": (
        "Return the change as unified diffs (git format: `--- a/<path>` / `+++ b/<path>`) against the files "
        "shown. Keep edits minimal and confined to the shown spans unless the task requires otherwise."
    ),
    "greenfield": (
        "Emit every new or modified file as a fenced code block whose info string is `path=<relative/path>`, "
        "containing the complete file. Respect the repository map and satisfy every listed requirement."
    ),
}


@dataclass(frozen=True)
class ContextBlock:
    kind: NodeKind
    node_ids: tuple[int, ...]
    title: str
    text: str
    score: float
    distance: float
    energy: float
    tokens: int
    path: str | None = None
    start_line: int | None = None
    end_line: int | None = None
    geo_rank: int | None = None  # 0-based rank in the geometric list
    lex_rank: int | None = None  # 0-based rank in the lexical list, None if not a lexical hit
    symbols: tuple[str, ...] = ()  # symbols of the chunks merged into this block
    full: bool = True  # False: only the signature lines are shown (focused context)


class LocationList(NamedTuple):
    text: str
    entries: int
    tokens: int
    files: tuple[str, ...]


@dataclass(frozen=True)
class RoutedContext:
    query: str
    mode: Mode
    blocks: tuple[ContextBlock, ...]
    file_map: tuple[str, ...]
    token_estimate: int
    token_budget: int
    pinned: tuple[str, ...] = ()  # "path:line" locations from the query's error output, pinned first
    selection: tuple[tuple[int, float], ...] = ()  # (node id, score) in the order the router chose them (diagnostics)
    harmonic_field: HarmonicField | None = None

    @property
    def node_ids(self) -> tuple[int, ...]:
        return tuple(i for b in self.blocks for i in b.node_ids)

    def render(self, *, line_numbers: bool = False, contract: str | None = None) -> str:
        out: list[str] = [f"MODE: {self.mode.upper()}", self.render_context(line_numbers=line_numbers)]
        out.append(f"\n## Task\n{self.query.strip()}")
        out.append(f"\n## Output contract\n{contract or _CONTRACT[self.mode]}")
        return "\n".join(out)

    def render_context(self, *, line_numbers: bool = False, map_hint: str = "ask for a body if you need it") -> str:
        """Requirements, repository map and code spans only: for clients that bring their own task and contract (MCP)."""
        out: list[str] = []
        texts = [b for b in self.blocks if b.kind in TEXT_KINDS]
        code = [b for b in self.blocks if b.kind not in TEXT_KINDS and b.full]
        outlines = [b for b in self.blocks if b.kind not in TEXT_KINDS and not b.full]
        if texts:
            out.append("\n## Requirements & decisions")
            for b in texts:
                out.append(f"- [{b.kind.name} #{b.node_ids[0]}] {b.text.strip()}")
        if self.file_map:
            out.append("\n## Repository map")
            out.append(_render_tree(self.file_map))
        if code:
            out.append("\n## Code context (exact spans)")
            for b in code:
                lang = _LANG.get(PurePosixPath(b.path or "").suffix.lower(), "")
                body = b.text
                if line_numbers and b.start_line is not None:
                    width = len(str(b.end_line or b.start_line))
                    body = "\n".join(f"{n:>{width}} | {line}" for n, line in enumerate(b.text.splitlines(), start=b.start_line))
                out.append(f"### {b.title}\n```{lang}\n{body}\n```")
        if outlines:
            out.append(f"\n## Other relevant code (signatures only: {map_hint})")
            out.extend(f"- {b.path}  L{b.start_line}-{b.end_line}  {b.text}" for b in outlines)
        if self.harmonic_field and (self.harmonic_field.call_slices or self.harmonic_field.resonant_tests):
            out.append("\n## Harmonic resonance (coupled call-sites & verification)")
            for cs in self.harmonic_field.call_slices:
                out.append(f"- [call] {cs.caller_path}:{cs.call_line}  [{cs.caller_symbol} -> {cs.target_symbol}]  `{cs.snippet}`")
            for rt in self.harmonic_field.resonant_tests:
                via = f" (calls {rt.coupled_to})" if rt.coupled_to else ""
                out.append(f"- [test] pytest {rt.pytest_target}  L{rt.start_line}-{rt.end_line}{via}")
        return "\n".join(out)

    def render_locations(self, *, max_tokens: int = 500, chars_per_token: float = 3.2) -> LocationList:
        """Octave-separated locations:
        1. Fundamental band (Omega_0, production symbols) first.
        2. Impedance-matched coupled call-site micro-slices (1-line impact slices).
        3. Upper octave (Omega_1, resonant verification tests) capped so tests never crowd out source code.
        """
        def cost(t: str) -> int:
            return max(1, math.ceil(len(t) / chars_per_token))

        def format_block(b: ContextBlock) -> str | None:
            if b.kind in TEXT_KINDS:
                text = " ".join(b.text.split())
                return f"- [{b.kind.name} #{b.node_ids[0]}] {text[:200]}{'…' if len(text) > 200 else ''}"
            if b.kind == NodeKind.FILE or b.start_line is None or all(s.startswith("<header") for s in b.symbols or ("<header>",)):
                return None
            if not b.full:
                return f"- {b.path}  L{b.start_line}-{b.end_line}  {b.text}"
            items = spectroscopic_outlines_for_span(b.path, b.text, b.start_line) or outline(b.text, b.start_line, b.symbols)
            head = f"- {b.path}  L{b.start_line}-{b.end_line}"
            if len(items) == 1:
                o = items[0]
                return f"{head}  {o.signature}" + (f"  — {o.doc}" if o.doc else "")
            return "\n".join([head] + [f"    L{o.line}  {o.signature}" + (f"  — {o.doc}" if o.doc else "") for o in items])

        entries: list[str] = []
        files: dict[str, None] = {}
        used = 0

        # Partition blocks into Fundamental (Omega_0) and Test Octave (Omega_1)
        prod_blocks = sorted(
            (b for b in self.blocks if not _is_test_path(b.path)),
            key=lambda b: -b.score,
        )
        test_blocks = sorted(
            (b for b in self.blocks if _is_test_path(b.path)),
            key=lambda b: -b.score,
        )

        hf = self.harmonic_field
        has_harmonic = hf is not None and bool(hf.call_slices or hf.resonant_tests)
        # Reserve up to ~22% of budget for harmonic call-slices + resonant test octave when present
        prod_budget = int(max_tokens * 0.78) if (has_harmonic and prod_blocks and max_tokens >= 200) else max_tokens

        overflow_prod: list[ tuple[ContextBlock, str, int] ] = []
        for b in prod_blocks:
            entry = format_block(b)
            if entry is None:
                continue
            c = cost(entry) + 1
            if entries and used + c > prod_budget:
                overflow_prod.append((b, entry, c))
                continue
            if entries and used + c > max_tokens:
                break
            entries.append(entry)
            used += c
            if b.path:
                files[b.path] = None

        # Band 2: Impedance-matched call-site micro-slices (Omega_0 <-> Omega_0)
        if hf and hf.call_slices:
            for cs in hf.call_slices[:2]:
                entry = f"- [call] {cs.caller_path}:{cs.call_line}  [{cs.caller_symbol} -> {cs.target_symbol}]  {cs.snippet}"
                c = cost(entry) + 1
                if entries and used + c > max_tokens:
                    break
                entries.append(entry)
                used += c
                if cs.caller_path:
                    files[cs.caller_path] = None

        # Band 3: Upper Octave (Omega_1) Resonant Tests (max 2 so tests never dominate)
        shown_tests = 0
        if hf and hf.resonant_tests:
            for rt in hf.resonant_tests[:2]:
                via = f" (calls {rt.coupled_to})" if rt.coupled_to else ""
                entry = f"- [test] {rt.path}  L{rt.start_line}-{rt.end_line}  pytest: {rt.pytest_target}{via}"
                c = cost(entry) + 1
                if entries and used + c > max_tokens:
                    break
                entries.append(entry)
                used += c
                files[rt.path] = None
                shown_tests += 1

        # Reinvest any remaining budget into overflow production symbols first, then test blocks
        for b, entry, c in overflow_prod:
            if entries and used + c > max_tokens:
                break
            entries.append(entry)
            used += c
            if b.path:
                files[b.path] = None

        for b in test_blocks:
            if shown_tests >= 2 and prod_blocks:
                break
            entry = format_block(b)
            if entry is None:
                continue
            c = cost(entry) + 1
            if entries and used + c > max_tokens:
                break
            entries.append(entry)
            used += c
            if b.path:
                files[b.path] = None
            shown_tests += 1

        return LocationList("\n".join(entries), len(entries), used, tuple(files))

    def to_chat_messages(self, system_prompt: str | None = None, *, line_numbers: bool = False,
                         contract: str | None = None) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": system_prompt or DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": self.render(line_numbers=line_numbers, contract=contract)},
        ]


def _render_tree(paths: tuple[str, ...]) -> str:
    lines: list[str] = []
    printed: set[tuple[str, ...]] = set()
    for p in sorted(paths):
        parts = PurePosixPath(p).parts
        for depth in range(1, len(parts)):
            prefix = parts[:depth]
            if prefix not in printed:
                printed.add(prefix)
                lines.append("  " * (depth - 1) + parts[depth - 1] + "/")
        lines.append("  " * (len(parts) - 1) + parts[-1])
    return "\n".join(lines)


_TEST_DIRS = frozenset({"test", "tests", "spec", "specs", "__tests__", "testing"})


_STUB_SUFFIXES = (".pyi", ".d.ts", ".d.mts", ".d.cts")


def _is_glue(symbol: str | None) -> bool:
    """Module code between symbols: `<header>` and `<gap:...>` chunks (also nested, `Class.<gap:...>`)."""
    tail = (symbol or "").split("~", 1)[0].rsplit(".", 1)[-1]
    return tail.startswith(("<header", "<gap"))


def _is_stub_path(path: str | None) -> bool:
    return bool(path) and str(path).lower().endswith(_STUB_SUFFIXES)


def _is_test_path(path: str | None) -> bool:
    if not path:
        return False
    p = PurePosixPath(path)
    name = p.name.lower()
    return (bool(_TEST_DIRS & {part.lower() for part in p.parts[:-1]}) or name.startswith("test_")
            or name.endswith(("_test.py", "_test.go", ".test.ts", ".test.js", ".spec.ts", ".spec.js", "test.java")))


@dataclass
class _Scored:
    mode: Mode
    z_q: np.ndarray
    ids: np.ndarray
    records: dict[int, NodeRecord]
    score: np.ndarray
    distance: np.ndarray
    energy: np.ndarray
    relevance: np.ndarray
    geo_rank: np.ndarray
    lex_rank: np.ndarray
    vectors: np.ndarray | None = None  # unit directions of the candidates (rows follow ids)


@dataclass(frozen=True)
class _Anchor:
    node_id: int
    path: str
    line: int


@dataclass
class _Selected:
    rec: NodeRecord
    text: str
    tokens: int
    score: float
    distance: float
    energy: float
    relevance: float
    start_line: int | None
    end_line: int | None
    geo_rank: int | None
    lex_rank: int | None
    full: bool = True
    pauli_turn: int | None = None


class GeometricRouter:
    def __init__(
        self,
        manager: NewtonianMemoryManager,
        config: RouterConfig | None = None,
        *,
        token_counter: Callable[[str], int] | None = None,
    ) -> None:
        self._m = manager
        self.config = config or RouterConfig()
        cpt = self.config.chars_per_token
        self._count = token_counter or (lambda s: max(1, math.ceil(len(s) / cpt)))

    # -- public -------------------------------------------------------------------------------
    def route_to_compute_engine(
        self,
        query: str,
        *,
        mode: Literal["auto", "brownfield", "greenfield"] = "auto",
        token_budget: int | None = None,
        pauli_session: PauliSessionCache | None = None,
    ) -> RoutedContext:
        if not query.strip():
            raise ValueError("empty query")
        cfg = self.config
        m = self._m
        budget = token_budget or cfg.token_budget
        with m.lock:
            anchors = self._anchors(query)
            sc = self._score(query, mode, [a.node_id for a in anchors])
            if sc.ids.size == 0:
                return RoutedContext(query, sc.mode, (), self._file_map(sc.mode, []), 0, budget)
            remaining = budget - cfg.template_overhead_tokens - self._count(query)
            selected: list[_Selected] = []
            resynced: set[str] = set()
            order = np.argsort(-sc.score, kind="stable")
            top = float(sc.score[order[0]])
            index = {int(i): j for j, i in enumerate(sc.ids)}
            pinned: list[str] = []
            for rank, a in enumerate(anchors):
                if remaining < cfg.min_block_tokens:
                    break
                if a.node_id not in index:
                    continue
                picked = self._take_anchor(sc, index[a.node_id], a.line, resynced, remaining)
                if picked is not None:
                    picked.score = max(top, 1e-9) * (2.0 - 0.01 * rank)  # rendered ahead of ranked code, in frame order
                    selected.append(picked)
                    remaining -= picked.tokens
                    pinned.append(f"{a.path}:{a.line}")
            taken = {s.rec.id for s in selected}
            if (cfg.dedup or cfg.diversity_lambda > 0 or cfg.focus_blocks > 0 or cfg.focus_mass > 0 or cfg.signature_extra > 0
                    or (cfg.traceback_focus_blocks > 0 and pinned)):
                blocks = self._route_focused(sc, order, top, selected, taken, resynced, remaining, budget, query, pauli_session=pauli_session)
                if cfg.reinforce and selected:
                    m.reinforce([s.rec.id for s in selected], sc.z_q, [s.relevance for s in selected])
                if pauli_session is not None:
                    self._record_pauli_blocks(blocks, pauli_session)
                h_field = self._build_harmonic_field(sc, order)
                total = sum(b.tokens for b in blocks) + cfg.template_overhead_tokens + self._count(query)
                return RoutedContext(query, sc.mode, tuple(blocks), self._file_map(sc.mode, blocks), total, budget, tuple(pinned),
                                     tuple((s.rec.id, s.score) for s in selected), h_field)
            ranked = 0
            for j in order:
                if remaining < cfg.min_block_tokens:
                    break
                if float(sc.score[j]) <= 0.0:
                    break  # candidates no ranked list supports (e.g. geometric seeds when the geometric weight is 0)
                # Dynamic cut: past the guaranteed minimum, stop when the score collapses relative to the best
                # candidate instead of filling the budget with weakly related code (wasted KV cache, distraction).
                if ranked >= cfg.min_blocks and float(sc.score[j]) < cfg.cutoff_alpha * top:
                    break
                if int(sc.ids[j]) in taken:
                    continue
                picked = self._take(sc, int(j), resynced, remaining, allow_truncate=not selected)
                if picked is not None:
                    selected.append(picked)
                    remaining -= picked.tokens
                    ranked += 1
            remaining = self._add_structural_headers(sc, selected, resynced, remaining)

            blocks = self._assemble(selected)
            if pauli_session is not None:
                self._record_pauli_blocks(blocks, pauli_session)
            if cfg.reinforce and selected:
                m.reinforce([s.rec.id for s in selected], sc.z_q, [s.relevance for s in selected])
            h_field = self._build_harmonic_field(sc, order)
            total = sum(b.tokens for b in blocks) + cfg.template_overhead_tokens + self._count(query)
            return RoutedContext(query, sc.mode, tuple(blocks), self._file_map(sc.mode, blocks), total, budget, tuple(pinned),
                                 tuple((s.rec.id, s.score) for s in selected), h_field)

    def _record_pauli_blocks(self, blocks: Sequence[ContextBlock], pauli_session: PauliSessionCache) -> None:
        full_chunks = [b for b in blocks if b.kind == NodeKind.CHUNK and b.full and b.path and b.start_line and b.end_line]
        if not full_chunks:
            return
        t = pauli_session._bump_turn()
        for b in full_chunks:
            pauli_session.record_span(b.path, b.start_line, b.end_line, b.text, turn=t)

    def _build_harmonic_field(self, sc: "_Scored", order: np.ndarray) -> HarmonicField:
        fundamental_recs: list[tuple[NodeRecord, float]] = []
        test_recs: list[tuple[NodeRecord, float]] = []
        for j in order:
            s = float(sc.score[int(j)])
            if s <= 0.0:
                break
            rec = sc.records[int(sc.ids[int(j)])]
            if rec.kind != NodeKind.CHUNK or not rec.path:
                continue
            if _is_test_path(rec.path):
                if len(test_recs) < 6:
                    test_recs.append((rec, s))
            else:
                if len(fundamental_recs) < 6:
                    fundamental_recs.append((rec, s))
        return compute_harmonic_resonance(
            self._m,
            sc.z_q,
            fundamental_recs,
            candidate_test_records=test_recs,
        )

    def ranked_files(self, query: str, *, mode: Literal["auto", "brownfield", "greenfield"] = "auto") -> list[tuple[str, float]]:
        """Retrieval ranking only (no packing, no reinforcement): files by their best candidate score.
        Used by evaluations; does not read code from disk."""
        with self._m.lock:
            anchors = self._anchors(query)
            sc = self._score(query, mode, [a.node_id for a in anchors])
            best: dict[str, float] = {a.path: math.inf for a in anchors}
            # Octave-aware file ranking: fundamental production files first when scores are within harmonic band
            prod_order = [j for j in np.argsort(-sc.score, kind="stable") if not _is_test_path(sc.records[int(sc.ids[j])].path)]
            test_order = [j for j in np.argsort(-sc.score, kind="stable") if _is_test_path(sc.records[int(sc.ids[j])].path)]
            for j in [*prod_order, *test_order]:
                rec = sc.records[int(sc.ids[j])]
                if rec.path and rec.kind in (NodeKind.CHUNK, NodeKind.FILE) and rec.path not in best:
                    best[rec.path] = float(sc.score[j])
            return list(best.items())

    def _anchors(self, query: str) -> list["_Anchor"]:
        """Symbols at the file:line locations the query's error output names: source before tests, innermost first."""
        limit = self.config.trace_anchors
        locations = parse_locations(query) if limit > 0 else []
        if not locations:
            return []
        resolver = PathResolver(self._m.file_paths())
        source: list[_Anchor] = []
        tests: list[_Anchor] = []
        seen: set[int] = set()
        for loc in locations:
            rel = resolver.resolve(loc.path)
            if rel is None:
                continue
            covering = [c for c in self._m.meta.chunks_of_file(rel) if c.start_line and c.end_line and c.start_line <= loc.line <= c.end_line]
            if not covering:
                continue
            rec = min(covering, key=lambda c: (c.end_line - c.start_line, c.start_line))  # innermost symbol
            if rec.id in seen:
                continue
            seen.add(rec.id)
            (tests if _is_test_path(rel) else source).append(_Anchor(rec.id, rel, loc.line))
        return [*source, *tests][:limit]

    def _take_anchor(self, sc: "_Scored", j: int, line: int, resynced: set[str], remaining: int) -> "_Selected | None":
        """A pinned symbol, cut to a window around the named line when it is long or does not fit."""
        picked = self._take(sc, j, resynced, remaining, allow_truncate=False)
        window = self.config.trace_window_lines
        if picked is not None and (picked.start_line is None or picked.end_line is None
                                   or picked.end_line - picked.start_line < window):
            return picked
        rec = sc.records[int(sc.ids[j])]
        if not rec.path or rec.start_line is None or rec.end_line is None:
            return None
        start = max(rec.start_line, line - window // 2)
        end = min(rec.end_line, start + window - 1)
        text = self._m.read_lines(rec.path, start, end)
        if text is None:
            return None
        tokens = self._count(text)
        if tokens > remaining:
            text, end = self._truncate(text, start, remaining)
            tokens = self._count(text)
        return _Selected(
            rec, text, tokens, float(sc.score[j]), float(sc.distance[j]), float(sc.energy[j]), float(sc.relevance[j]),
            start, end, int(sc.geo_rank[j]), int(sc.lex_rank[j]) if sc.lex_rank[j] >= 0 else None,
        )

    def _score(self, query: str, mode: str, extra: Sequence[int] = ()) -> "_Scored":
        cfg = self.config
        m = self._m
        resolved = self._resolve_mode(mode)
        weights = cfg.kind_weights(resolved)
        z_q = m.embed_query(query)
        lex_query = build_query(query)
        lex_hits = m.lexical_search(lex_query, cfg.lexical_k, SEED_KINDS | {NodeKind.FILE}) if cfg.lexical_weight > 0 else []
        cand = self._collect_candidates(z_q, lex_hits)
        if extra:
            cand = list(dict.fromkeys([*cand, *extra]))
        if cfg.recency_weight > 0 and cfg.recency_seeds > 0:
            cand = list(dict.fromkeys([*cand, *(i for i, _ in m.hottest(cfg.recency_seeds, frozenset({NodeKind.CHUNK})))]))
        records = m.records(cand)
        ids = np.array([i for i in cand if i in records and weights.get(records[i].kind, 0.0) > 0 and not (records[i].path and is_excluded(records[i].path, m.config))], dtype=np.int64)
        empty = np.zeros(0)
        if ids.size == 0:
            return _Scored(resolved, z_q, ids, records, empty, empty, empty, empty, np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64))
        x = torch.from_numpy(m.positions(ids)).to(torch.float64)
        d = H.distance(torch.from_numpy(z_q).to(torch.float64)[None, :], x, m.config.curvature).numpy()
        geo_rank = np.empty(ids.size, dtype=np.int64)
        geo_rank[np.argsort(d, kind="stable")] = np.arange(ids.size)
        lex_of = {h.node_id: h.rank for h in lex_hits}
        lex_rank = np.array([lex_of.get(int(i), -1) for i in ids], dtype=np.int64)
        w_lex = cfg.lexical_weight * (cfg.code_query_lexical_boost if lex_query.code_like else 1.0)
        fused = cfg.geometric_weight / (cfg.rrf_k + 1.0 + geo_rank)
        hit = lex_rank >= 0
        fused[hit] += w_lex / (cfg.rrf_k + 1.0 + lex_rank[hit])
        energy = m.energies(ids)
        if cfg.recency_weight > 0:
            rec_rank = np.empty(ids.size, dtype=np.int64)
            rec_rank[np.argsort(-energy, kind="stable")] = np.arange(ids.size)
            fused = fused + cfg.recency_weight / (cfg.rrf_k + 1.0 + rec_rank)
        rel = fused / fused.max() if fused.max() > 0 else fused
        log_e = np.log1p(np.maximum(energy, 0.0))
        e_norm = log_e / log_e.max() if log_e.max() > 0 else np.zeros_like(log_e)
        kw = np.array([
            weights.get(records[int(i)].kind, 0.0)
            * (cfg.nonsymbol_chunk_weight if records[int(i)].kind == NodeKind.CHUNK and (records[int(i)].symbol or "<").startswith("<") and not is_spectroscopic_path(records[int(i)].path) else 1.0)
            * (cfg.test_path_weight if _is_test_path(records[int(i)].path) else 1.0)
            for i in ids
        ])
        score = kw * rel * (1.0 + cfg.energy_weight * e_norm)
        pos = x.numpy()
        unit = pos / np.maximum(np.linalg.norm(pos, axis=1, keepdims=True), 1e-12)
        return _Scored(resolved, z_q, ids, records, score, d, energy, rel, geo_rank, lex_rank, unit)

    def _take(self, sc: "_Scored", j: int, resynced: set[str], remaining: int, *, allow_truncate: bool) -> "_Selected | None":
        mat = self._materialize(sc.records[int(sc.ids[j])], resynced)
        if mat is None:
            return None
        rec, text, start, end = mat
        tokens = self._count(text)
        if tokens > remaining:
            if not allow_truncate or rec.kind != NodeKind.CHUNK:
                return None
            text, end = self._truncate(text, start, remaining)
            tokens = self._count(text)
        return _Selected(
            rec, text, tokens, float(sc.score[j]), float(sc.distance[j]), float(sc.energy[j]), float(sc.relevance[j]),
            start, end, int(sc.geo_rank[j]), int(sc.lex_rank[j]) if sc.lex_rank[j] >= 0 else None,
        )

    def _take_harmonic_signature(self, sc: "_Scored", j: int, resynced: set[str], remaining: int) -> "_Selected | None":
        """When a fundamental symbol's full body exceeds `remaining` (e.g. under a tight 500-token `locations` budget),
        compress it harmonically to its signature line (`full=False`) instead of dropping the symbol."""
        rec0 = sc.records[int(sc.ids[j])]
        if rec0.kind != NodeKind.CHUNK:
            return None
        mat = self._materialize(rec0, resynced)
        if mat is None:
            return None
        rec, text, start, end = mat
        sig = self._signature_block(rec, text, start, end, float(sc.score[j]), float(sc.distance[j]), float(sc.energy[j]))
        if sig is None or sig.tokens > remaining:
            return None
        return _Selected(
            rec, text, sig.tokens, float(sc.score[j]), float(sc.distance[j]), float(sc.energy[j]), float(sc.relevance[j]),
            start, end, int(sc.geo_rank[j]), int(sc.lex_rank[j]) if sc.lex_rank[j] >= 0 else None,
            full=False,
        )

    def _add_structural_headers(self, sc: "_Scored", selected: list["_Selected"], resynced: set[str], remaining: int) -> int:
        """Guaranteed structural context: the header (imports/constants) of the top files that contributed code,
        even when the header itself scored below the cut."""
        cfg = self.config
        if cfg.structural_headers <= 0:
            return remaining
        have = {s.rec.id for s in selected}
        files: list[int] = []
        for s in selected:
            if s.rec.kind == NodeKind.CHUNK and s.rec.parent_id is not None and s.rec.parent_id not in files:
                files.append(s.rec.parent_id)
        index = {int(i): j for j, i in enumerate(sc.ids)}
        for fid in files[: cfg.structural_headers]:
            if fid in have or fid not in index or remaining < cfg.min_block_tokens:
                continue
            picked = self._take(sc, index[fid], resynced, remaining, allow_truncate=False)
            if picked is not None:
                selected.append(picked)
                remaining -= picked.tokens
        return remaining

    # -- focused context ----------------------------------------------------------------------
    def _route_focused(self, sc: "_Scored", order: np.ndarray, top: float, selected: list["_Selected"], taken: set[int],
                       resynced: set[str], remaining: int, budget: int, query: str,
                       pauli_session: PauliSessionCache | None = None) -> list[ContextBlock]:
        """Anti-clone order (stub filter, near-duplicates, MMR), then focus: bodies only where they matter."""
        base = self.config
        if taken and base.traceback_focus_blocks > 0:  # pinned traceback symbols: fixed focus, no extended map
            self.config = replace(base, focus_blocks=base.traceback_focus_blocks, focus_mass=0.0, focus_flat_blocks=0,
                                  signature_extra=0)
        try:
            return self._route_focused_inner(sc, order, top, selected, taken, resynced, remaining, budget, query, pauli_session=pauli_session)
        finally:
            self.config = base

    def _route_focused_inner(self, sc: "_Scored", order: np.ndarray, top: float, selected: list["_Selected"], taken: set[int],
                             resynced: set[str], remaining: int, budget: int, query: str,
                             pauli_session: PauliSessionCache | None = None) -> list[ContextBlock]:
        cfg = self.config
        pinned_ids = set(taken)
        positive = [int(j) for j in order if float(sc.score[j]) > 0.0]
        # Octave-aware cutoff: anchor the fundamental threshold to the best non-test symbol so upper-octave tests
        # cannot suppress production symbols via relative cutoff.
        prod_top = max(
            (float(sc.score[j]) for j in positive if not _is_test_path(sc.records[int(sc.ids[j])].path)),
            default=top,
        )
        pool: list[int] = []
        prod_seen = 0
        for r, j in enumerate(positive):
            rec = sc.records[int(sc.ids[j])]
            is_test = _is_test_path(rec.path)
            ref_top = top if is_test else prod_top
            if r < cfg.min_blocks or (not is_test and prod_seen < cfg.min_blocks) or float(sc.score[j]) >= cfg.cutoff_alpha * ref_top:
                pool.append(j)
                if not is_test:
                    prod_seen += 1
        stub_only: set[int] = set()
        implemented: set[str] = set()
        if cfg.dedup:
            pool, stub_only, implemented = self._filter_stubs(sc, pool, positive)
        mmr = self._mmr_order(sc, pool, top, chosen=[j for j in range(sc.ids.size) if int(sc.ids[j]) in pinned_ids])
        # Octave separation during packing: pack fundamental production symbols (Omega_0) before test octave (Omega_1)
        ordered = (
            [j for j in mmr if not _is_test_path(sc.records[int(sc.ids[j])].path)]
            + [j for j in mmr if _is_test_path(sc.records[int(sc.ids[j])].path)]
        )
        ranked_sel: list[_Selected] = []
        for j in ordered:
            if remaining < cfg.min_block_tokens // 2:
                break
            if int(sc.ids[j]) in taken:
                continue
            picked = self._take(sc, j, resynced, remaining, allow_truncate=not selected)
            if picked is None:
                # Harmonic fallback: if a symbol's full body exceeds `remaining`, keep its 1-line signature
                picked = self._take_harmonic_signature(sc, j, resynced, remaining)
            if (picked is not None and picked.full and pauli_session is not None
                    and picked.rec.kind == NodeKind.CHUNK and picked.rec.path and picked.start_line and picked.end_line):
                cov = pauli_session.find_covering_unchanged(picked.rec.path, picked.start_line, picked.end_line, picked.text)
                if cov is not None:
                    sig = self._signature_block(picked.rec, picked.text, picked.start_line, picked.end_line,
                                                picked.score, picked.distance, picked.energy, pauli_turn=cov.turn)
                    if sig is not None and sig.tokens < picked.tokens:
                        picked.full = False
                        picked.pauli_turn = cov.turn
                        picked.tokens = sig.tokens
            if picked is not None:
                taken.add(picked.rec.id)
                selected.append(picked)
                ranked_sel.append(picked)
                remaining -= picked.tokens
        self._add_structural_headers(sc, selected, resynced, remaining)
        if cfg.focus_blocks > 0 or cfg.focus_mass > 0:
            self._mark_focus(selected, ranked_sel, pinned_ids, stub_only, prod_top)
        elif stub_only:
            for s in selected:
                if s.rec.id in stub_only:
                    s.full = False
        blocks = self._assemble(selected)
        if cfg.signature_extra > 0:
            used = sum(b.tokens for b in blocks) + cfg.template_overhead_tokens + self._count(query)
            blocks += self._extra_signatures(sc, ordered, positive, taken, budget - used, blocks, implemented)
        return blocks

    def _filter_stubs(self, sc: "_Scored", pool: list[int], positive: list[int]) -> tuple[list[int], set[int], set[str]]:
        """A stub never stands in for its implementation: dropped when the implementation is a candidate."""
        implemented: set[str] = set()
        for j in positive:
            rec = sc.records[int(sc.ids[j])]
            if rec.kind == NodeKind.CHUNK and not _is_stub_path(rec.path):
                name = short_name(rec.symbol or "")
                if name:
                    implemented.add(name)
        kept, stub_only = [], set()
        for j in pool:
            rec = sc.records[int(sc.ids[j])]
            if rec.kind == NodeKind.CHUNK and _is_stub_path(rec.path):
                if short_name(rec.symbol or "") in implemented:
                    continue
                stub_only.add(rec.id)
            kept.append(j)
        return kept, stub_only, implemented

    def _mmr_order(self, sc: "_Scored", pool: list[int], top: float, *, chosen: list[int]) -> list[int]:
        """Greedy MMR among chunks: score/best - lambda * max cosine with chunks already chosen; chunks above
        dup_threshold are dropped (with dedup). Files and notes keep their score and are never compared."""
        cfg = self.config
        lam = cfg.diversity_lambda
        if (lam <= 0 and not cfg.dedup) or sc.vectors is None:
            return list(pool)
        is_chunk = {j: sc.records[int(sc.ids[j])].kind == NodeKind.CHUNK for j in pool}
        chosen_vecs = [sc.vectors[j] for j in chosen if sc.records[int(sc.ids[j])].kind == NodeKind.CHUNK]
        remaining = list(pool)
        out: list[int] = []
        best = max(top, 1e-12)
        while remaining:
            if chosen_vecs:
                sims = np.stack(chosen_vecs) @ sc.vectors[remaining].T  # (chosen, remaining)
                max_sim = sims.max(axis=0)
            else:
                max_sim = np.zeros(len(remaining))
            values = []
            for k, j in enumerate(remaining):
                penalty = lam * float(max_sim[k]) if is_chunk[j] else 0.0
                values.append(float(sc.score[j]) / best - penalty)
            k = int(np.argmax(values))
            j = remaining.pop(k)
            if is_chunk[j] and cfg.dedup and float(max_sim[k]) > cfg.dup_threshold:
                continue  # a clone of a chunk already in: never sent
            out.append(j)
            if is_chunk[j]:
                chosen_vecs.append(sc.vectors[j])
        return out

    def _mark_focus(self, selected: list["_Selected"], ranked: list["_Selected"], pinned: set[int], stub_only: set[int],
                    top: float) -> None:
        """Bodies for pinned chunks and for the production chunks (not tests, not stubs) that win a focus slot."""
        cfg = self.config

        def production(s: "_Selected") -> bool:
            return (s.rec.kind == NodeKind.CHUNK and s.rec.id not in pinned and s.rec.id not in stub_only
                    and not _is_stub_path(s.rec.path) and not _is_test_path(s.rec.path)
                    and not (cfg.focus_skip_glue and _is_glue(s.rec.symbol)))

        prod = [s for s in ranked if production(s) and s.full]
        width = cfg.focus_blocks
        if cfg.focus_flat_blocks > 0 and len(prod) >= 5:
            top5 = sorted((p.score for p in prod), reverse=True)[:5]
            if top5[0] > 0 and (top5[0] - top5[4]) / top5[0] < cfg.focus_flat_drop:
                width = max(width, cfg.focus_flat_blocks)
        focus = {p.rec.id for p in prod[:width]}
        if cfg.focus_mass > 0:
            total, cum = sum(p.score for p in prod), 0.0
            for p in prod:
                if cum < cfg.focus_mass * total:
                    focus.add(p.rec.id)
                cum += p.score
        full_paths: set[str] = set()
        for s in selected:
            if s.rec.kind != NodeKind.CHUNK:
                continue
            if not s.full:
                # Already compressed to harmonic signature during budget packing
                continue
            if s.rec.id in pinned:
                s.full = True
            elif not production(s):
                s.full = False
            else:
                s.full = s.rec.id in focus or s.score >= cfg.focus_ratio * top
            if s.full and s.rec.path:
                full_paths.add(s.rec.path)
        for s in selected:  # file headers and module glue only for files whose code is shown
            if s.rec.kind == NodeKind.FILE:
                s.full = s.rec.path in full_paths
            elif (cfg.focus_skip_glue and s.rec.kind == NodeKind.CHUNK and _is_glue(s.rec.symbol) and s.rec.id not in pinned
                  and not _is_test_path(s.rec.path) and not _is_stub_path(s.rec.path)):
                s.full = s.rec.path in full_paths

    def _extra_signatures(self, sc: "_Scored", ordered: list[int], positive: list[int], taken: set[int], room: int,
                          blocks: list[ContextBlock], implemented: set[str]) -> list[ContextBlock]:
        """Reinvest the saved tokens: more candidates, as signature lines only (no stub of a shown implementation,
        no clone of a chunk already in)."""
        cfg = self.config
        shown = {(b.path, b.start_line, b.end_line) for b in blocks}
        index = {int(i): j for j, i in enumerate(sc.ids)}
        in_vecs = [sc.vectors[index[i]] for i in taken if i in index and sc.records[i].kind == NodeKind.CHUNK] \
            if sc.vectors is not None else []
        extra: list[ContextBlock] = []
        seen = set(taken)
        for j in [*ordered, *positive]:
            if len(extra) >= cfg.signature_extra or room < cfg.min_block_tokens // 2:
                break
            rec = sc.records[int(sc.ids[j])]
            if rec.id in seen or rec.kind != NodeKind.CHUNK or not rec.path:
                continue
            seen.add(rec.id)
            if _is_stub_path(rec.path) and short_name(rec.symbol or "") in implemented:
                continue
            if cfg.dedup and in_vecs and float(np.max(np.stack(in_vecs) @ sc.vectors[j])) > cfg.dup_threshold:
                continue
            mat = self._materialize(rec, set())
            if mat is None:
                continue
            cur, text, start, end = mat
            block = self._signature_block(cur, text, start, end, float(sc.score[j]), float(sc.distance[j]), float(sc.energy[j]))
            if block is None or (block.path, block.start_line, block.end_line) in shown or block.tokens > room:
                continue
            extra.append(block)
            room -= block.tokens
            if sc.vectors is not None:
                in_vecs.append(sc.vectors[j])
        return extra

    def _signature_block(self, rec: NodeRecord, text: str, start: int | None, end: int | None, score: float,
                         distance: float, energy: float, *, pauli_turn: int | None = None) -> ContextBlock | None:
        if start is None:
            return None
        lines = spectroscopic_outlines_for_span(rec.path, text, start) or outline(text, start, [rec.symbol or "?"])
        if not lines:
            return None
        o = lines[0]
        prefix = f"[PAULI_IN_CONTEXT @ turn #{pauli_turn}] " if pauli_turn is not None else ""
        line = prefix + o.signature + (f"  — {o.doc}" if o.doc else "")
        return ContextBlock(NodeKind.CHUNK, (rec.id,), f"{rec.path}  L{start}-{end}  [{rec.symbol}]", line, score, distance,
                            energy, self._count(f"- {rec.path}  L{start}-{end}  {line}"), rec.path, start, end,
                            symbols=(rec.symbol or "?",), full=False)

    # -- internals ----------------------------------------------------------------------------
    def _resolve_mode(self, mode: str) -> Mode:
        if mode in ("brownfield", "greenfield"):
            return mode  # type: ignore[return-value]
        stored = self._m.project_mode()
        if stored in ("brownfield", "greenfield"):
            return stored  # type: ignore[return-value]
        return "greenfield" if self._m.count_by_kind()[NodeKind.CHUNK] == 0 else "brownfield"

    def _collect_candidates(self, z_q: np.ndarray, lex_hits: list[LexicalHit] | None = None) -> list[int]:
        cfg = self.config
        m = self._m
        seed_ids, _ = m.search(z_q[None, :], cfg.seed_k, SEED_KINDS)
        seeds = [int(i) for i in seed_ids[0] if i >= 0]
        lexical = [h.node_id for h in (lex_hits or [])]
        cand: dict[int, None] = dict.fromkeys(seeds)
        cand.update(dict.fromkeys(lexical))
        top = list(dict.fromkeys(seeds[: cfg.expansion_seeds] + lexical[: cfg.lexical_seeds]))
        for e in m.neighbors(top):
            cand[e.src] = None
            cand[e.dst] = None
        if cfg.include_file_headers:
            for rec in m.records(top).values():
                if rec.kind == NodeKind.CHUNK and rec.parent_id is not None:
                    cand[rec.parent_id] = None
        return list(cand)

    def _materialize(self, rec: NodeRecord, resynced: set[str]) -> tuple[NodeRecord, str, int | None, int | None] | None:
        m = self._m
        cur = m.record(rec.id)
        if cur is None or cur.key != rec.key:
            return None
        if cur.kind in TEXT_KINDS:
            return (cur, cur.payload, None, None) if cur.payload else None
        if cur.kind == NodeKind.FILE and cur.path:
            header = m.read_file_header(cur.path, self.config.header_lines)
            return (cur, header[0], 1, header[1]) if header else None
        if cur.kind == NodeKind.CHUNK and cur.path:
            text = m.read_chunk(cur)
            if text is None and cur.path not in resynced:
                resynced.add(cur.path)
                m.ingest_files([cur.path])  # file drifted since ingestion: resync, then re-resolve by key
                cur = m.record_by_key(rec.key)
                text = m.read_chunk(cur) if cur is not None else None
            if text is None or cur is None:
                return None
            return cur, text, cur.start_line, cur.end_line
        return None

    def _truncate(self, text: str, start: int | None, max_tokens: int) -> tuple[str, int | None]:
        kept: list[str] = []
        used = self._count("# ... truncated")
        for line in text.splitlines():
            cost = self._count(line + "\n")
            if used + cost > max_tokens:
                break
            kept.append(line)
            used += cost
        kept.append("# ... truncated")
        end = None if start is None else start + len(kept) - 2
        return "\n".join(kept), end

    def _assemble(self, selected: list[_Selected]) -> list[ContextBlock]:
        signatures: list[tuple[_Selected, ContextBlock]] = []
        for s in selected:
            if not s.full and s.rec.kind == NodeKind.CHUNK:
                sig = self._signature_block(s.rec, s.text, s.start_line, s.end_line, s.score, s.distance, s.energy,
                                            pauli_turn=s.pauli_turn)
                if sig is None or (sig.tokens >= s.tokens and not _is_stub_path(s.rec.path) and s.pauli_turn is None):
                    s.full = True  # a signature that costs as much as the body saves nothing: keep the body
                else:
                    signatures.append((s, sig))
        selected = [s for s in selected if s.full]
        texts = sorted((s for s in selected if s.rec.kind in TEXT_KINDS), key=lambda s: -s.score)
        blocks = [
            ContextBlock(s.rec.kind, (s.rec.id,), s.rec.kind.name, s.text, s.score, s.distance, s.energy, s.tokens,
                         geo_rank=s.geo_rank, lex_rank=s.lex_rank)
            for s in texts
        ]
        by_path: dict[str, list[_Selected]] = {}
        for s in selected:
            if s.rec.kind in (NodeKind.CHUNK, NodeKind.FILE) and s.rec.path:
                by_path.setdefault(s.rec.path, []).append(s)
        file_order = sorted(by_path, key=lambda p: -max(s.score for s in by_path[p]))
        for path in file_order:
            items = by_path[path]
            chunks = sorted((s for s in items if s.rec.kind == NodeKind.CHUNK), key=lambda s: s.start_line or 0)
            header = next((s for s in items if s.rec.kind == NodeKind.FILE), None)
            if header is not None and not any((c.start_line or 0) <= (header.end_line or 0) for c in chunks):
                blocks.append(ContextBlock(NodeKind.FILE, (header.rec.id,), f"{path}  L1-{header.end_line}  [header]",
                                           header.text, header.score, header.distance, header.energy, header.tokens,
                                           path, 1, header.end_line, header.geo_rank, header.lex_rank))
            for group in _merge_overlapping(chunks):
                if len(group) == 1:
                    s = group[0]
                    text, start, end = s.text, s.start_line, s.end_line
                else:
                    start = min(g.start_line or 0 for g in group)
                    end = max(g.end_line or 0 for g in group)
                    text = self._m.read_lines(path, start, end) or "\n".join(g.text for g in group)
                names = tuple(g.rec.symbol or "?" for g in group)
                symbols = ", ".join(names)
                blocks.append(ContextBlock(
                    NodeKind.CHUNK, tuple(g.rec.id for g in group), f"{path}  L{start}-{end}  [{symbols}]", text,
                    max(g.score for g in group), min(g.distance for g in group), max(g.energy for g in group),
                    self._count(text), path, start, end,
                    min(g.geo_rank for g in group if g.geo_rank is not None) if any(g.geo_rank is not None for g in group) else None,
                    min((g.lex_rank for g in group if g.lex_rank is not None), default=None),
                    names,
                ))
        full_ranges = [(b.path, b.start_line or 0, b.end_line or 0) for b in blocks if b.kind == NodeKind.CHUNK]
        for s, sig in sorted(signatures, key=lambda t: -t[0].score):
            if not any(p == s.rec.path and a <= (s.start_line or 0) and (s.end_line or 0) <= b for p, a, b in full_ranges):
                blocks.append(sig)
        return blocks

    def _file_map(self, mode: Mode, blocks: list[ContextBlock] | tuple[ContextBlock, ...]) -> tuple[str, ...]:
        limit = self.config.file_map_limit
        if mode == "greenfield":
            return tuple(self._m.file_paths()[:limit])
        return tuple(sorted({b.path for b in blocks if b.path})[:limit])


def _merge_overlapping(chunks: list[_Selected]) -> list[list[_Selected]]:
    groups: list[list[_Selected]] = []
    for c in chunks:
        if groups and c.start_line is not None and (groups[-1][-1].end_line or 0) >= c.start_line:
            groups[-1].append(c)
        else:
            groups.append([c])
    return groups


def route_to_compute_engine(
    query: str,
    manager: NewtonianMemoryManager,
    *,
    config: RouterConfig | None = None,
    mode: Literal["auto", "brownfield", "greenfield"] = "auto",
    token_budget: int | None = None,
    token_counter: Callable[[str], int] | None = None,
) -> RoutedContext:
    """Functional entry point: query -> hybrid routing over the SSD memory -> formatted payload for the VRAM model."""
    router = GeometricRouter(manager, config, token_counter=token_counter)
    return router.route_to_compute_engine(query, mode=mode, token_budget=token_budget)
