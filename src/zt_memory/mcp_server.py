"""zt-memory as a Model Context Protocol server (stdio transport, JSON-RPC 2.0, one message per line).

Any MCP client (Claude Code, Claude Desktop, Cursor, Cline, local agents) gets these tools:

    zt_query_context(query, mode?, token_budget?, line_numbers?)
                                                          where the relevant code is (mode locations), the exact spans
                                                          (code), or a few whole bodies plus a map of signatures (focus)
    zt_read_span(spans, line_numbers?)                    up to 3 line ranges read from disk: the NEED round over MCP,
                                                          for clients without a file-reading tool (`zt mcp --read-span`)
    zt_record_decision(summary, paths?)                    architectural decision stored as a DECISION node,
                                                          linked to the files/symbols it is about
    zt_status(limit?)                                      hottest symbols (where work is happening), pending
                                                          changes, store size

Design:
- Starts in any folder. The project (nearest `.zt/config.toml` above the session folder) is looked up on each tool
  call until found, so a project initialised during a session works without restarting the client; without one,
  tools answer with the activation steps instead of failing at launch.
- The encoder (the expensive part: seconds to load, 1-2 GB of RAM) is loaded on the first tool call and released
  after `idle_unload_s` without calls, so idle sessions and sessions on projects without zt cost no RAM.
- A tool call refreshes the index only when few files changed (`max_inline_sync_files`); a first full index or a
  large change is left to `zt sync`, which can take many minutes and would outlive the client's tool timeout.
- The store is opened per tool call and closed afterwards: its exclusive lock is never held while idle, so git hooks,
  `zt watch` and `zt code` keep working next to the server.
- zt_read_span needs neither the encoder nor the store: it only reads files, under the same rule as the index
  (spans.py: never outside the project, never a file in exclude_globs).
- stdout carries protocol messages only: anything else printed by libraries is redirected to stderr.
"""
from __future__ import annotations

import copy
import io
import json
import sys
import threading
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, BinaryIO, Callable, Iterator

from .config import NodeKind
from .encoders import HiddenStateEncoder
from .project import CONFIG_DIR, ROUTER_PRESETS, ProjectConfig, ProjectConfigError
from .spans import MAX_SPAN_LINES, MAX_TOTAL_LINES, SpanPolicy, SpanRefused, parse_span, parse_span_ex
from .storage import StoreLockedError

SERVER_NAME = "zt-memory"
SERVER_VERSION = "0.4.0"
# Default context size for MCP clients. The project's [router] token_budget (12000) is sized for the local 14B, which
# sees the context once; an agent like Claude Code re-reads it on every later turn. A/B on 20 real bugs: 12000 tokens
# cost +37% per bug against no zt, 4000 about the same (examples/ab_claude_code.py). Clients can ask for more.
MCP_DEFAULT_BUDGET = 4000
# `mode="locations"`: the router still selects with the project budget, but only file, lines, signatures and the
# first docstring line come back, cut to this many tokens. For agents that open files themselves. It is the default
# of `zt mcp`: on 25 bugs described in words, Claude Code with it cost what Claude Code alone costs (-3% in total,
# median per-bug ratio 0.98), against +16% (median) with 4000 tokens of code.
LOCATIONS_DEFAULT_TOKENS = 500
# `mode="focus"`: the `api` router preset (anti-clone, 5 whole bodies plus those >= 0.6 x the best, up to 20 more
# signatures as a map; with a traceback in the query 10 bodies and no map), selected with the project budget. Bodies
# listed only as signatures are read with zt_read_span. In-session test on 12 commit tasks: -49% input tokens against
# the plain code context, the right span asked for 6 times out of 6 (docs/NEED.md).
QUERY_MODES = ("code", "locations", "focus")
MAX_READ_SPANS = 3
# Newest first. A client asking for one of these gets it back; otherwise we answer with the newest and the client
# decides whether to continue (MCP lifecycle rules).
SUPPORTED_PROTOCOL_VERSIONS = ("2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05")

PARSE_ERROR, INVALID_REQUEST, METHOD_NOT_FOUND, INVALID_PARAMS, INTERNAL_ERROR = -32700, -32600, -32601, -32602, -32603

INSTRUCTIONS = (
    "zt-memory indexes this repository by symbol (functions, classes, methods) and keeps a usage 'energy' for each. "
    "Call zt_query_context before searching or reading files: it returns where the relevant code is (file, line range, "
    "signature; mode=locations), the exact spans (mode=code) or a few whole bodies plus a map of signatures "
    "(mode=focus), so whole files rarely need to be opened. Record design decisions with zt_record_decision; check "
    "zt_status to see which symbols are being worked on. If a tool says zt is not enabled for the project, read files "
    "directly and enable it only if the user asks."
)
INSTRUCTIONS_READ_SPAN = (
    " To see the body of something listed only by its line range, call zt_read_span with those ranges (up to 3 per "
    "call) instead of reading whole files."
)

TOOLS: list[dict[str, Any]] = [
    {
        "name": "zt_query_context",
        "title": "Query code context",
        "description": (
            "Return the code spans of this repository most relevant to a task or question (bug report, feature, "
            "identifier, error message), selected by lexical + embedding search and recent-edit energy, merged per "
            "file and cut to a token budget. Each span is headed by `path  Lstart-end  [symbols]` and holds the exact "
            "current text. Refreshes the index first (only changed files are re-read). Best results: write the query in "
            "the language of the code's identifiers and comments (usually English) even if the user writes in another "
            "language - if unsure, include both - and add the identifiers, file names or error messages you expect "
            "(the code is searched both by meaning and by exact words). If there is a traceback or compiler/test error, "
            "paste its file:line lines verbatim: the symbols at those lines are pinned first. With mode=\"locations\" "
            "only file, line range, signature and first docstring line of each match come back (~500 tokens), best "
            "first: open the lines you need with your own file-reading tool. Prefer it in long sessions, where a "
            "returned code body is re-read on every later turn. mode=\"focus\" sits in between: the few most "
            "relevant bodies in full, the rest as a map of signatures."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "the task or question, in natural language or code terms"},
                "token_budget": {"type": "integer", "minimum": 100, "maximum": 200000,
                                 "description": "maximum size of the result in tokens (code: default 4000, ask for more only if "
                                                "needed; locations: default 500)"},
                "mode": {"type": "string", "enum": list(QUERY_MODES),
                         "description": "code: the exact code spans; locations: only where to look (file, lines, "
                                        "signatures, docstrings); focus: the 5 best bodies in full plus a map of "
                                        "signatures with line ranges (10 bodies when the query holds a traceback). "
                                        "Default: the server's setting (`zt mcp --mode`, locations if not set)"},
                "line_numbers": {"type": "boolean", "description": "prefix every line with its number (default false)"},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        "annotations": {"readOnlyHint": True, "idempotentHint": True, "openWorldHint": False},
    },
    {
        "name": "zt_record_decision",
        "title": "Record design decision",
        "description": (
            "Store an architectural or design decision in the project memory. It is returned by later "
            "zt_query_context calls about the linked code. `paths` may name files (`src/app/db.py`) or symbols "
            "(`src/app/db.py::Pool.acquire`)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "summary": {"type": "string", "description": "the decision and its reason, one or a few sentences"},
                "paths": {"type": "array", "items": {"type": "string"}, "description": "files or file::symbol it concerns"},
            },
            "required": ["summary"],
            "additionalProperties": False,
        },
        "annotations": {"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
    },
    {
        "name": "zt_status",
        "title": "Memory status",
        "description": ("List the hottest symbols (highest energy: recently edited or retrieved), files changed since "
                        "the last sync, and store size."),
        "inputSchema": {
            "type": "object",
            "properties": {"limit": {"type": "integer", "minimum": 1, "maximum": 100, "description": "symbols to list (default 10)"}},
            "additionalProperties": False,
        },
        "annotations": {"readOnlyHint": True, "idempotentHint": True, "openWorldHint": False},
    },
]
READ_SPAN_TOOL: dict[str, Any] = {
    "name": "zt_read_span",
    "title": "Read line ranges",
    "description": (
        "Read up to 3 line ranges of this project's files, exactly as they are on disk now. Use it for code that "
        "zt_query_context listed only by line range (`path  Lstart-end  signature`): copy those ranges here instead "
        "of reading whole files. Write each span as `path:L<start>-<end>` with the path relative to the project "
        f"root. At most {MAX_SPAN_LINES} lines per span and {MAX_TOTAL_LINES} per call. Files the project excludes "
        "(exclude_globs), hidden or git-ignored files and paths outside the project are refused."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "spans": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": MAX_READ_SPANS,
                      "description": "e.g. [\"src/app/db.py:L120-180\", \"src/app/pool.py:L12-40\"]"},
            "line_numbers": {"type": "boolean", "description": "prefix every line with its number (default false)"},
        },
        "required": ["spans"],
        "additionalProperties": False,
    },
    "annotations": {"readOnlyHint": True, "idempotentHint": True, "openWorldHint": False},
}
_LOCATIONS_TAIL = "open the lines you need with your own file-reading tool."
_LOCATIONS_TAIL_SPAN = "read the lines you need with zt_read_span (or your own file-reading tool)."


def tool_definitions(read_span: bool) -> list[dict[str, Any]]:
    """The advertised tools; with `read_span` the query tool points to zt_read_span for bodies it did not return."""
    if not read_span:
        return TOOLS
    query = dict(TOOLS[0], description=TOOLS[0]["description"].replace(_LOCATIONS_TAIL, _LOCATIONS_TAIL_SPAN))
    return [query, *TOOLS[1:], READ_SPAN_TOOL, {
        "name": "zt_verify_tests",
        "title": "Run pytest via Dissipative Interferometer",
        "description": (
            "Execute pytest targets through the Dissipative Interferometer: cancels passing boilerplate to 1 line (~24 tokens, "
            "-96%), extracts deduplicated fault waves on failures (-87%), and injects a Newtonian failure shockwave into "
            "memory to heat the responsible production symbols."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "targets": {"type": "array", "items": {"type": "string"},
                            "description": "pytest targets, e.g. [\"tests/test_app.py::test_login\"]"},
                "query": {"type": "string", "description": "optional query to auto-select resonant `[test]` targets"},
                "max_tokens": {"type": "integer", "minimum": 50, "maximum": 2000, "description": "max output tokens (default 220)"},
            },
            "additionalProperties": False,
        },
        "annotations": {"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
    }]




class ToolInputError(ValueError):
    """Bad arguments: reported to the model as a tool error it can correct."""


class ToolUnavailable(RuntimeError):
    """The tool cannot answer yet (no project, empty index): reported to the model with the steps to fix it."""


ACTIVATION_STEPS = (
    "1. `zt init --encoder hf:Qwen/Qwen3-Embedding-0.6B --rank 128` in the project root.\n"
    "2. Before any sync, put files an agent must not read (real data, secrets, fixtures with real content, files the "
    "project rules forbid) in `exclude_globs` under [memory] of `.zt/config.toml`, and have the user confirm them.\n"
    "3. Check the file list with `zt status --json` (pending.added), then run `zt sync` in the background "
    "(the first one takes 15-45 minutes on CPU).\n"
    "4. Call this tool again: no client restart is needed."
)


@dataclass
class _Request:
    id: Any
    method: str
    params: dict[str, Any]


class ZtTools:
    """The tool implementations, independent of the transport."""

    def __init__(self, project: ProjectConfig | None = None, *, start: Path | None = None,
                 encoder_factory: Callable[[ProjectConfig], HiddenStateEncoder] | None = None,
                 default_budget: int | None = None, device: str | None = None, idle_unload_s: float = 900.0,
                 max_inline_sync_files: int = 100, default_mode: str = "code", read_span: bool = False,
                 log: Callable[[str], None] = lambda s: None) -> None:
        if default_mode not in QUERY_MODES:
            raise ValueError(f"default_mode must be one of {QUERY_MODES}")
        self._project = project
        self.default_mode = default_mode
        self.read_span_enabled = read_span
        self._start = Path(start or Path.cwd())
        self.default_budget = default_budget
        self.device = device
        self.idle_unload_s = idle_unload_s
        self.max_inline_sync_files = max_inline_sync_files
        self._log = log
        self._factory = encoder_factory or (lambda pc: pc.build_encoder())
        self._encoder: HiddenStateEncoder | None = None
        self._pauli: Any = None
        self._use = threading.RLock()  # held for a whole tool call and by the idle reaper
        self._last_used = time.monotonic()
        self._reaper: threading.Thread | None = None
        self._closed = threading.Event()

    # -- project ------------------------------------------------------------------------------
    @property
    def project(self) -> ProjectConfig | None:
        return self._project

    def _pauli_for(self, project: ProjectConfig) -> Any:
        if self._pauli is None:
            from .pauli import PauliSessionCache
            self._pauli = PauliSessionCache(project.store / "pauli_session.json")
        return self._pauli

    def resolve_project(self) -> ProjectConfig:
        if self._project is None:
            try:
                found = ProjectConfig.find(self._start)
            except ProjectConfigError:
                raise ToolUnavailable(
                    f"zt-memory is not enabled for this project: no {CONFIG_DIR}/config.toml in {self._start} or its "
                    f"parents. Read files directly, or enable it (ask the user first):\n{ACTIVATION_STEPS}") from None
            if self.device:
                found.encoder["device"] = self.device
            self._project = found
            self._log(f"project found: {found.repo_root} (store {found.store})")
        return self._project

    # -- encoder lifecycle --------------------------------------------------------------------
    def preload(self) -> None:
        """Load the encoder now, in the background (opt-in: `zt mcp --preload`)."""
        def run() -> None:
            try:
                with self._use:
                    self._encoder_for(self.resolve_project())
            except Exception as exc:  # noqa: BLE001 - surfaced again on the first tool call
                self._log(f"preload skipped: {exc}")
        threading.Thread(target=run, name="zt-preload", daemon=True).start()

    def _encoder_for(self, project: ProjectConfig) -> HiddenStateEncoder:
        if self._encoder is None:
            t = time.perf_counter()
            self._encoder = self._factory(project)
            self._log(f"encoder loaded in {time.perf_counter() - t:.1f}s: {project.encoder.get('spec')}")
            if self.idle_unload_s > 0 and self._reaper is None:
                self._reaper = threading.Thread(target=self._reap, name="zt-idle-unload", daemon=True)
                self._reaper.start()
        self._last_used = time.monotonic()
        return self._encoder

    def _reap(self) -> None:
        tick = max(1.0, min(30.0, self.idle_unload_s / 4))
        while not self._closed.wait(tick):
            if self._encoder is None or time.monotonic() - self._last_used < self.idle_unload_s:
                continue
            if self._use.acquire(blocking=False):  # never while a tool call is running
                try:
                    if self._encoder is not None and time.monotonic() - self._last_used >= self.idle_unload_s:
                        self.unload()
                finally:
                    self._use.release()

    def unload(self) -> None:
        with self._use:
            if self._encoder is None:
                return
            release = getattr(self._encoder, "release", None)
            if callable(release):
                release()
            self._encoder = None
            import gc

            gc.collect()
            self._log(f"encoder released after {self.idle_unload_s:.0f}s idle")

    @property
    def encoder_loaded(self) -> bool:
        return self._encoder is not None

    @contextmanager
    def _session(self) -> Iterator[Any]:
        with self._use:
            project = self.resolve_project()
            if not (project.store / "meta.sqlite3").exists() or not (project.store / "projector.pt").exists():
                raise ToolUnavailable(
                    f"zt-memory is configured in {project.repo_root} but the index was never built. Run `zt sync` there "
                    f"in the background (first run: 15-45 minutes on CPU), after checking exclude_globs. Steps:\n{ACTIVATION_STEPS}")
            mgr = project.open_manager(self._encoder_for(project))
            try:
                yield mgr
            finally:
                mgr.close()
                self._last_used = time.monotonic()

    def _refresh(self, mgr) -> str:  # noqa: ANN001
        """Incremental sync when cheap; otherwise answer from the current index and say so."""
        pending = mgr.scan_changes().pending
        if not pending:
            return ""
        if len(pending) > self.max_inline_sync_files:
            return (f"; index NOT refreshed: {len(pending)} files changed since the last sync (more than "
                    f"{self.max_inline_sync_files}); results may be stale until `zt sync` runs")
        report = mgr.sync()
        return f"; index refreshed: {report.summary()}"

    def close(self) -> None:
        self._closed.set()
        self.unload()

    # -- tools --------------------------------------------------------------------------------
    @property
    def tool_definitions(self) -> list[dict[str, Any]]:
        return tool_definitions(self.read_span_enabled)

    @property
    def instructions(self) -> str:
        return INSTRUCTIONS + (INSTRUCTIONS_READ_SPAN if self.read_span_enabled else "")

    def call(self, name: str, args: dict[str, Any]) -> str:
        if name == "zt_query_context":
            mode = args.get("mode")
            if mode is not None and mode not in QUERY_MODES:
                raise ToolInputError(f"`mode` must be one of {', '.join(QUERY_MODES)}")
            budget = args.get("token_budget")
            if (mode or self.default_mode) != "locations" and isinstance(budget, int) and budget < 256:
                raise ToolInputError(f"`token_budget` must be at least 256 in {mode or self.default_mode} mode")
            return self.query_context(_str(args, "query", required=True), token_budget=_int(args, "token_budget", 100, 200_000),
                                      line_numbers=bool(args.get("line_numbers", False)), mode=mode)
        if name == "zt_record_decision":
            paths = args.get("paths") or []
            if not isinstance(paths, list) or not all(isinstance(p, str) for p in paths):
                raise ToolInputError("`paths` must be a list of strings")
            return self.record_decision(_str(args, "summary", required=True), paths)
        if name == "zt_status":
            return self.status(limit=_int(args, "limit", 1, 100) or 10)
        if name == "zt_read_span" and self.read_span_enabled:
            spans = args.get("spans")
            if isinstance(spans, str):
                spans = [spans]
            if not isinstance(spans, list) or not spans or not all(isinstance(x, str) for x in spans):
                raise ToolInputError("`spans` must be a list of strings like \"src/app.py:L10-40\"")
            if len(spans) > MAX_READ_SPANS:
                raise ToolInputError(f"at most {MAX_READ_SPANS} spans per call")
            return self.read_spans(spans, line_numbers=bool(args.get("line_numbers", False)))
        if name == "zt_verify_tests" and self.read_span_enabled:
            targets = args.get("targets") or []
            if isinstance(targets, str):
                targets = [targets]
            if not isinstance(targets, list) or not all(isinstance(x, str) for x in targets):
                raise ToolInputError("`targets` must be a list of pytest target strings")
            query = _str(args, "query", required=False) or None
            max_tok = _int(args, "max_tokens", 50, 2000) or 220
            return self.verify_tests(targets, query=query, max_tokens=max_tok)
        raise ToolInputError(f"unknown tool {name!r}")

    def query_context(self, query: str, *, token_budget: int | None = None, line_numbers: bool = False,
                      mode: str | None = None) -> str:
        from .router import GeometricRouter

        mode = mode or self.default_mode
        with self._session() as mgr:
            project = self.resolve_project()
            rc = project.router_config()
            if mode == "locations":  # select with API anti-clone/MMR + harmonic octave separation
                budget = int(project.router.get("token_budget", 12_000))
                rc = replace(rc, **ROUTER_PRESETS["api"])
            elif mode == "focus":  # select as for the local model; the focus rule keeps few bodies whole
                budget = token_budget or int(project.router.get("token_budget", 12_000))
                rc = replace(rc, **ROUTER_PRESETS["api"])
            else:
                budget = token_budget or self.default_budget or MCP_DEFAULT_BUDGET
            note = self._refresh(mgr)
            pauli = self._pauli_for(project) if mode == "focus" else None
            ctx = GeometricRouter(mgr, rc).route_to_compute_engine(query, token_budget=budget, pauli_session=pauli)
            if pauli is not None:
                pauli.save()
        if mode == "locations":
            loc = ctx.render_locations(max_tokens=token_budget or LOCATIONS_DEFAULT_TOKENS)
            head = (f"zt-memory: {loc.entries} location(s) in {len(loc.files)} file(s), ~{loc.tokens} tokens, best first "
                    f"(signatures only: read the lines you need){note}")
            if ctx.pinned:
                head += f"\npinned from the error output in the query: {', '.join(ctx.pinned)}"
            if not loc.entries:
                return head + "\nNo indexed code matches this query. Try identifiers, file names or error text."
            return head + "\n" + loc.text
        files = {b.path for b in ctx.blocks if b.path}
        whole = sum(1 for b in ctx.blocks if b.full)
        shape = f"{whole} whole, {len(ctx.blocks) - whole} as signatures" if mode == "focus" else f"{len(ctx.blocks)} span(s)"
        head = (f"zt-memory: {shape} from {len(files)} file(s), ~{ctx.token_estimate} tokens "
                f"(budget {budget}){note}")
        if ctx.pinned:
            head += f"\npinned from the error output in the query: {', '.join(ctx.pinned)}"
        if not ctx.blocks:
            return head + "\nNo indexed code matches this query. Try identifiers, file names or error text."
        hint = ("get a body with zt_read_span (path:Lstart-end) if you need it" if self.read_span_enabled
                else "read the lines you need if you need a body")
        return head + "\n" + ctx.render_context(line_numbers=line_numbers, map_hint=hint)

    def read_spans(self, spans: list[str], *, line_numbers: bool = False) -> str:
        """The NEED round over MCP: spans read from disk under the index rules + Pauli Exclusion & Spectroscopic TOC."""
        from .router import _LANG
        from .spectroscopy import render_spectroscopic_toc

        project = self.resolve_project()
        policy = SpanPolicy(project.repo_root, project.memory_config())
        pauli = self._pauli_for(project)
        out: list[str] = []
        refused: list[str] = []
        left = MAX_TOTAL_LINES
        lines_read = 0
        lines_saved = 0
        for raw in spans:
            if left <= 0:
                refused.append(f"{raw}: not read, the {MAX_TOTAL_LINES}-line limit of one call was reached")
                continue
            try:
                path, start, end, force_full, is_toc = parse_span_ex(raw)
                if is_toc:
                    rel, target = policy.resolve(path)
                    full_text = target.read_text(encoding="utf-8", errors="replace")
                    toc_block = render_spectroscopic_toc(rel, full_text)
                    out.append(toc_block)
                    toc_lines = len(toc_block.splitlines())
                    lines_read += toc_lines
                    left -= min(left, toc_lines)
                    continue
                rel, text, first, last, total = policy.read(path, start, end, max_lines=min(MAX_SPAN_LINES, left))
            except SpanRefused as exc:
                refused.append(f"{raw}: {exc}")
                continue
            cut = " (cut: ask for the rest in another call)" if last < min(max(start, end), total) else ""
            lang = _LANG.get(Path(rel).suffix.lower(), "")
            outcome = pauli.read_with_refraction(
                rel, text, first, last, total,
                lang=lang, line_numbers=line_numbers, cut_note=cut, force_full=force_full,
            )
            left -= last - first + 1
            lines_read += outcome.lines_emitted
            lines_saved += outcome.lines_saved
            out.append(outcome.rendered_block)
        pauli.save()
        if not out:
            raise ToolInputError("nothing read:\n" + "\n".join(f"- {r}" for r in refused))
        saved_note = f" (pauli refracted: {lines_saved} duplicate lines saved)" if lines_saved > 0 else ""
        head = f"zt-memory: {len(out)} span(s), {lines_read} lines, read from disk now{saved_note}"
        tail = ["", "refused:", *(f"- {r}" for r in refused)] if refused else []
        return "\n".join([head, *out, *tail])

    def record_decision(self, summary: str, paths: list[str]) -> str:
        with self._session() as mgr:
            note = self._refresh(mgr)
            related: list[int] = []
            linked: list[str] = []
            missing: list[str] = []
            for raw in paths:
                target = raw.strip().replace("\\", "/")
                path, _, symbol = target.partition("::")
                try:
                    rel = mgr._normalize_rel(path)  # noqa: SLF001 - same normalization as the sync
                except ValueError:
                    missing.append(raw)
                    continue
                if symbol:
                    rec = mgr.record_by_key(f"chunk:{rel}::{symbol}")
                    ids = [rec.id] if rec is not None else []
                else:
                    rec = mgr.record_by_key(f"file:{rel}")
                    # the file node plus its symbols: graph expansion starts from symbol-level seeds
                    ids = [rec.id, *(c.id for c in mgr.meta.chunks_of_file(rel))] if rec is not None else []
                if ids:
                    related.extend(ids)
                    linked.append(target)
                else:
                    missing.append(raw)
            node = mgr.record_decision(summary.strip(), related_ids=sorted(set(related)))
        out = [f"decision #{node} recorded" + (f", linked to {', '.join(linked)}" if linked else " (not linked to code)")]
        if missing:
            out.append(f"not found in the index (check the path, or `file::Class.method` for symbols): {', '.join(missing)}")
        if note.startswith("; index NOT"):
            out.append(note[2:])
        return "\n".join(out)

    def verify_tests(self, targets: list[str] | None = None, *, query: str | None = None, max_tokens: int = 220) -> str:
        """Execute resonant tests through the Dissipative Interferometer and inject failure shockwaves into memory."""
        from .interferometer import run_interferometer_verify
        from .router import GeometricRouter

        with self._session() as mgr:
            project = self.resolve_project()
            self._refresh(mgr)
            chosen_targets = list(targets or [])
            if not chosen_targets and query:
                rc = replace(project.router_config(), **ROUTER_PRESETS["api"])
                ctx = GeometricRouter(mgr, rc).route_to_compute_engine(query, token_budget=2000)
                if ctx.harmonic_field and ctx.harmonic_field.resonant_tests:
                    chosen_targets = [rt.pytest_target for rt in ctx.harmonic_field.resonant_tests[:2]]
            rep = run_interferometer_verify(
                project.repo_root,
                chosen_targets,
                manager=mgr,
                max_tokens=max_tokens,
            )
            return rep.compact_output

    def status(self, *, limit: int = 10) -> str:
        with self._session() as mgr:
            cs = mgr.scan_changes()
            st = mgr.stats()
            hot = mgr.hottest(limit, frozenset({NodeKind.CHUNK}))
            recs = mgr.records(i for i, _ in hot)
            project = self.resolve_project()
        lines = [f"repo {project.repo_root}",
                 "nodes " + ", ".join(f"{k}={v}" for k, v in st.nodes_by_kind.items() if v) + f" | {st.disk_bytes / 2**20:.1f} MiB"]
        pending = len(cs.pending)
        lines.append("index up to date" if not pending else
                     f"{pending} file(s) changed since last sync (+{len(cs.added)} ~{len(cs.changed)} -{len(cs.removed)}): "
                     + ", ".join(cs.pending[:10]) + (" ..." if pending > 10 else ""))
        lines.append("hottest symbols (energy, path::symbol, lines):")
        for i, e in hot:
            r = recs.get(i)
            if r is None:
                continue
            span = f"L{r.start_line}-{r.end_line}" if r.start_line else ""
            lines.append(f"  {e:7.3f}  {r.path}::{r.symbol}  {span}")
        return "\n".join(lines)


class MultiProjectTools:
    """Several projects behind one server (`zt mcp --project PATH --project NAME=PATH`), for clients with one server
    entry and no session folder, like Claude Desktop. Every tool takes `project`; each project keeps its own encoder,
    loaded on its first call and released when idle like with one project."""

    def __init__(self, projects: dict[str, ZtTools]) -> None:
        if not projects:
            raise ValueError("no projects")
        self.projects = projects

    @property
    def tool_definitions(self) -> list[dict[str, Any]]:
        first = next(iter(self.projects.values()))
        names = list(self.projects)
        out = copy.deepcopy(first.tool_definitions)
        for tool in out:
            schema = tool["inputSchema"]
            schema["properties"] = {"project": {"type": "string", "enum": names,
                                                "description": "which project to work on: " + ", ".join(names)},
                                    **schema["properties"]}
            schema["required"] = ["project", *schema.get("required", [])]
        return out

    @property
    def instructions(self) -> str:
        first = next(iter(self.projects.values()))
        return (first.instructions + " This server holds several projects (" + ", ".join(self.projects) + "): pass "
                "`project` on every call, the one the user is talking about; if unclear, ask the user.")

    def call(self, name: str, args: dict[str, Any]) -> str:
        args = dict(args)
        project = args.pop("project", None)
        if project not in self.projects:
            raise ToolInputError(f"`project` must be one of {', '.join(self.projects)}")
        return self.projects[project].call(name, args)

    def preload(self) -> None:
        for tools in self.projects.values():
            tools.preload()

    def close(self) -> None:
        for tools in self.projects.values():
            tools.close()


def project_names(paths: list[str]) -> dict[str, Path]:
    """`PATH` or `NAME=PATH` -> {name: path}; the name defaults to the folder name and must be unique."""
    out: dict[str, Path] = {}
    for raw in paths:
        name, sep, path = raw.partition("=")
        if not sep:
            name, path = "", raw
        folder = Path(path).expanduser().resolve()
        name = name.strip() or folder.name
        if name in out:
            raise ValueError(f"two projects named {name!r}: give one a name with NAME=PATH")
        if not folder.is_dir():
            raise ValueError(f"project folder not found: {folder}")
        out[name] = folder
    return out


def _str(args: dict[str, Any], key: str, *, required: bool = False) -> str:
    value = args.get(key)
    if value is None and not required:
        return ""
    if not isinstance(value, str) or (required and not value.strip()):
        raise ToolInputError(f"`{key}` must be a non-empty string")
    return value


def _int(args: dict[str, Any], key: str, lo: int, hi: int) -> int | None:
    value = args.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or int(value) != value:
        raise ToolInputError(f"`{key}` must be an integer")
    if not lo <= int(value) <= hi:
        raise ToolInputError(f"`{key}` must be between {lo} and {hi}")
    return int(value)


class McpStdioServer:
    """Line-delimited JSON-RPC over binary streams (stdin/stdout by default)."""

    def __init__(self, tools: ZtTools | MultiProjectTools, *, inp: BinaryIO | None = None, out: BinaryIO | None = None,
                 log: Callable[[str], None] | None = None) -> None:
        self.tools = tools
        self._in = inp if inp is not None else sys.stdin.buffer
        self._out = out if out is not None else sys.stdout.buffer
        self._log = log or (lambda s: print(f"[zt-mcp] {s}", file=sys.stderr, flush=True))
        self._write_lock = threading.Lock()

    def _send(self, message: dict[str, Any]) -> None:
        data = json.dumps(message, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n"
        with self._write_lock:
            self._out.write(data)
            self._out.flush()

    def _result(self, id_: Any, result: dict[str, Any]) -> None:
        self._send({"jsonrpc": "2.0", "id": id_, "result": result})

    def _error(self, id_: Any, code: int, message: str) -> None:
        self._send({"jsonrpc": "2.0", "id": id_, "error": {"code": code, "message": message}})

    def serve(self) -> None:
        for raw in self._in:
            line = raw.strip()
            if not line:
                continue
            try:
                msg = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                self._error(None, PARSE_ERROR, f"parse error: {exc}")
                continue
            for item in (msg if isinstance(msg, list) else [msg]):  # batches: accepted for older clients
                self._dispatch(item)

    def _dispatch(self, msg: Any) -> None:
        if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0" or not isinstance(msg.get("method", ""), str):
            if isinstance(msg, dict) and ("result" in msg or "error" in msg):
                return  # a response to a request we never send
            self._error(msg.get("id") if isinstance(msg, dict) else None, INVALID_REQUEST, "invalid JSON-RPC 2.0 message")
            return
        if "method" not in msg:
            return
        params = msg.get("params") or {}
        if "id" not in msg:  # notification: never answered
            if msg["method"] not in ("notifications/initialized", "notifications/cancelled"):
                self._log(f"ignored notification {msg['method']}")
            return
        req = _Request(msg["id"], msg["method"], params if isinstance(params, dict) else {})
        try:
            handler = {"initialize": self._initialize, "ping": lambda r: {}, "tools/list": self._tools_list,
                       "tools/call": self._tools_call}.get(req.method)
            if handler is None:
                self._error(req.id, METHOD_NOT_FOUND, f"method not found: {req.method}")
                return
            result = handler(req)
            if result is not None:
                self._result(req.id, result)
        except Exception as exc:  # noqa: BLE001 - a broken request must not kill the server
            self._log(traceback.format_exc())
            self._error(req.id, INTERNAL_ERROR, f"internal error: {exc}")

    def _initialize(self, req: _Request) -> dict[str, Any]:
        asked = req.params.get("protocolVersion")
        version = asked if asked in SUPPORTED_PROTOCOL_VERSIONS else SUPPORTED_PROTOCOL_VERSIONS[0]
        client = (req.params.get("clientInfo") or {}).get("name", "?")
        self._log(f"initialize from {client} (protocol {asked} -> {version})")
        return {"protocolVersion": version, "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION}, "instructions": self.tools.instructions}

    def _tools_list(self, req: _Request) -> dict[str, Any]:
        return {"tools": self.tools.tool_definitions}

    def _tools_call(self, req: _Request) -> dict[str, Any] | None:
        name = req.params.get("name")
        args = req.params.get("arguments") or {}
        known = {t["name"] for t in self.tools.tool_definitions}
        if name not in known or not isinstance(args, dict):
            self._error(req.id, INVALID_PARAMS, f"unknown tool {name!r}" if name not in known else "`arguments` must be an object")
            return None
        try:
            text, is_error = self.tools.call(str(name), args), False
        except ToolInputError as exc:
            text, is_error = f"invalid arguments: {exc}", True
        except ToolUnavailable as exc:
            text, is_error = str(exc), True
        except StoreLockedError as exc:
            text, is_error = f"the memory store is busy (another zt process is syncing): {exc}. Retry in a few seconds.", True
        except Exception as exc:  # noqa: BLE001 - execution errors go back to the model, not the protocol
            self._log(traceback.format_exc())
            text, is_error = f"{type(exc).__name__}: {exc}", True
        return {"content": [{"type": "text", "text": text}], "isError": is_error}


def serve_stdio(project: ProjectConfig | None = None, *, start: Path | None = None, default_budget: int | None = None,
                device: str | None = None, preload: bool = False, idle_unload_s: float = 900.0,
                max_inline_sync_files: int = 100, default_mode: str = "locations", read_span: bool = False,
                projects: dict[str, Path] | None = None) -> None:
    """Entry point of `zt mcp`: protocol on the real stdout, everything else on stderr."""
    out = sys.stdout.buffer
    inp = sys.stdin.buffer
    sys.stdout = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)
    log = lambda s: print(f"[zt-mcp] {s}", file=sys.stderr, flush=True)  # noqa: E731

    def make(pc: ProjectConfig | None, folder: Path | None, tag: str = "") -> ZtTools:
        return ZtTools(pc, start=folder, default_budget=default_budget, device=device, idle_unload_s=idle_unload_s,
                       max_inline_sync_files=max_inline_sync_files, default_mode=default_mode, read_span=read_span,
                       log=(lambda s: log(f"[{tag}] {s}")) if tag else log)

    tools: ZtTools | MultiProjectTools
    if projects:
        subs = {}
        for name, folder in projects.items():
            pc = ProjectConfig.load(folder) if (folder / CONFIG_DIR / "config.toml").is_file() else None
            if pc is not None and device:
                pc.encoder["device"] = device
            subs[name] = make(pc, folder, name)
            log(f"project {name}: {folder}" + ("" if pc else " (zt not enabled there yet)"))
        tools = MultiProjectTools(subs)
    else:
        tools = make(project, start)
        where = project.repo_root if project is not None else Path(start or Path.cwd())
        log(f"serving {where} ({'project ' + str(project.store) if project else 'project looked up on first call'})")
    if preload:
        tools.preload()
    try:
        McpStdioServer(tools, inp=inp, out=out, log=log).serve()
    finally:
        tools.close()
