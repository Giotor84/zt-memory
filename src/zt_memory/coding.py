"""End-to-end edit loop: navigator -> compute engine -> verified patch -> sync (physics) -> optional check.

    route (dynamic cut, budget = served context - answer reserve, verified with the server tokenizer)
      -> chat (SEARCH/REPLACE contract)
      -> plan all blocks in memory (all-or-nothing); on failure, feed the real file text back and retry
      -> confirm -> backup + atomic write
      -> manager.ingest_files(touched): content-hash diff, surprise + edit impulse heat the edited symbols
      -> optional check command (tests); on failure optionally undo and resync
"""
from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal

from .config import RouterConfig
from .editing import (
    SEARCH_REPLACE_CONTRACT, AppliedEdit, EditError, EditPlan, failure_feedback, parse_edit_blocks, plan_edits, undo_last,
    write_plan,
)
from .engine_client import ChatResult, ComputeEngineClient, ComputeEngineError
from .manager import NewtonianMemoryManager
from .need import NEED_CONTRACT, NeedSettings, NeedState, offer_need, parse_need
from .router import GeometricRouter, RoutedContext
from .spans import SpanPolicy

CODE_SYSTEM_PROMPT = (
    "You are a senior software engineer making a change to an existing repository. The context contains the "
    "relevant source spans, selected from the project memory, with their exact current text. Make the smallest "
    "correct change that fulfils the task, keep the existing style, and follow the output contract exactly."
)

Status = Literal["applied", "dry_run", "declined", "no_edits", "edit_failed", "check_failed", "reverted"]


@dataclass
class CodeResult:
    status: Status
    answer: str = ""
    attempts: int = 0
    context_tokens: int | None = None  # exact, from the server tokenizer when available
    context_budget: int = 0
    served_ctx: int | None = None
    blocks: int = 0
    llm: list[ChatResult] = field(default_factory=list)
    plan: EditPlan | None = None
    applied: AppliedEdit | None = None
    heated: list[tuple[str, float, float]] = field(default_factory=list)  # (symbol key, energy before, after)
    check_returncode: int | None = None
    check_output: str = ""
    errors: list[str] = field(default_factory=list)
    routed: RoutedContext | None = None
    need_rounds: int = 0
    need_requests: list[list[str]] = field(default_factory=list)

    @property
    def llm_seconds(self) -> float:
        return sum(r.seconds for r in self.llm)


def fit_context(router: GeometricRouter, client: ComputeEngineClient, instruction: str, *, served_ctx: int,
                max_new_tokens: int, budget_cap: int, margin: int = 128) -> tuple[RoutedContext, list[dict[str, str]], int | None, int]:
    """Route under budget = served context - answer reserve; shrink until the *exact* prompt size fits."""
    limit = served_ctx - max_new_tokens - margin
    if limit < 512:
        raise ComputeEngineError(f"server context {served_ctx} too small for max_new_tokens={max_new_tokens}")
    budget = min(budget_cap, limit)
    for _ in range(4):
        routed = router.route_to_compute_engine(instruction, token_budget=budget)
        messages = routed.to_chat_messages(CODE_SYSTEM_PROMPT, contract=SEARCH_REPLACE_CONTRACT)
        exact = client.count_tokens(messages[0]["content"] + "\n" + messages[1]["content"])
        if exact is None or exact + 32 <= limit:  # +32: chat template wrapper tokens
            return routed, messages, exact, budget
        budget = max(256, int(budget * (limit - 32) / exact * 0.95))
    return routed, messages, exact, budget


@dataclass
class EditAttempts:
    """Outcome of asking the engine for edits: the last plan (ok or not) and every exchange."""

    plan: EditPlan | None = None
    answer: str = ""
    attempts: int = 0
    blocks: int = 0
    llm: list[ChatResult] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    need_rounds: int = 0  # extra rounds granted because the model asked for code with NEED
    need_requests: list[list[str]] = field(default_factory=list)  # spans asked, per round

    @property
    def ok(self) -> bool:
        return self.plan is not None and self.plan.ok


def request_edits(client: ComputeEngineClient, messages: list[dict[str, str]], repo: Path, *, max_new_tokens: int,
                  temperature: float, retries: int, echo: Callable[[str], None] = lambda s: None,
                  need: NeedState | None = None) -> EditAttempts:
    """Chat until the answer resolves into a complete edit plan; on failure, feed back the real file text and retry.
    With `need`, an answer made only of NEED lines gets the requested code (round 2) instead of counting as a failure."""
    out = EditAttempts()
    retried = 0
    while True:
        out.attempts += 1
        stop = need.stop if need is not None else None  # END_NEED while a NEED answer is possible
        chat = client.chat_full(messages, max_tokens=max_new_tokens, temperature=temperature, stop=stop)
        out.llm.append(chat)
        out.answer = chat.text
        tps = f", {chat.tokens_per_second:.1f} tok/s" if chat.tokens_per_second else ""
        echo(f"[llm] call {out.attempts}: {chat.completion_tokens or '?'} tokens in {chat.seconds:.1f}s{tps} (finish: {chat.finish_reason})")
        try:
            blocks = parse_edit_blocks(chat.text)
        except EditError as exc:
            blocks, parse_error = [], str(exc)
        else:
            parse_error = ""
        out.blocks = len(blocks)
        if not blocks and need is not None and need.offered:
            asked = parse_need(chat.text, max_requests=need.settings.max_requests)
            needed, kept = need.resolve(asked) if asked.needs else ([], [])
            if needed:
                messages = need.next_messages(asked, needed, kept)
                out.need_rounds += 1
                out.need_requests.append([r.label() for r, *_ in needed])
                echo(f"[need] round {out.need_rounds}: {', '.join(out.need_requests[-1])}")
                continue
        if not blocks:
            out.errors.append(parse_error or "no edit blocks in the answer")
            feedback = (f"Your answer could not be used: {parse_error or 'it contains no edit blocks'}."
                        + (" It was cut off by the length limit: send fewer, smaller blocks." if chat.finish_reason == "length" else "")
                        + " Reply again following the output contract exactly.")
        else:
            out.plan = plan_edits(repo, blocks)
            if out.plan.ok:
                return out
            out.errors.extend(f"block {f.block.index} ({f.block.path}): {f.reason}" for f in out.plan.failures)
            feedback = failure_feedback(out.plan, repo)
        if retried >= retries:
            return out
        retried += 1
        echo(f"[edit] call {out.attempts} unusable ({out.errors[-1]}), retrying with feedback")
        messages = [*messages, {"role": "assistant", "content": chat.text}, {"role": "user", "content": feedback}]


def _run_check(cmd: str, cwd: Path, timeout_s: float) -> tuple[int, str]:
    try:
        proc = subprocess.run(cmd, shell=True, cwd=str(cwd), capture_output=True, text=True, timeout=timeout_s,
                              encoding="utf-8", errors="replace")
        return proc.returncode, (proc.stdout + proc.stderr)
    except subprocess.TimeoutExpired as exc:
        return 124, f"check timed out after {timeout_s:.0f}s: {exc}"


def run_code_task(
    manager: NewtonianMemoryManager,
    client: ComputeEngineClient,
    instruction: str,
    *,
    router_config: RouterConfig | None = None,
    max_new_tokens: int = 3072,
    temperature: float = 0.1,
    retries: int = 1,
    budget_cap: int | None = None,
    default_ctx: int = 16384,
    dry_run: bool = False,
    confirm: Callable[[EditPlan], bool] | None = None,
    check_cmd: str | None = None,
    check_timeout_s: float = 1800.0,
    revert_on_fail: bool = False,
    echo: Callable[[str], None] = lambda s: None,
    need: NeedSettings | None = None,
) -> CodeResult:
    repo = manager.config.repo_root
    res = CodeResult("no_edits")
    rc = router_config or RouterConfig()
    manager.sync()
    res.served_ctx = client.context_size()
    served = res.served_ctx or default_ctx
    router = GeometricRouter(manager, rc)
    routed, messages, exact, budget = fit_context(router, client, instruction, served_ctx=served, max_new_tokens=max_new_tokens,
                                                  budget_cap=budget_cap or rc.token_budget)
    res.routed, res.context_tokens, res.context_budget = routed, exact, budget
    echo(f"[route] {len(routed.blocks)} blocks, {exact if exact is not None else routed.token_estimate} tokens"
         f"{' (exact)' if exact is not None else ' (estimated)'} | budget {budget} | served ctx {served}")

    state = None
    if offer_need(routed, need):
        assert need is not None
        messages = routed.to_chat_messages(CODE_SYSTEM_PROMPT, contract=SEARCH_REPLACE_CONTRACT + "\n\n" + NEED_CONTRACT)
        state = NeedState(need, repo, routed, instruction, CODE_SYSTEM_PROMPT, SEARCH_REPLACE_CONTRACT,
                          policy=SpanPolicy(repo, manager.config))
        echo(f"[need] offered: up to {need.rounds} extra round(s)")
    got = request_edits(client, messages, repo, max_new_tokens=max_new_tokens, temperature=temperature, retries=retries,
                        echo=echo, need=state)
    res.llm, res.answer, res.attempts, res.blocks, res.plan = got.llm, got.answer, got.attempts, got.blocks, got.plan
    res.need_rounds, res.need_requests = got.need_rounds, got.need_requests
    res.errors.extend(got.errors)
    if got.plan is None or not got.plan.ok:
        res.status = "edit_failed" if res.blocks else "no_edits"
        return res
    plan = got.plan
    if dry_run:
        res.status = "dry_run"
        return res
    if confirm is not None and not confirm(plan):
        res.status = "declined"
        return res

    keys_before = _chunk_energies(manager, plan.new_contents)
    res.applied = write_plan(repo, plan, label=instruction[:200])
    touched = [*res.applied.changed, *res.applied.created]
    report = manager.ingest_files(touched)  # the edit drives surprise + impulse on exactly these symbols
    after = _chunk_energies(manager, plan.new_contents)
    changed_keys = {f"chunk:{s[1:].split('::', 1)[0]}::{s.split('::', 1)[1]}" for s in report.changed_symbols if s[0] in "~+"}
    res.heated = sorted(((k, keys_before.get(k, 0.0), after.get(k, 0.0)) for k in changed_keys if k in after),
                        key=lambda t: -(t[2] - t[1]))
    try:
        manager.add_ephemeral(f"recent edit: {instruction[:300]} -> {', '.join(touched)}")
    except Exception:  # noqa: BLE001 - memory note is best effort
        pass
    echo(f"[apply] {len(res.applied.changed)} changed, {len(res.applied.created)} created (backup {res.applied.backup_id}) | "
         f"sync: ~{report.chunks_updated} +{report.chunks_added} -{report.chunks_removed} chunks, surprise max {report.surprise_max:.3f}")
    res.status = "applied"

    if check_cmd:
        code, output = _run_check(check_cmd, repo, check_timeout_s)
        res.check_returncode, res.check_output = code, output
        echo(f"[check] `{check_cmd}` -> exit {code}")
        if code != 0:
            res.status = "check_failed"
            if revert_on_fail:
                undone = undo_last(repo)
                if undone:
                    manager.ingest_files(undone[1])
                    res.status = "reverted"
                    echo(f"[undo] restored {len(undone[1])} file(s) from backup {undone[0]}")
    return res


def _chunk_energies(manager: NewtonianMemoryManager, paths) -> dict[str, float]:  # noqa: ANN001
    recs = [r for p in paths for r in manager.meta.chunks_of_file(p)]
    if not recs:
        return {}
    energies = manager.energies([r.id for r in recs])
    return {r.key: float(e) for r, e in zip(recs, energies)}


def wait_for_engine(client: ComputeEngineClient, timeout_s: float = 0.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while True:
        if client.health():
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(1.0)
