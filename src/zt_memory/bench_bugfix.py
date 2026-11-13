"""End-to-end bug-fixing benchmark: same local model, same edit protocol, different context.

Cases are real bug fixes mined from a Python repository's git history (SWE-bench style):

    base = parent of the fix commit            (the buggy code the model sees)
    instruction = the fix commit's message     (the "bug report")
    tests = the test files the fix changed     (hidden from the model; applied only to grade)
    valid  <=>  tests fail on base + new tests  and  pass on fix

Conditions, all through `request_edits` (SEARCH/REPLACE, one feedback retry) and the same system prompt:

    bm25-files    BM25 over whole files, files packed in rank order until the budget is full (the last one cut):
                  the "fill the context" baseline that can fall into the recall trap
    zt            zt-memory router: hybrid lexical + embedding + energy, symbol spans, dynamic cut
    bm25-matched  control: bm25-files packed only up to the tokens zt actually used, to separate
                  "fewer tokens" from "better selection"

Graded by running the fix's test files; reported: resolve rate, exact prompt tokens, LLM seconds, attempts and
whether the gold (non-test) files were in the context. Paired comparison with an exact McNemar test.
"""
from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Callable, Iterable, Sequence

from .coding import CODE_SYSTEM_PROMPT, fit_context, request_edits
from .config import MemoryConfig, RouterConfig
from .discovery import discover
from .editing import SEARCH_REPLACE_CONTRACT
from .encoders import HiddenStateEncoder
from .engine_client import ComputeEngineClient
from .lexical import LEXICAL_STOPWORDS, lexical_tokens
from .manager import NewtonianMemoryManager
from .replay import _git, _rmtree, clean_message
from .router import _LANG, GeometricRouter

CONDITIONS = ("bm25-files", "zt", "bm25-matched")
BUG_WORDS = re.compile(r"\b(fix(es|ed)?|bug|regression|incorrect|wrong|broken|crash(es)?|error|issue|fail(s|ing|ure)?)\b", re.I)
_TEST_PATH = re.compile(r"(^|/)(tests?|testing)/|(^|/)test_[^/]*\.py$|_test\.py$|(^|/)conftest\.py$")


def is_test_file(rel: str) -> bool:
    return bool(_TEST_PATH.search(rel))


# ------------------------------------------------------------------------------------------
# cases
# ------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class BugCase:
    id: str
    fix: str
    base: str
    date: int  # commit time (unix), cases run oldest first so the memory follows the history
    instruction: str
    source_files: tuple[str, ...]  # gold: non-test files changed by the fix
    test_files: tuple[str, ...]  # changed/added by the fix, applied only for grading
    loc_changed: int
    failure_excerpt: str = ""  # pytest output on base + new tests (optional extra context)

    @classmethod
    def from_dict(cls, d: dict) -> "BugCase":
        return cls(d["id"], d["fix"], d["base"], int(d.get("date", 0)), d["instruction"], tuple(d["source_files"]),
                   tuple(d["test_files"]), int(d.get("loc_changed", 0)), d.get("failure_excerpt", ""))


def save_cases(path: Path, repo: Path, cases: Sequence[BugCase]) -> None:
    path.write_text(json.dumps({"repo": str(Path(repo).resolve()), "cases": [asdict(c) for c in cases]}, indent=2), encoding="utf-8")


def load_cases(path: Path) -> tuple[Path, list[BugCase]]:
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    return Path(data["repo"]), sorted((BugCase.from_dict(c) for c in data["cases"]), key=lambda c: c.date)


class Worktree:
    """One detached git worktree, moved from commit to commit (cheap) and always reset after a run."""

    def __init__(self, repo: Path, path: Path) -> None:
        self.repo, self.path = Path(repo).resolve(), Path(path).resolve()
        if not (self.path / ".git").exists():
            if self.path.exists():
                _rmtree(self.path)
            _git(self.repo, "worktree", "prune")
            _git(self.repo, "worktree", "add", "--detach", "--force", str(self.path), "HEAD")

    def checkout(self, sha: str) -> None:
        _git(self.path, "checkout", "-f", "-q", "--detach", sha)
        _git(self.path, "clean", "-fdq")

    def take_files(self, sha: str, paths: Sequence[str]) -> None:
        if paths:
            _git(self.path, "checkout", sha, "--", *paths)

    def remove(self) -> None:
        try:
            _git(self.repo, "worktree", "remove", "--force", str(self.path))
        except subprocess.CalledProcessError:
            _rmtree(self.path)
            _git(self.repo, "worktree", "prune")


def run_tests(root: Path, test_files: Sequence[str], *, python: str = sys.executable, timeout_s: float = 300.0,
              keep_addopts: bool = False) -> tuple[int, str]:
    """pytest on the given files against the working tree's code (src/ layout too), never an installed copy."""
    env = dict(os.environ)
    paths = [str(root / "src"), str(root)] if (root / "src").is_dir() else [str(root)]
    env["PYTHONPATH"] = os.pathsep.join([*paths, *([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])])
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    cmd = [python, "-m", "pytest", "-q", "-p", "no:cacheprovider", *([] if keep_addopts else ["-o", "addopts="]), *test_files]
    try:
        proc = subprocess.run(cmd, cwd=str(root), capture_output=True, text=True, encoding="utf-8", errors="replace",
                              timeout=timeout_s, env=env)
        return proc.returncode, proc.stdout + proc.stderr
    except subprocess.TimeoutExpired:
        return 124, f"timeout after {timeout_s:.0f}s"


def mine_cases(repo: Path, workdir: Path, *, max_commits: int = 600, limit: int = 15, max_source_files: int = 2,
               max_loc: int = 40, require_bug_words: bool = True, python: str = sys.executable, test_timeout_s: float = 120.0,
               keep_addopts: bool = False, excerpt_lines: int = 40, echo: Callable[[str], None] = print) -> list[BugCase]:
    """Fix commits that change Python source + tests, validated by running the tests before/after.
    The case keeps the last `excerpt_lines` lines of the failing test output (the traceback a user would paste)."""
    repo = Path(repo).resolve()
    raw = _git(repo, "log", "--no-merges", f"-n{max_commits}", "--format=%H%x00%P%x00%ct%x00%s%x00%b%x1e")
    candidates = []
    for rec in [r for r in raw.split("\x1e") if r.strip()]:
        sha, parents, ct, subject, body = (rec.strip("\n").split("\x00") + [""] * 5)[:5]
        if len(parents.split()) != 1 or (require_bug_words and not BUG_WORDS.search(subject)):
            continue
        source, tests, loc, other_code = [], [], 0, False
        for line in _git(repo, "diff", "--numstat", "--no-renames", parents, sha).splitlines():
            parts = line.split("\t")
            if len(parts) != 3:
                continue
            added, deleted, rel = parts
            if rel.endswith(".py") and is_test_file(rel):
                tests.append(rel)
            elif rel.endswith(".py"):
                source.append(rel)
                loc += (int(added) if added.isdigit() else 0) + (int(deleted) if deleted.isdigit() else 0)
            elif PurePosixPath(rel).suffix in (".c", ".h", ".pyx", ".rs", ".cpp", ".js", ".ts"):
                other_code = True
        if source and tests and not other_code and len(source) <= max_source_files and loc <= max_loc:
            candidates.append((sha, parents, int(ct), clean_message(subject, body, 600), source, tests, loc))
    echo(f"[mine] {len(candidates)} candidate fix commits (source+tests, <= {max_source_files} files, <= {max_loc} lines)")
    wt = Worktree(repo, Path(workdir) / "worktree")
    cases: list[BugCase] = []
    seen: set[str] = set()
    try:
        for sha, base, ct, message, source, tests, loc in candidates:
            if len(cases) >= limit:
                break
            fingerprint = _git(repo, "diff", base, sha, "--", *source)
            if fingerprint in seen:  # the same fix landed twice (backport, cherry-pick): keep one
                echo(f"[mine] skip {sha[:9]}: duplicate of an earlier case")
                continue
            seen.add(fingerprint)
            wt.checkout(base)
            present = [t for t in tests if _exists_at(repo, sha, t)]  # deleted test files cannot grade anything
            wt.take_files(sha, present)
            rc_before, out_before = run_tests(wt.path, present, python=python, timeout_s=test_timeout_s, keep_addopts=keep_addopts)
            if rc_before not in (1, 2):  # must fail as a test failure/collection error (not usage error, no tests, timeout)
                echo(f"[mine] skip {sha[:9]}: tests do not fail before the fix (exit {rc_before})")
                continue
            wt.take_files(sha, source)
            rc_after, out_after = run_tests(wt.path, present, python=python, timeout_s=test_timeout_s, keep_addopts=keep_addopts)
            if rc_after != 0:
                echo(f"[mine] skip {sha[:9]}: tests do not pass after the fix (exit {rc_after}): {_last_line(out_after)}")
                continue
            excerpt = "\n".join(out_before.strip().splitlines()[-excerpt_lines:])
            cases.append(BugCase(sha[:10], sha, base, ct, message, tuple(source), tuple(present), loc, excerpt))
            echo(f"[mine] ok   {sha[:9]}: {message[:70]!r} ({', '.join(source)}; {loc} lines)")
    finally:
        wt.remove()
    return sorted(cases, key=lambda c: c.date)


def _exists_at(repo: Path, sha: str, rel: str) -> bool:
    try:
        _git(repo, "cat-file", "-e", f"{sha}:{rel}")
        return True
    except subprocess.CalledProcessError:
        return False


def _last_line(text: str) -> str:
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    return lines[-1][:160] if lines else ""


# ------------------------------------------------------------------------------------------
# baseline context: BM25 over whole files
# ------------------------------------------------------------------------------------------
def _terms(text: str) -> list[str]:
    return [t for t in lexical_tokens(text) if t not in LEXICAL_STOPWORDS and len(t) > 1]


def bm25_rank_files(root: Path, files: Sequence[str], query: str, *, k1: float = 1.2, b: float = 0.75) -> list[tuple[str, float]]:
    """Okapi BM25 with code-aware tokens (identifiers whole + split), over whole files."""
    docs: dict[str, Counter] = {}
    for rel in files:
        try:
            docs[rel] = Counter(_terms((Path(root) / rel).read_text(encoding="utf-8", errors="replace")))
        except OSError:
            continue
    if not docs:
        return []
    n = len(docs)
    avgdl = sum(sum(c.values()) for c in docs.values()) / n
    df: Counter = Counter()
    for c in docs.values():
        df.update(c.keys())
    q = set(_terms(query))
    scores = []
    for rel, tf in docs.items():
        dl = sum(tf.values())
        s = 0.0
        for t in q:
            f = tf.get(t, 0)
            if f:
                idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                s += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * dl / max(avgdl, 1e-9)))
        scores.append((rel, s))
    return sorted(scores, key=lambda t: (-t[1], t[0]))


def _render_files(instruction: str, files: list[tuple[str, str, bool]]) -> str:
    out = ["## Code context (whole files, ranked by BM25)"]
    for rel, text, cut in files:
        lang = _LANG.get(PurePosixPath(rel).suffix.lower(), "")
        out.append(f"### {rel}{'  [truncated]' if cut else ''}\n```{lang}\n{text}\n```")
    out.append(f"\n## Task\n{instruction.strip()}")
    out.append(f"\n## Output contract\n{SEARCH_REPLACE_CONTRACT}")
    return "\n".join(out)


def file_pack_messages(root: Path, ranked: Sequence[tuple[str, float]], instruction: str, *, budget: int,
                       count: Callable[[str], int]) -> tuple[list[dict[str, str]], int, list[str]]:
    """Whole files in rank order until `budget` tokens; the first file that no longer fits is cut to fill the rest.
    Sizes are checked with `count` (the served tokenizer) on the final prompt and shrunk until it fits."""
    texts = {rel: (Path(root) / rel).read_text(encoding="utf-8", errors="replace") for rel, s in ranked if s > 0}
    order = [rel for rel, s in ranked if s > 0]
    scale = 1.0
    for _ in range(6):
        limit = int(budget * scale)
        packed: list[tuple[str, str, bool]] = []
        used = count(_render_files(instruction, []))
        for rel in order:
            body = texts[rel]
            need = count(f"### {rel}\n```python\n{body}\n```\n")
            if used + need <= limit:
                packed.append((rel, body, False))
                used += need
                continue
            room = limit - used
            if room > 64:  # cut this file to fill the remaining budget, then stop
                lines = body.splitlines()
                keep = max(1, int(len(lines) * room / max(need, 1)))
                packed.append((rel, "\n".join(lines[:keep]), True))
            break
        user = _render_files(instruction, packed)
        exact = count(CODE_SYSTEM_PROMPT + "\n" + user)
        if exact + 32 <= budget or not packed:
            return ([{"role": "system", "content": CODE_SYSTEM_PROMPT}, {"role": "user", "content": user}], exact,
                    [rel for rel, _, _ in packed])
        scale *= (budget - 32) / exact * 0.97
    return ([{"role": "system", "content": CODE_SYSTEM_PROMPT}, {"role": "user", "content": user}], exact,
            [rel for rel, _, _ in packed])


# ------------------------------------------------------------------------------------------
# running
# ------------------------------------------------------------------------------------------
@dataclass
class RunResult:
    case: str
    condition: str
    status: str  # resolved | tests_failed | edit_failed | no_edits | error
    resolved: bool
    context_tokens: int
    prompt_tokens: int
    completion_tokens: int
    llm_seconds: float
    attempts: int
    context_files: list[str] = field(default_factory=list)
    gold_in_context: float = 0.0
    patched_files: list[str] = field(default_factory=list)
    test_exit: int | None = None
    error: str = ""


def _count_fn(client: ComputeEngineClient) -> Callable[[str], int]:
    estimate = getattr(client, "estimate_tokens", None)  # Bedrock: no tokenizer, a conservative estimate

    def count(text: str) -> int:
        n = client.count_tokens(text)
        if n is not None:
            return n
        return int(estimate(text)) if callable(estimate) else math.ceil(len(text) / 3.0)
    return count


def run_benchmark(
    repo: Path,
    cases: Sequence[BugCase],
    workdir: Path,
    client: ComputeEngineClient,
    encoder: HiddenStateEncoder,
    *,
    conditions: Sequence[str] = CONDITIONS,
    rank: int = 128,
    router_config: RouterConfig | None = None,
    max_new_tokens: int = 2048,
    temperature: float = 0.0,
    retries: int = 1,
    served_ctx: int | None = None,
    with_failure_log: bool = False,
    python: str = sys.executable,
    test_timeout_s: float = 300.0,
    keep_addopts: bool = False,
    memory_overrides: dict | None = None,
    echo: Callable[[str], None] = print,
) -> list[RunResult]:
    """Run every (case, condition) not already in workdir/results.jsonl; returns all results (old + new)."""
    unknown = set(conditions) - set(CONDITIONS)
    if unknown:
        raise ValueError(f"unknown conditions {sorted(unknown)} (choose from {CONDITIONS})")
    if "bm25-matched" in conditions and "zt" not in conditions:
        raise ValueError("bm25-matched needs the zt condition (it reuses zt's token count)")
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    results_path = workdir / "results.jsonl"
    done = load_results(results_path)
    have = {(r.case, r.condition) for r in done}
    served = served_ctx or client.context_size() or 16384
    limit = served - max_new_tokens - 128
    rc = router_config or RouterConfig()
    count = _count_fn(client)
    wt = Worktree(repo, workdir / "worktree")
    mem_cfg = MemoryConfig(storage_dir=workdir / "zt_store", repo_root=wt.path, rank=rank, **(memory_overrides or {}))
    out = list(done)
    with NewtonianMemoryManager(mem_cfg, encoder) as mgr:
        for case in sorted(cases, key=lambda c: c.date):
            todo = [c for c in conditions if (case.id, c) not in have]
            if not todo:
                continue
            wt.checkout(case.base)
            t = time.perf_counter()
            report = mgr.sync()  # incremental: only symbols changed since the previous case are re-embedded
            echo(f"[case {case.id}] {case.instruction[:80]!r} | sync chunks +{report.ingest.chunks_added} ~{report.ingest.chunks_updated} in {time.perf_counter() - t:.1f}s")
            instruction = case.instruction + (f"\n\nFailing test output:\n```\n{case.failure_excerpt}\n```" if with_failure_log and case.failure_excerpt else "")
            files = discover(mem_cfg)
            ranked = bm25_rank_files(wt.path, files, instruction)
            zt_tokens: int | None = next((r.context_tokens for r in done if r.case == case.id and r.condition == "zt"), None)
            for cond in [c for c in CONDITIONS if c in todo]:  # fixed order: zt before bm25-matched
                try:
                    if cond == "zt":
                        router = GeometricRouter(mgr, rc)
                        routed, messages, exact, _ = fit_context(router, client, instruction, served_ctx=served,
                                                                 max_new_tokens=max_new_tokens, budget_cap=rc.token_budget)
                        ctx_tokens = exact if exact is not None else count(messages[0]["content"] + "\n" + messages[1]["content"])
                        ctx_files = sorted({b.path for b in routed.blocks if b.path})
                        zt_tokens = ctx_tokens
                    else:
                        budget = limit if cond == "bm25-files" else (zt_tokens or limit)
                        messages, ctx_tokens, ctx_files = file_pack_messages(wt.path, ranked, instruction, budget=budget, count=count)
                    res = _attempt(case, cond, messages, ctx_tokens, ctx_files, wt, client, max_new_tokens=max_new_tokens,
                                   temperature=temperature, retries=retries, python=python, test_timeout_s=test_timeout_s,
                                   keep_addopts=keep_addopts)
                except Exception as exc:  # noqa: BLE001 - one broken case must not stop a night-long run
                    if getattr(exc, "stop_run", False):  # call limit reached: stop; a resumed run redoes this case
                        raise
                    res = RunResult(case.id, cond, "error", False, 0, 0, 0, 0.0, 0, error=f"{type(exc).__name__}: {exc}")
                finally:
                    wt.checkout(case.base)
                echo(f"  {cond:13s} {'RESOLVED' if res.resolved else res.status:12s} ctx {res.context_tokens:6d} tok | "
                     f"llm {res.llm_seconds:6.1f}s | attempts {res.attempts} | gold in ctx {res.gold_in_context:.0%}"
                     + (f" | {res.error[:80]}" if res.error else ""))
                with results_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(asdict(res)) + "\n")
                out.append(res)
    wt.remove()
    return out


def _attempt(case: BugCase, cond: str, messages, ctx_tokens: int, ctx_files: list[str], wt: Worktree,  # noqa: ANN001
             client: ComputeEngineClient, *, max_new_tokens: int, temperature: float, retries: int, python: str,
             test_timeout_s: float, keep_addopts: bool) -> RunResult:
    got = request_edits(client, messages, wt.path, max_new_tokens=max_new_tokens, temperature=temperature, retries=retries)
    gold = set(case.source_files)
    res = RunResult(case.id, cond, "no_edits", False, int(ctx_tokens),
                    sum(r.prompt_tokens or 0 for r in got.llm), sum(r.completion_tokens or 0 for r in got.llm),
                    round(sum(r.seconds for r in got.llm), 2), got.attempts, ctx_files,
                    len(gold & set(ctx_files)) / len(gold) if gold else 0.0)
    if not got.ok:
        res.status = "edit_failed" if got.blocks else "no_edits"
        res.error = (got.errors[-1] if got.errors else "")[:300]
        return res
    assert got.plan is not None
    for rel, text in got.plan.new_contents.items():
        target = wt.path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(text.encode("utf-8"))
    res.patched_files = sorted(got.plan.new_contents)
    wt.take_files(case.fix, case.test_files)  # grading tests overwrite any edit the model made to them
    code, output = run_tests(wt.path, case.test_files, python=python, timeout_s=test_timeout_s, keep_addopts=keep_addopts)
    res.test_exit = code
    res.resolved = code == 0
    res.status = "resolved" if res.resolved else "tests_failed"
    if not res.resolved:
        res.error = _last_line(output)
    return res


def load_results(path: Path) -> list[RunResult]:
    if not Path(path).exists():
        return []
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(RunResult(**json.loads(line)))
    return out


# ------------------------------------------------------------------------------------------
# report
# ------------------------------------------------------------------------------------------
def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value on discordant pairs (b: only A solved, c: only B solved)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)


def summarize(results: Iterable[RunResult], conditions: Sequence[str] = CONDITIONS) -> str:
    by: dict[str, dict[str, RunResult]] = {}
    for r in results:
        by.setdefault(r.condition, {})[r.case] = r
    conds = [c for c in conditions if c in by]
    cases = sorted(set.intersection(*(set(by[c]) for c in conds))) if conds else []
    if not cases:
        return "no complete cases yet"
    n = len(cases)
    lines = [f"Cases with every condition: {n}", "",
             "| condition | resolved | mean ctx tokens | mean LLM s | mean attempts | gold file in ctx | patch unusable | errors |",
             "|---|---|---|---|---|---|---|---|"]
    for c in conds:
        rs = [by[c][k] for k in cases]
        solved = sum(r.resolved for r in rs)
        lines.append(f"| {c} | {solved}/{n} ({solved / n:.0%}) | {sum(r.context_tokens for r in rs) / n:.0f} | "
                     f"{sum(r.llm_seconds for r in rs) / n:.1f} | {sum(r.attempts for r in rs) / n:.2f} | "
                     f"{sum(r.gold_in_context for r in rs) / n:.0%} | {sum(r.status in ('no_edits', 'edit_failed') for r in rs)} | "
                     f"{sum(r.status == 'error' for r in rs)} |")
    lines += ["", "| A vs B | only A | only B | both | neither | exact McNemar p | ctx tokens A/B |", "|---|---|---|---|---|---|---|"]
    for i, a in enumerate(conds):
        for bname in conds[i + 1:]:
            ra, rb = [by[a][k] for k in cases], [by[bname][k] for k in cases]
            oa = sum(x.resolved and not y.resolved for x, y in zip(ra, rb))
            ob = sum(y.resolved and not x.resolved for x, y in zip(ra, rb))
            both = sum(x.resolved and y.resolved for x, y in zip(ra, rb))
            ta, tb = sum(x.context_tokens for x in ra), sum(y.context_tokens for y in rb)
            ratio = f"{ta / tb:.2f}" if tb else "-"
            lines.append(f"| {a} vs {bname} | {oa} | {ob} | {both} | {n - oa - ob - both} | {mcnemar_exact(oa, ob):.3f} | {ratio} |")
    lines += ["", "With few cases the p-values cannot be small: read the discordant pairs and the per-case table."]
    lines += ["", "| case | " + " | ".join(conds) + " |", "|---|" + "---|" * len(conds)]
    for k in cases:
        lines.append(f"| {k} | " + " | ".join(
            f"{'PASS' if by[c][k].resolved else 'fail: ' + by[c][k].status} ({by[c][k].context_tokens})" for c in conds) + " |")
    return "\n".join(lines)
