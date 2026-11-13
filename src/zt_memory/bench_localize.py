"""Localization benchmark: does the context contain the code that the real fix changed?

No LLM involved. Cases are bug-fix commits from a repository's history (any language): the commit message is the
question, the base commit is the code, and the lines the fix changed (old side of the diff) are the target.

Methods (all on the same base commit; the memory follows the history, reinforcement off so methods never interact):

    zt            default router: hybrid lexical + embedding + edit energy, symbol spans, dynamic cut
    bm25-chunks   the same router with only its lexical list (FTS5/BM25 over symbols): the honest baseline
    bm25-chunks-matched  bm25-chunks without the dynamic cut, packed up to the tokens zt used on the same case:
                  separates "better selection" from "more context"
    bm25-files    BM25 over whole files, files packed in rank order until the budget is full (last one cut)
    grep          files ranked by how many distinct query words they contain (ranking only, no packing)

Per case and method: file ranking (hit@1, hit@5, recall@10, MRR) and, for the packed contexts, whether the changed
lines are inside the context and how many tokens it uses. Tokens are estimated (chars / 3.2, the router's rule) for
every method alike.
"""
from __future__ import annotations

import json
import math
import re
import subprocess
import time
from collections import Counter
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path, PurePosixPath
from typing import Callable, Iterable, Sequence

from .bench_bugfix import BUG_WORDS, Worktree, bm25_rank_files, mcnemar_exact
from .config import MemoryConfig, NodeKind, RouterConfig
from .discovery import discover
from .encoders import HiddenStateEncoder
from .lexical import LEXICAL_STOPWORDS, lexical_tokens
from .manager import NewtonianMemoryManager
from .replay import _git, clean_message, paired_bootstrap
from .router import GeometricRouter

METHODS = ("zt", "bm25-chunks", "bm25-chunks-matched", "bm25-files", "grep")
PACKED = ("zt", "bm25-chunks", "bm25-chunks-matched", "bm25-files")
CODE_SUFFIXES = frozenset({".py", ".go", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".rs", ".java", ".kt", ".c",
                           ".h", ".cc", ".cpp", ".hpp", ".cs", ".rb", ".php", ".swift"})
_TEST = re.compile(r"(^|/)(tests?|testing|__tests__|spec|specs|test_data|testdata|fixtures?)/|(^|/)test_[^/]*$"
                   r"|_test\.(py|go)$|\.(test|spec)\.[cm]?[jt]sx?$|(^|/)conftest\.py$")
_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+\d+(?:,\d+)? @@", re.MULTILINE)
CHARS_PER_TOKEN = 3.2
TEMPLATE_TOKENS = 256


def is_test_path(rel: str) -> bool:
    return bool(_TEST.search(rel))


def estimate_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / CHARS_PER_TOKEN))


@dataclass(frozen=True)
class LocCase:
    repo: str
    id: str
    fix: str
    base: str
    date: int
    query: str
    gold: dict[str, list[int]]  # file -> changed line numbers on the base side
    has_tests: bool

    @classmethod
    def from_dict(cls, d: dict) -> "LocCase":
        return cls(d["repo"], d["id"], d["fix"], d["base"], int(d["date"]), d["query"],
                   {k: list(v) for k, v in d["gold"].items()}, bool(d["has_tests"]))


def changed_lines(repo: Path, base: str, fix: str, rel: str) -> list[int]:
    """Base-side line numbers touched by the fix; a pure insertion marks the line it follows."""
    lines: set[int] = set()
    for m in _HUNK.finditer(_git(repo, "diff", "-U0", "--no-renames", base, fix, "--", rel)):
        start, count = int(m.group(1)), int(m.group(2) or "1")
        if count == 0:
            lines.update({max(start, 1), start + 1})
        else:
            lines.update(range(start, start + count))
    return sorted(lines)


def mine_localization_cases(repo: Path, *, max_commits: int = 1500, limit: int = 25, max_files: int = 3,
                            max_loc: int = 100, min_words: int = 3) -> list[LocCase]:
    """Bug-fix commits that modify 1..max_files existing non-test code files; newest first, returned oldest first."""
    name = Path(repo).name  # as given: a symlink or folder name the user chose, not the resolved target
    repo = Path(repo).resolve()
    raw = _git(repo, "log", "--no-merges", f"-n{max_commits}", "--format=%H%x00%P%x00%ct%x00%s%x00%b%x1e")
    cases: list[LocCase] = []
    seen: set[str] = set()
    for rec in [r for r in raw.split("\x1e") if r.strip()]:
        if len(cases) >= limit:
            break
        sha, parents, ct, subject, body = (rec.strip("\n").split("\x00") + [""] * 5)[:5]
        if len(parents.split()) != 1 or not BUG_WORDS.search(subject):
            continue
        query = clean_message(subject, body, 600)
        if len(query.split()) < min_words:
            continue
        gold_files, tests, loc = [], False, 0
        for line in _git(repo, "diff", "--numstat", "--no-renames", parents, sha).splitlines():
            parts = line.split("\t")
            if len(parts) != 3:
                continue
            added, deleted, rel = parts
            if PurePosixPath(rel).suffix.lower() not in CODE_SUFFIXES:
                continue
            if is_test_path(rel):
                tests = True
                continue
            if deleted == "0" and _added_file(repo, parents, rel):
                continue  # a file created by the fix cannot be found in the base
            gold_files.append(rel)
            loc += (int(added) if added.isdigit() else 0) + (int(deleted) if deleted.isdigit() else 0)
        if not gold_files or len(gold_files) > max_files or loc > max_loc:
            continue
        gold = {rel: changed_lines(repo, parents, sha, rel) for rel in gold_files}
        key = json.dumps(gold, sort_keys=True) + query[:80]
        if key in seen:
            continue
        seen.add(key)
        cases.append(LocCase(name, sha[:10], sha, parents, int(ct), query, gold, tests))
    return sorted(cases, key=lambda c: c.date)


def _added_file(repo: Path, base: str, rel: str) -> bool:
    try:
        _git(repo, "cat-file", "-e", f"{base}:{rel}")
        return False
    except subprocess.CalledProcessError:
        return True


# ------------------------------------------------------------------------------------------
# baselines
# ------------------------------------------------------------------------------------------
def _terms(text: str) -> list[str]:
    return [t for t in lexical_tokens(text) if t not in LEXICAL_STOPWORDS and len(t) > 1]


def grep_rank_files(root: Path, files: Sequence[str], query: str) -> list[tuple[str, float]]:
    """What a quick grep gives: files with more distinct query words first (ties: more occurrences)."""
    q = set(_terms(query))
    out = []
    for rel in files:
        try:
            counts = Counter(t for t in _terms((Path(root) / rel).read_text(encoding="utf-8", errors="replace")) if t in q)
        except OSError:
            continue
        if counts:
            out.append((rel, len(counts) + sum(counts.values()) / 1e6))
    return sorted(out, key=lambda t: (-t[1], t[0]))


def pack_whole_files(root: Path, ranked: Sequence[tuple[str, float]], budget: int) -> tuple[dict[str, tuple[int, int]], int]:
    """Whole files in rank order until `budget`; the first one that does not fit is cut to fill the rest."""
    used = TEMPLATE_TOKENS
    spans: dict[str, tuple[int, int]] = {}
    for rel, score in ranked:
        if score <= 0:
            break
        text = (Path(root) / rel).read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        need = estimate_tokens(text) + 8
        if used + need <= budget:
            spans[rel] = (1, max(1, len(lines)))
            used += need
            continue
        room = budget - used
        if room > 64:
            keep = max(1, int(len(lines) * room / need))
            spans[rel] = (1, keep)
            used += estimate_tokens("\n".join(lines[:keep])) + 8
        break
    return spans, used


# ------------------------------------------------------------------------------------------
# scoring
# ------------------------------------------------------------------------------------------
@dataclass
class LocResult:
    repo: str
    case: str
    method: str
    ranked_gold: list[int | None]  # 1-based file rank of each gold file (None = not ranked)
    hit1: bool
    hit5: bool
    recall10: float
    mrr: float
    tokens: int | None = None
    file_in_ctx: float | None = None  # gold files with at least one span in the context
    lines_in_ctx: float | None = None  # changed lines inside the context
    spot_in_ctx: bool | None = None  # every gold file has at least one changed line in the context
    ctx_files: int | None = None
    seconds: float = 0.0


def _ranking_scores(ranked_files: Sequence[str], gold: Iterable[str]) -> tuple[list[int | None], bool, bool, float, float]:
    pos = {rel: i + 1 for i, rel in enumerate(ranked_files)}
    ranks = [pos.get(g) for g in gold]
    hits = [r for r in ranks if r is not None]
    first = min(hits) if hits else None
    n = len(ranks)
    return (ranks, first == 1, first is not None and first <= 5,
            sum(1 for r in hits if r <= 10) / n if n else 0.0, 1.0 / first if first else 0.0)


def _coverage(spans: dict[str, list[tuple[int, int]]], gold: dict[str, list[int]]) -> tuple[float, float, bool]:
    files_hit, lines_hit, lines_total, spot = 0, 0, 0, True
    for rel, lines in gold.items():
        ranges = spans.get(rel, [])
        if ranges:
            files_hit += 1
        inside = [ln for ln in lines if any(a <= ln <= b for a, b in ranges)]
        lines_hit += len(inside)
        lines_total += len(lines)
        spot = spot and bool(inside)
    return files_hit / len(gold), (lines_hit / lines_total if lines_total else 0.0), spot


def _routed_spans(ctx) -> dict[str, list[tuple[int, int]]]:  # noqa: ANN001
    spans: dict[str, list[tuple[int, int]]] = {}
    for b in ctx.blocks:
        if b.path and b.start_line is not None and b.kind in (NodeKind.CHUNK, NodeKind.FILE):
            spans.setdefault(b.path, []).append((int(b.start_line), int(b.end_line or b.start_line)))
    return spans


def router_configs(base: RouterConfig | None = None) -> dict[str, RouterConfig]:
    b = replace(base or RouterConfig(), reinforce=False)
    return {"zt": b, "bm25-chunks": replace(b, geometric_weight=0.0, energy_weight=0.0, recency_weight=0.0)}


def run_localization(
    repo: Path,
    cases: Sequence[LocCase],
    workdir: Path,
    encoder: HiddenStateEncoder,
    *,
    rank: int = 64,
    budget: int = 12_000,
    memory_overrides: dict | None = None,
    echo: Callable[[str], None] = print,
) -> list[LocResult]:
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    out_path = workdir / "loc_results.jsonl"
    done = load_loc_results(out_path)
    have = {(r.case, r.method) for r in done}
    results = list(done)
    wt = Worktree(repo, workdir / "worktree")
    cfg = MemoryConfig(storage_dir=workdir / "zt_store", repo_root=wt.path, rank=rank, **(memory_overrides or {}))
    configs = router_configs()
    try:
        with NewtonianMemoryManager(cfg, encoder) as mgr:
            for case in sorted(cases, key=lambda c: c.date):
                todo = [m for m in METHODS if (case.id, m) not in have]
                if not todo:
                    continue
                wt.checkout(case.base)
                t = time.perf_counter()
                mgr.sync()
                sync_s = time.perf_counter() - t
                files = discover(cfg)
                rows = []
                zt_tokens = next((r.tokens for r in results if r.repo == case.repo and r.case == case.id and r.method == "zt"), None)
                for method in todo:
                    t = time.perf_counter()
                    if method == "bm25-chunks-matched":
                        router = GeometricRouter(mgr, replace(configs["bm25-chunks"], cutoff_alpha=0.0))
                        ranked = [p for p, s in router.ranked_files(case.query) if s > 0]
                        ctx = router.route_to_compute_engine(case.query, token_budget=int(zt_tokens or budget))
                        spans, tokens = _routed_spans(ctx), ctx.token_estimate
                    elif method in configs:
                        router = GeometricRouter(mgr, configs[method])
                        ranked = [p for p, s in router.ranked_files(case.query) if s > 0]
                        ctx = router.route_to_compute_engine(case.query, token_budget=budget)
                        spans, tokens = _routed_spans(ctx), ctx.token_estimate
                    elif method == "bm25-files":
                        scored = bm25_rank_files(wt.path, files, case.query)
                        ranked = [p for p, s in scored if s > 0]
                        whole, tokens = pack_whole_files(wt.path, scored, budget)
                        spans = {rel: [span] for rel, span in whole.items()}
                    else:  # grep: ranking only
                        ranked = [p for p, _ in grep_rank_files(wt.path, files, case.query)]
                        spans, tokens = None, None
                    ranks, h1, h5, r10, mrr = _ranking_scores(ranked, case.gold)
                    res = LocResult(case.repo, case.id, method, ranks, h1, h5, r10, mrr, seconds=round(time.perf_counter() - t, 3))
                    if spans is not None:
                        res.tokens = int(tokens)
                        res.file_in_ctx, res.lines_in_ctx, res.spot_in_ctx = _coverage(spans, case.gold)
                        res.ctx_files = len(spans)
                    rows.append(res)
                    if method == "zt":
                        zt_tokens = res.tokens
                with out_path.open("a", encoding="utf-8") as fh:
                    for r in rows:
                        fh.write(json.dumps(asdict(r)) + "\n")
                results.extend(rows)
                zt = next((r for r in rows if r.method == "zt"), None)
                echo(f"[{case.repo} {case.id}] sync {sync_s:.1f}s | " + " | ".join(
                    f"{r.method} rank {r.ranked_gold}" + (f" spot {'yes' if r.spot_in_ctx else 'no'} {r.tokens}t" if r.tokens else "")
                    for r in rows) if zt else f"[{case.repo} {case.id}] done")
    finally:
        wt.remove()
    return results


def load_loc_results(path: Path) -> list[LocResult]:
    if not Path(path).exists():
        return []
    return [LocResult(**json.loads(line)) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


# ------------------------------------------------------------------------------------------
# report
# ------------------------------------------------------------------------------------------
def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def summarize_localization(results: Sequence[LocResult], *, title: str = "all repositories") -> str:
    by: dict[tuple[str, str], LocResult] = {(r.repo + "/" + r.case, r.method): r for r in results}
    cases = sorted({k for k, _ in by if all((k, m) in by for m in METHODS)})
    n = len(cases)
    if not n:
        return "no complete cases"
    lines = [f"### {title}: {n} cases", "",
             "| method | right file 1st | right file in top 5 | recall@10 | MRR | right file in context | changed lines in context | fix spot in context | mean tokens |",
             "|---|---|---|---|---|---|---|---|---|"]
    for m in METHODS:
        rs = [by[(k, m)] for k in cases]
        packed = m in PACKED
        lines.append(
            f"| {m} | {_mean([r.hit1 for r in rs]):.0%} | {_mean([r.hit5 for r in rs]):.0%} | {_mean([r.recall10 for r in rs]):.2f} | "
            f"{_mean([r.mrr for r in rs]):.3f} | "
            + (f"{_mean([r.file_in_ctx for r in rs]):.0%} | {_mean([r.lines_in_ctx for r in rs]):.0%} | "
               f"{_mean([float(r.spot_in_ctx) for r in rs]):.0%} | {_mean([r.tokens for r in rs]):.0f} |" if packed else "- | - | - | - |"))
    lines += ["", "| zt vs | MRR diff [95% CI] | fix spot: only zt / only other | McNemar p | changed lines diff [95% CI] | tokens zt / other |",
              "|---|---|---|---|---|---|"]
    zt = [by[(k, "zt")] for k in cases]
    for m in METHODS[1:]:
        other = [by[(k, m)] for k in cases]
        d_mrr = paired_bootstrap([b.mrr for b in other], [a.mrr for a in zt])  # zt - other
        row = f"| {m} | {d_mrr['diff']:+.3f} [{d_mrr['lo']:+.3f}, {d_mrr['hi']:+.3f}] | "
        if m in PACKED:
            only_zt = sum(a.spot_in_ctx and not b.spot_in_ctx for a, b in zip(zt, other))
            only_o = sum(b.spot_in_ctx and not a.spot_in_ctx for a, b in zip(zt, other))
            d_lines = paired_bootstrap([b.lines_in_ctx for b in other], [a.lines_in_ctx for a in zt])
            ratio = _mean([a.tokens for a in zt]) / max(_mean([b.tokens for b in other]), 1)
            row += (f"{only_zt} / {only_o} | {mcnemar_exact(only_zt, only_o):.4f} | "
                    f"{d_lines['diff']:+.2f} [{d_lines['lo']:+.2f}, {d_lines['hi']:+.2f}] | {ratio:.2f} |")
        else:
            row += "- | - | - | - |"
        lines.append(row)
    return "\n".join(lines)


def per_repo_tables(results: Sequence[LocResult]) -> str:
    repos = sorted({r.repo for r in results})
    return "\n\n".join(summarize_localization([r for r in results if r.repo == repo], title=repo) for repo in repos)
