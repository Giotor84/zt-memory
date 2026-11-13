"""Ablation benchmark by replaying a repository's git history.

For every commit c on the first-parent chain (oldest -> newest), with the store synced to c's parent:
  query  = c's commit message (subject + first body paragraph, trailers and PR refs stripped)
  truth  = files c modified, deleted or renamed that existed in the parent (added files cannot be retrieved)
Every router configuration ranks files on the *same* store state (paired comparison, no reinforcement),
then c is checked out and synced, so the Newtonian state evolves exactly as it would have in real use.

Reported per configuration: hit@k (any true file in the top-k), recall@k (share of true files), MRR.
Baseline "recency" ranks files by current energy alone, ignoring the query: it isolates the physics prior.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Callable, Mapping, Sequence

from .config import MemoryConfig, NodeKind, RouterConfig
from .discovery import is_indexable
from .encoders import HiddenStateEncoder
from .manager import NewtonianMemoryManager
from .router import GeometricRouter

_TRAILER = re.compile(r"^(signed-off-by|co-authored-by|reviewed-by|change-id|cc):", re.IGNORECASE)
_PR_REF = re.compile(r"\s*\((?:#|gh-)\d+\)|\s*#\d+\b")


def _rmtree(path: Path) -> None:
    """rmtree that also removes read-only files (git object files are read-only on Windows)."""
    import os
    import stat

    def onerror(func, p, _exc):  # noqa: ANN001
        os.chmod(p, stat.S_IWRITE)
        func(p)

    if path.exists():
        shutil.rmtree(path, onerror=onerror)


def _git(repo: Path, *args: str, timeout: float = 300.0) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True,
                          timeout=timeout, encoding="utf-8", errors="replace").stdout


def clean_message(subject: str, body: str, max_chars: int = 400) -> str:
    paragraph = body.strip().split("\n\n", 1)[0] if body.strip() else ""
    lines = [ln for ln in paragraph.splitlines() if not _TRAILER.match(ln.strip())]
    text = " ".join([subject.strip(), *(ln.strip() for ln in lines)]).strip()
    return _PR_REF.sub("", text)[:max_chars].strip()


@dataclass(frozen=True)
class CommitCase:
    sha: str
    parent: str
    query: str
    truth: tuple[str, ...]
    is_merge: bool
    n_changed: int


def list_commits(repo: Path, n: int, config: MemoryConfig, *, max_files: int = 20, min_words: int = 3) -> list[CommitCase]:
    """Last n first-parent commits, oldest first. Merges are kept (to be applied) but never queried."""
    raw = _git(repo, "log", "--first-parent", f"-n{n}", "--format=%H%x00%P%x00%s%x00%b%x1e")
    cases: list[CommitCase] = []
    for rec in reversed([r for r in raw.split("\x1e") if r.strip()]):
        sha, parents, subject, body = (rec.strip("\n").split("\x00") + ["", "", "", ""])[:4]
        plist = parents.split()
        if not plist:
            continue  # root commit: nothing to diff against
        status = _git(repo, "diff", "--name-status", "--no-renames", plist[0], sha)
        truth: list[str] = []
        changed = 0
        for line in status.splitlines():
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            changed += 1
            if parts[0][0] in "MD" and is_indexable(parts[1], config):
                truth.append(parts[1])
        query = clean_message(subject, body)
        is_merge = len(plist) > 1
        usable = not is_merge and truth and changed <= max_files and len(query.split()) >= min_words
        cases.append(CommitCase(sha, plist[0], query, tuple(sorted(set(truth))) if usable else (), is_merge, changed))
    return cases


@dataclass
class ConfigScore:
    name: str
    cases: int = 0
    hits: dict[int, int] = field(default_factory=dict)
    recall_sum: dict[int, float] = field(default_factory=dict)
    rr_sum: float = 0.0
    per_case_rr: list[float] = field(default_factory=list)
    per_case_recall10: list[float] = field(default_factory=list)

    def add(self, ranked: Sequence[str], truth: Sequence[str], ks: Sequence[int]) -> None:
        self.cases += 1
        gt = set(truth)
        first = next((i for i, p in enumerate(ranked) if p in gt), None)
        rr = 0.0 if first is None else 1.0 / (first + 1)
        self.rr_sum += rr
        self.per_case_rr.append(rr)
        self.per_case_recall10.append(len(set(ranked[:10]) & gt) / len(gt))
        for k in ks:
            top = set(ranked[:k])
            self.hits[k] = self.hits.get(k, 0) + int(bool(top & gt))
            self.recall_sum[k] = self.recall_sum.get(k, 0.0) + len(top & gt) / len(gt)

    def summary(self, ks: Sequence[int]) -> dict[str, float]:
        n = max(self.cases, 1)
        out: dict[str, float] = {"cases": self.cases, "mrr": round(self.rr_sum / n, 4)}
        for k in ks:
            out[f"hit@{k}"] = round(self.hits.get(k, 0) / n, 4)
            out[f"recall@{k}"] = round(self.recall_sum.get(k, 0.0) / n, 4)
        return out


@dataclass
class CutoffScore:
    alpha: float
    cases: int = 0
    coverage_sum: float = 0.0  # share of true files present in the packed context
    tokens_sum: int = 0
    blocks_sum: int = 0

    def summary(self) -> dict[str, float]:
        n = max(self.cases, 1)
        return {"alpha": self.alpha, "coverage": round(self.coverage_sum / n, 4),
                "avg_tokens": round(self.tokens_sum / n, 1), "avg_blocks": round(self.blocks_sum / n, 2)}


def paired_bootstrap(a: Sequence[float], b: Sequence[float], *, iters: int = 5000, seed: int = 0) -> dict[str, float]:
    """Mean of (b - a) over the same cases with a 95% bootstrap CI. CI excluding 0 = a real difference."""
    import random

    diffs = [y - x for x, y in zip(a, b)]
    if not diffs:
        return {"diff": 0.0, "lo": 0.0, "hi": 0.0}
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(iters))
    return {"diff": round(sum(diffs) / n, 4), "lo": round(means[int(0.025 * iters)], 4), "hi": round(means[int(0.975 * iters)], 4)}


COMPARISONS = (
    ("A_hybrid_flat", "L_lexical_only", "hybrid vs lexical-only"),
    ("P_physics", "A_hybrid_flat", "physics prior (w_E=1.5)"),
    ("R_physics_rrf", "A_hybrid_flat", "physics as RRF list (w=1.0)"),
    ("F_defaults", "A_hybrid_flat", "shipped defaults vs flat hybrid"),
)


@dataclass
class ReplayReport:
    repo: str
    commits_scanned: int
    cases: int
    seconds: float
    initial_sync_seconds: float
    ks: list[int]
    configs: dict[str, dict[str, float]]
    cutoff: list[dict[str, float]]
    comparisons: dict[str, dict[str, dict[str, float]]] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def default_configs(base: RouterConfig | None = None) -> dict[str, RouterConfig]:
    """L, G: single signals. A: hybrid, flat (no physics). P: + energy prior. R: energy as a third RRF list.
    F: the shipped defaults. Every ablation knob is set explicitly so changing a default never changes L-R."""
    b = replace(base or RouterConfig(), reinforce=False, cutoff_alpha=0.0)
    flat = dict(energy_weight=0.0, recency_weight=0.0)
    return {
        "L_lexical_only": replace(b, geometric_weight=0.0, **flat),
        "G_geometric_only": replace(b, lexical_weight=0.0, **flat),
        "A_hybrid_flat": replace(b, **flat),
        "P_physics": replace(b, energy_weight=1.5, recency_weight=0.0),
        "R_physics_rrf": replace(b, energy_weight=0.0, recency_weight=1.0),
        "F_defaults": replace(RouterConfig(), reinforce=False, cutoff_alpha=0.0),
    }


def _recency_ranking(mgr: NewtonianMemoryManager, limit: int = 500) -> list[str]:
    hot = mgr.hottest(limit, frozenset({NodeKind.CHUNK}))
    recs = mgr.records(i for i, _ in hot)
    seen: dict[str, None] = {}
    for i, _ in hot:
        rec = recs.get(i)
        if rec is not None and rec.path:
            seen.setdefault(rec.path, None)
    return list(seen)


def replay(
    source: str,
    workdir: Path,
    encoder: HiddenStateEncoder,
    *,
    commits: int = 200,
    configs: Mapping[str, RouterConfig] | None = None,
    ks: Sequence[int] = (1, 5, 10),
    cutoff_alphas: Sequence[float] = (0.0, 0.25, 0.4, 0.55),
    cutoff_base: str = "F_defaults",
    cutoff_budget: int = 12_000,
    max_files: int = 20,
    memory_overrides: Mapping[str, object] | None = None,
    progress: Callable[[int, int, CommitCase], None] | None = None,
    projector_path: Path | None = None,
) -> ReplayReport:
    workdir = Path(workdir)
    clone = workdir / "repo"
    store = workdir / "store"
    _rmtree(clone)
    _rmtree(store)
    workdir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--quiet", "--no-local" if Path(source).exists() else "--single-branch", source, str(clone)],
                   check=True, capture_output=True, timeout=1800)
    mem = MemoryConfig(storage_dir=store, repo_root=clone, **dict(memory_overrides or {}))
    cases = list_commits(clone, commits, mem, max_files=max_files)
    if not cases:
        raise ValueError("no usable commits")
    cfgs = dict(configs or default_configs())
    scores = {name: ConfigScore(name) for name in [*cfgs, "R_recency_only"]}
    cut = [CutoffScore(a) for a in cutoff_alphas]
    t0 = time.perf_counter()
    _git(clone, "checkout", "--quiet", "--detach", cases[0].parent)
    projector = None
    if projector_path is not None:
        from .projector import SubspaceProjector

        projector = SubspaceProjector.load(Path(projector_path))
    with NewtonianMemoryManager(mem, encoder, projector=projector) as mgr:
        mgr.sync()
        initial = time.perf_counter() - t0
        queried = 0
        for i, case in enumerate(cases):
            if case.truth:
                known = set(mgr.file_paths())
                truth = [p for p in case.truth if p in known]
                if truth:
                    queried += 1
                    for name, rc in cfgs.items():
                        ranked = [p for p, _ in GeometricRouter(mgr, rc).ranked_files(case.query)]
                        scores[name].add(ranked, truth, ks)
                    scores["R_recency_only"].add(_recency_ranking(mgr), truth, ks)
                    base = cfgs.get(cutoff_base)
                    if base is not None:
                        for c in cut:
                            ctx = GeometricRouter(mgr, replace(base, cutoff_alpha=c.alpha)).route_to_compute_engine(
                                case.query, token_budget=cutoff_budget)
                            paths = {b.path for b in ctx.blocks if b.path}
                            c.cases += 1
                            c.coverage_sum += len(paths & set(truth)) / len(truth)
                            c.tokens_sum += ctx.token_estimate
                            c.blocks_sum += len(ctx.blocks)
            _git(clone, "checkout", "--quiet", "--detach", case.sha)
            mgr.sync()  # the commit's edits drive surprise, momentum and friction
            if progress:
                progress(i + 1, len(cases), case)
    comparisons: dict[str, dict[str, dict[str, float]]] = {}
    for new, old, label in COMPARISONS:
        if new in scores and old in scores:
            comparisons[f"{label}: {new} - {old}"] = {
                "mrr": paired_bootstrap(scores[old].per_case_rr, scores[new].per_case_rr),
                "recall@10": paired_bootstrap(scores[old].per_case_recall10, scores[new].per_case_recall10),
            }
    return ReplayReport(
        source, len(cases), queried, time.perf_counter() - t0, initial, list(ks),
        {name: sc.summary(ks) for name, sc in scores.items()}, [c.summary() for c in cut], comparisons,
    )
