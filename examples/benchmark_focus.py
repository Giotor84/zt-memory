"""Phase 1 of the focused context: anti-clone order and bodies only where they matter, measured offline.

No LLM call. For every case the repository is checked out at the parent of the fix, indexed, and routed with each
variant; the context is then compared with the lines the real fix changed:
  body  - at least one changed line, for every changed file, is inside a code body sent whole;
  map   - the same, counting also the signature-only lines (the model sees where to look, not the code).

Variants (all on top of today's zt with the same encoder and budget):
  A  today's zt
  B  anti-clone only: stub filter + near-duplicates + MMR order, everything rendered whole
  C  B + focus (F=3, ratio 0.6), no reinvestment
  D  B + focus (F=5, ratio 0.6), no reinvestment
  E3 / E5  C / D + up to 20 extra signatures with the saved tokens (the report keeps the one of the best of C, D)

    python examples/benchmark_focus.py --workdir DIR --repos-dir DIR \\
        --loc examples/data/loc_cases_it.json --loc-en EN_CASES.json --traceback A/cases.json B/cases.json ...
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from zt_memory.bench_bugfix import Worktree, load_cases, mcnemar_exact
from zt_memory.bench_localize import LocCase, _added_file, _coverage, changed_lines
from zt_memory.config import MemoryConfig, NodeKind, RouterConfig
from zt_memory.encoders import build_encoder
from zt_memory.manager import NewtonianMemoryManager
from zt_memory.router import GeometricRouter

LAMBDA = 0.3
BASE = RouterConfig(reinforce=False)
VARIANTS = {
    "A": BASE,
    "B": replace(BASE, dedup=True, diversity_lambda=LAMBDA),
    "C": replace(BASE, dedup=True, diversity_lambda=LAMBDA, focus_blocks=3, focus_ratio=0.6),
    "D": replace(BASE, dedup=True, diversity_lambda=LAMBDA, focus_blocks=5, focus_ratio=0.6),
    "E3": replace(BASE, dedup=True, diversity_lambda=LAMBDA, focus_blocks=3, focus_ratio=0.6, signature_extra=20),
    "E5": replace(BASE, dedup=True, diversity_lambda=LAMBDA, focus_blocks=5, focus_ratio=0.6, signature_extra=20),
    # same focus without anti-clone: isolates what MMR + filters give at equal F
    "C0": replace(BASE, focus_blocks=3, focus_ratio=0.6),
    "D0": replace(BASE, focus_blocks=5, focus_ratio=0.6),
    # curve beyond the planned grid: how wide must the focus be to keep today's recall?
    "F10": replace(BASE, dedup=True, diversity_lambda=LAMBDA, focus_blocks=10, focus_ratio=0.6),
    "F15": replace(BASE, dedup=True, diversity_lambda=LAMBDA, focus_blocks=15, focus_ratio=0.6),
}
MAX_QUERY_CHARS = 8000


def spans(ctx, *, body_only: bool) -> dict[str, list[tuple[int, int]]]:  # noqa: ANN001
    out: dict[str, list[tuple[int, int]]] = {}
    for b in ctx.blocks:
        if b.path and b.start_line is not None and b.kind in (NodeKind.CHUNK, NodeKind.FILE) and (b.full or not body_only):
            out.setdefault(b.path, []).append((int(b.start_line), int(b.end_line or b.start_line)))
    return out


def run_repo(repo: Path, name: str, items: list[dict], wd: Path, encoder, *, rank: int, budget: int, out: Path) -> None:  # noqa: ANN001
    """items: {"set", "id", "base", "fix", "tests", "queries": {set: query}, "gold"}; one sync per case."""
    done = set()
    if out.exists():
        done = {(r["set"], r["repo"], r["case"], r["variant"]) for r in map(json.loads, out.read_text().splitlines())}
    wt = Worktree(repo, wd / "worktree")
    cfg = MemoryConfig(storage_dir=wd / "zt_store", repo_root=wt.path, rank=rank, calibration_samples=1024)
    try:
        with NewtonianMemoryManager(cfg, encoder) as mgr:
            for it in items:
                todo = [(s, v) for s in it["queries"] for v in VARIANTS if (s, name, it["id"], v) not in done]
                if not todo:
                    continue
                wt.checkout(it["base"])
                if it["tests"]:
                    wt.take_files(it["fix"], it["tests"])
                mgr.sync()
                rows = []
                for qset, variant in todo:
                    ctx = GeometricRouter(mgr, VARIANTS[variant]).route_to_compute_engine(it["queries"][qset], token_budget=budget)
                    _, _, body = _coverage(spans(ctx, body_only=True), it["gold"])
                    _, _, seen = _coverage(spans(ctx, body_only=False), it["gold"])
                    rows.append({"set": qset, "repo": name, "case": it["id"], "variant": variant, "tokens": ctx.token_estimate,
                                 "body": body, "map": seen, "full": sum(1 for b in ctx.blocks if b.full),
                                 "signatures": sum(1 for b in ctx.blocks if not b.full)})
                with out.open("a", encoding="utf-8") as fh:
                    fh.writelines(json.dumps(r) + "\n" for r in rows)
                print(f"[{name} {it['id'][:10]}] " + " ".join(
                    f"{r['set']}/{r['variant']}:{'B' if r['body'] else ('m' if r['map'] else '-')}{r['tokens']}" for r in rows), flush=True)
    finally:
        wt.remove()


def report(rows: list[dict]) -> str:
    by = {(r["set"], r["repo"], r["case"], r["variant"]): r for r in rows}
    lines = []
    for qset in sorted({r["set"] for r in rows}):
        cases = sorted({(r["repo"], r["case"]) for r in rows if r["set"] == qset
                        and all((qset, r["repo"], r["case"], v) in by for v in VARIANTS)})
        if not cases:
            continue
        n = len(cases)
        get = lambda c, v, k: by[(qset, *c, v)][k]  # noqa: E731
        mean_tok = {v: sum(get(c, v, "tokens") for c in cases) / n for v in VARIANTS}
        body = {v: sum(get(c, v, "body") for c in cases) for v in VARIANTS}
        seen = {v: sum(get(c, v, "map") for c in cases) for v in VARIANTS}
        # E keeps the focus width of the better of C and D: fewer tokens among those within 2 points of A's body recall
        ok = [v for v in ("C", "D") if (body[v] - body["A"]) / n >= -0.02]
        best = min(ok, key=lambda v: mean_tok[v]) if ok else max(("C", "D"), key=lambda v: body[v])
        e = "E3" if best == "C" else "E5"
        lines += [f"### {qset} — {n} casi", "",
                  "| variante | token medi | vs A | punto nel corpo | punto almeno nella mappa | blocchi interi / firme (medi) |",
                  "|---|---|---|---|---|---|"]
        for v, label in (("A", "A oggi"), ("B", "B anti-cloni"), ("C", "C fuoco F=3"), ("D", "D fuoco F=5"),
                         (e, f"E = {best} + reinvestimento")):
            full = sum(get(c, v, "full") for c in cases) / n
            sig = sum(get(c, v, "signatures") for c in cases) / n
            lines.append(f"| {label} | {mean_tok[v]:.0f} | {100 * (mean_tok[v] / mean_tok['A'] - 1):+.0f}% | "
                         f"{body[v]}/{n} ({100 * body[v] / n:.0f}%) | {seen[v]}/{n} ({100 * seen[v] / n:.0f}%) | {full:.1f} / {sig:.1f} |")
        lines += ["", "| confronto (punto nel corpo) | vinti | persi | McNemar p |", "|---|---|---|---|"]
        for a, b, what in (("B", "A", "B contro A (effetto anti-cloni)"), ("C", "C0", "C contro C0 (anti-cloni a parità di F=3)"),
                           ("D", "D0", "D contro D0 (anti-cloni a parità di F=5)"), ("C", "A", "C contro A"), ("D", "A", "D contro A"),
                           ("C", "B", "C contro B (costo del fuoco F=3)"), ("D", "B", "D contro B (costo del fuoco F=5)"),
                           (e, "A", f"E contro A")):
            wins = sum(get(c, a, "body") and not get(c, b, "body") for c in cases)
            losses = sum(get(c, b, "body") and not get(c, a, "body") for c in cases)
            lines.append(f"| {what} | {wins} | {losses} | {mcnemar_exact(wins, losses):.3g} |")
        wins = sum(get(c, e, "map") and not get(c, best, "map") for c in cases)
        losses = sum(get(c, best, "map") and not get(c, e, "map") for c in cases)
        lines += [f"| mappa: E contro {best} (resa del reinvestimento) | {wins} | {losses} | {mcnemar_exact(wins, losses):.3g} |", ""]
        lines += ["| curva del fuoco (oltre la griglia) | token medi | vs A | punto nel corpo | persi contro A (McNemar p) |",
                  "|---|---|---|---|---|"]
        for v, label in (("C0", "F=3 senza anti-cloni"), ("C", "F=3"), ("D0", "F=5 senza anti-cloni"), ("D", "F=5"),
                         ("F10", "F=10"), ("F15", "F=15")):
            losses = sum(get(c, "A", "body") and not get(c, v, "body") for c in cases)
            wins = sum(get(c, v, "body") and not get(c, "A", "body") for c in cases)
            lines.append(f"| {label} | {mean_tok[v]:.0f} | {100 * (mean_tok[v] / mean_tok['A'] - 1):+.0f}% | "
                         f"{body[v]}/{n} ({100 * body[v] / n:.0f}%) | {losses} ({mcnemar_exact(wins, losses):.3g}) |")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--workdir", type=Path, required=True)
    ap.add_argument("--repos-dir", type=Path, required=True, help="folder with one git clone per repository name")
    ap.add_argument("--loc", type=Path, help="localization cases (query = Italian question; query_en also used)")
    ap.add_argument("--loc-en", type=Path, help="localization cases whose query is the original commit message")
    ap.add_argument("--traceback", type=Path, nargs="*", default=[], help="cases.json from benchmark_bugfix.py mine")
    ap.add_argument("--encoder", default="hashing:1024")
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--budget", type=int, default=12_000)
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    out = args.workdir / "focus_results.jsonl"
    args.workdir.mkdir(parents=True, exist_ok=True)
    if not args.report_only:
        per_repo: dict[str, dict[str, dict]] = {}
        if args.loc:
            for name, cases in json.loads(args.loc.read_text(encoding="utf-8")).items():
                for d in cases:
                    c = LocCase.from_dict(d)
                    per_repo.setdefault(name, {})[c.id] = {"id": c.id, "base": c.base, "fix": c.fix, "tests": [],
                                                           "queries": {"italiano": c.query}, "gold": c.gold}
        if args.loc_en:
            for name, cases in json.loads(args.loc_en.read_text(encoding="utf-8")).items():
                for d in cases:
                    c = LocCase.from_dict(d)
                    item = per_repo.setdefault(name, {}).setdefault(c.id, {"id": c.id, "base": c.base, "fix": c.fix, "tests": [],
                                                                           "queries": {}, "gold": c.gold})
                    item["queries"]["commit"] = c.query
        tb: dict[str, list[dict]] = {}
        for path in args.traceback:
            repo, cases = load_cases(path)
            for c in cases:
                gold = {rel: changed_lines(repo, c.base, c.fix, rel) for rel in c.source_files if not _added_file(repo, c.base, rel)}
                if gold:
                    tb.setdefault(path.parent.name, []).append({"id": c.id, "base": c.base, "fix": c.fix, "tests": list(c.test_files),
                                                                "queries": {"traceback": c.failure_excerpt[-MAX_QUERY_CHARS:]},
                                                                "gold": gold})
        encoder = build_encoder(args.encoder)
        for name in sorted(set(per_repo) | set(tb)):
            repo = (args.repos_dir / name).resolve()
            print(f"== {name}", flush=True)
            if name in per_repo:
                run_repo(repo, name, list(per_repo[name].values()), args.workdir / name / "loc", encoder, rank=args.rank,
                         budget=args.budget, out=out)
            if name in tb:  # separate store: these cases add the fix's test files to the tree
                run_repo(repo, name, tb[name], args.workdir / name / "tb", encoder, rank=args.rank, budget=args.budget, out=out)
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
    text = report(rows)
    (args.workdir / "focus_report.md").write_text("# Contesto a fuoco — fase 1\n\n" + text + "\n", encoding="utf-8")
    print("\n" + text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
