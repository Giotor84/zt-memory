"""Does pinning the locations named in a traceback put the fix spot in the context more often?

Input: bug cases mined by `benchmark_bugfix.py mine` (tests fail before the fix, pass after). The query is the
failing test output - what a developer pastes into a chat. For each case the repository is checked out at the
parent commit with the fix's test files (so the traceback's line numbers are real), indexed, and routed with
traceback anchors on and off, plus BM25 over whole files as a reference.

    python examples/benchmark_traceback.py --workdir DIR CASES_JSON [CASES_JSON ...]
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from zt_memory.bench_bugfix import Worktree, bm25_rank_files, load_cases, mcnemar_exact
from zt_memory.bench_localize import _added_file, _coverage, _routed_spans, changed_lines, pack_whole_files, router_configs
from zt_memory.config import MemoryConfig
from zt_memory.discovery import discover
from zt_memory.encoders import build_encoder
from zt_memory.manager import NewtonianMemoryManager
from zt_memory.router import GeometricRouter

METHODS = ("zt", "zt-no-anchors", "bm25-chunks", "bm25-chunks-no-anchors", "bm25-files")
MAX_QUERY_CHARS = 8000  # the tail of the log: the part with the traceback and the error


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cases", nargs="+", type=Path, help="cases.json files written by benchmark_bugfix.py mine")
    ap.add_argument("--workdir", type=Path, required=True)
    ap.add_argument("--encoder", default="hashing:1024")
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--budget", type=int, default=12_000)
    args = ap.parse_args(argv)
    args.workdir.mkdir(parents=True, exist_ok=True)
    out = args.workdir / "traceback_results.jsonl"
    done = {(r["repo"], r["case"], r["method"]) for r in _load(out)}
    encoder = build_encoder(args.encoder)
    base_cfgs = router_configs()
    configs = {
        "zt": base_cfgs["zt"], "zt-no-anchors": replace(base_cfgs["zt"], trace_anchors=0),
        "bm25-chunks": base_cfgs["bm25-chunks"], "bm25-chunks-no-anchors": replace(base_cfgs["bm25-chunks"], trace_anchors=0),
    }
    for cases_file in args.cases:
        repo, cases = load_cases(cases_file)
        name = cases_file.parent.name
        wd = args.workdir / name
        wt = Worktree(repo, wd / "worktree")
        cfg = MemoryConfig(storage_dir=wd / "zt_store", repo_root=wt.path, rank=args.rank)
        try:
            with NewtonianMemoryManager(cfg, encoder) as mgr:
                for case in cases:
                    if all((name, case.id, m) in done for m in METHODS):
                        continue
                    gold = {rel: changed_lines(repo, case.base, case.fix, rel)
                            for rel in case.source_files if not _added_file(repo, case.base, rel)}
                    if not gold:
                        continue
                    wt.checkout(case.base)
                    wt.take_files(case.fix, case.test_files)
                    mgr.sync()
                    query = case.failure_excerpt[-MAX_QUERY_CHARS:]
                    rows = []
                    for method in METHODS:
                        if method == "bm25-files":
                            scored = bm25_rank_files(wt.path, discover(cfg), query)
                            whole, tokens = pack_whole_files(wt.path, scored, args.budget)
                            spans, pinned = {rel: [span] for rel, span in whole.items()}, ()
                        else:
                            ctx = GeometricRouter(mgr, configs[method]).route_to_compute_engine(query, token_budget=args.budget)
                            spans, tokens, pinned = _routed_spans(ctx), ctx.token_estimate, ctx.pinned
                        _, _, spot = _coverage(spans, gold)
                        pinned_gold = any(p.rsplit(":", 1)[0] in gold for p in pinned)
                        rows.append({"repo": name, "case": case.id, "method": method, "spot": spot, "tokens": int(tokens),
                                     "pinned": list(pinned), "pinned_gold_file": pinned_gold})
                    with out.open("a", encoding="utf-8") as fh:
                        for r in rows:
                            fh.write(json.dumps(r) + "\n")
                    print(f"[{name} {case.id}] " + " | ".join(
                        f"{r['method']} {'yes' if r['spot'] else 'no'} {r['tokens']}t" for r in rows)
                        + f" | pinned {rows[0]['pinned']}", flush=True)
        finally:
            wt.remove()
    print(report(_load(out)))
    return 0


def _load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def report(rows: list[dict]) -> str:
    by = {(r["repo"], r["case"], r["method"]): r for r in rows}
    cases = sorted({(r["repo"], r["case"]) for r in rows if all((r["repo"], r["case"], m) in by for m in METHODS)})
    if not cases:
        return "no complete cases"
    n = len(cases)
    with_frames = [c for c in cases if by[(*c, "zt")]["pinned"]]
    gold_frames = [c for c in cases if by[(*c, "zt")]["pinned_gold_file"]]
    lines = [f"## Traceback anchors - {n} bugs ({len({c[0] for c in cases})} repos)", "",
             f"- logs naming at least one indexed file:line: {len(with_frames)}/{n}",
             f"- logs naming a line in a file the fix changes: {len(gold_frames)}/{n}", "",
             "| method | fix spot in context | mean tokens |", "|---|---|---|"]
    for m in METHODS:
        hit = sum(by[(*c, m)]["spot"] for c in cases)
        tok = sum(by[(*c, m)]["tokens"] for c in cases) / n
        lines.append(f"| {m} | {hit}/{n} ({100 * hit / n:.0f}%) | {tok:.0f} |")
    lines.append("")
    for a, b in (("zt", "zt-no-anchors"), ("bm25-chunks", "bm25-chunks-no-anchors"), ("zt", "bm25-files")):
        wins = sum(by[(*c, a)]["spot"] and not by[(*c, b)]["spot"] for c in cases)
        losses = sum(by[(*c, b)]["spot"] and not by[(*c, a)]["spot"] for c in cases)
        lines.append(f"- {a} vs {b}: {wins} cases won, {losses} lost (McNemar p = {mcnemar_exact(wins, losses):.3g})")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
