"""Localization benchmark over several repositories (no LLM): is the code the real fix changed in the context?

    python examples/benchmark_localize.py --workdir C:\\zt_bench\\loc C:\\src\\click C:\\src\\flask C:\\src\\cobra
    python examples/benchmark_localize.py --workdir C:\\zt_bench\\loc --report-only

Cases are mined once per repository (cases.json, editable) and every run is resumable (loc_results.jsonl).
See zt_memory/bench_localize.py for the methods and metrics.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from zt_memory.bench_localize import (
    LocCase, load_loc_results, mine_localization_cases, per_repo_tables, run_localization, summarize_localization,
)
from zt_memory.encoders import build_encoder


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("repos", nargs="*", type=Path, help="local git clones")
    ap.add_argument("--workdir", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=25, help="cases per repository")
    ap.add_argument("--max-commits", type=int, default=1500)
    ap.add_argument("--max-files", type=int, default=3)
    ap.add_argument("--max-loc", type=int, default=100)
    ap.add_argument("--budget", type=int, default=12_000)
    ap.add_argument("--encoder", default="hashing:1024")
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--device")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--cases-file", type=Path,
                    help="ready-made cases per repository folder name, e.g. examples/data/loc_cases_it.json "
                         "(same cases, questions rewritten in plain Italian without code names)")
    args = ap.parse_args()

    results = []
    if not args.report_only:
        encoder = build_encoder(args.encoder, **({"device": args.device} if args.device else {}))
        import subprocess

    bad = [str(r) for r in args.repos
           if subprocess.run(["git", "-C", str(r), "rev-parse", "--git-dir"], capture_output=True).returncode != 0]
    if bad and not args.report_only:
        ap.error("not git repositories (clone them first): " + ", ".join(bad))
    names = [r.name for r in args.repos]
    if len(set(names)) != len(names):
        ap.error(f"repository folder names must be distinct: {names}")
    for repo in args.repos:
            wd = args.workdir / repo.name  # the folder name as given (names must be distinct)
            wd.mkdir(parents=True, exist_ok=True)
            cases_path = wd / "cases.json"
            if args.cases_file:
                given = json.loads(args.cases_file.read_text(encoding="utf-8")).get(repo.name)
                if given is None:
                    ap.error(f"{args.cases_file} has no cases for a repository folder named {repo.name!r}")
                cases = [LocCase.from_dict(d) for d in given]
                if cases_path.exists() and [c.query for c in cases] != [
                        d["query"] for d in json.loads(cases_path.read_text(encoding="utf-8"))]:
                    ap.error(f"{wd} already holds results for other questions: use a new --workdir")
                cases_path.write_text(json.dumps([asdict(c) for c in cases], indent=2, ensure_ascii=False), encoding="utf-8")
            elif cases_path.exists():
                cases = [LocCase.from_dict(d) for d in json.loads(cases_path.read_text(encoding="utf-8"))]
            else:
                cases = mine_localization_cases(repo, max_commits=args.max_commits, limit=args.limit,
                                                max_files=args.max_files, max_loc=args.max_loc)
                cases_path.write_text(json.dumps([asdict(c) for c in cases], indent=2), encoding="utf-8")
            print(f"== {repo.name}: {len(cases)} cases", flush=True)
            run_localization(repo, cases, wd, encoder, rank=args.rank, budget=args.budget,
                             echo=lambda s: print(s, flush=True))
    for wd in sorted(p for p in args.workdir.iterdir() if (p / "loc_results.jsonl").exists()):
        results.extend(load_loc_results(wd / "loc_results.jsonl"))
    text = summarize_localization(results) + "\n\n" + per_repo_tables(results)
    (args.workdir / "loc_report.md").write_text("# Localization benchmark\n\n" + text + "\n", encoding="utf-8")
    print("\n" + text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
