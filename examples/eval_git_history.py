"""Ablation on real history: do the hyperbolic tree and the Newtonian physics improve retrieval?

    python examples/eval_git_history.py --repo https://github.com/psf/requests --commits 200 --workdir C:\\zt_eval
    python examples/eval_git_history.py --repo C:\\code\\my-repo --encoder hf:Qwen/Qwen3-Embedding-0.6B \\
        --rank 128 --projector C:\\path\\to\\proj.pt --commits 150

Each commit message becomes a query; the files the commit modified are the answer. All configurations are
scored on the same store state at every step (paired), then the commit is applied and synced.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from zt_memory.encoders import build_encoder
from zt_memory.replay import default_configs, replay


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True, help="git URL or local path (it is cloned, never modified)")
    ap.add_argument("--workdir", type=Path, default=Path("zt_eval"))
    ap.add_argument("--commits", type=int, default=200)
    ap.add_argument("--encoder", default="hashing:1024")
    ap.add_argument("--rank", type=int, default=128)
    ap.add_argument("--projector", type=Path, help="frozen basis from calibrate_encoder.py (skips self-calibration)")
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--budget", type=int, default=12_000)
    ap.add_argument("--calibration-samples", type=int, default=512, help=">= 4 x rank; encoded once on the target repo")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    if args.threads:
        import torch

        torch.set_num_threads(args.threads)
    opts = {"max_length": args.max_length} if args.encoder.startswith("hf:") else {}
    encoder = build_encoder(args.encoder, **opts)
    def progress(i: int, n: int, case) -> None:  # noqa: ANN001
        tag = "merge" if case.is_merge else ("query" if case.truth else "skip ")
        print(f"\r[{i}/{n}] {tag} {case.sha[:8]} {case.query[:50]:50s}", end="", file=sys.stderr, flush=True)

    overrides: dict[str, object] = {"rank": args.rank, "calibration_samples": args.calibration_samples}
    report = replay(args.repo, args.workdir, encoder, commits=args.commits, configs=default_configs(),
                    cutoff_budget=args.budget, memory_overrides=overrides, progress=progress,
                    projector_path=args.projector)
    print(file=sys.stderr)
    print(f"\n{report.repo}: {report.commits_scanned} commits replayed, {report.cases} queried | "
          f"initial sync {report.initial_sync_seconds:.0f}s, total {report.seconds:.0f}s")
    ks = report.ks
    header = f"{'config':24s} " + " ".join(f"{'hit@' + str(k):>7s}" for k in ks) + " " + \
             " ".join(f"{'rec@' + str(k):>7s}" for k in ks) + f" {'MRR':>6s}"
    print(header)
    for name, s in report.configs.items():
        print(f"{name:24s} " + " ".join(f"{s[f'hit@{k}']:7.3f}" for k in ks) + " " +
              " ".join(f"{s[f'recall@{k}']:7.3f}" for k in ks) + f" {s['mrr']:6.3f}")
    print(f"\ncutoff on F_defaults (budget {args.budget}): share of true files inside the packed context vs size")
    for c in report.cutoff:
        print(f"  alpha {c['alpha']:.2f}: coverage {c['coverage']:.3f}  avg tokens {c['avg_tokens']:7.0f}  avg blocks {c['avg_blocks']:5.1f}")
    print("\npaired differences (95% bootstrap CI; a CI that contains 0 means: no measurable effect)")
    for label, metrics in report.comparisons.items():
        m, r = metrics["mrr"], metrics["recall@10"]
        verdict = "helps" if r["lo"] > 0 or m["lo"] > 0 else ("hurts" if r["hi"] < 0 or m["hi"] < 0 else "no effect")
        print(f"  {label:58s} MRR {m['diff']:+.3f} [{m['lo']:+.3f},{m['hi']:+.3f}]  rec@10 {r['diff']:+.3f} [{r['lo']:+.3f},{r['hi']:+.3f}]  -> {verdict}")
    if args.out:
        args.out.write_text(report.to_json())
    return 0


if __name__ == "__main__":
    sys.exit(main())
