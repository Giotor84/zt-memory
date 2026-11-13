"""Calibrate a real embedding model on YOUR code and choose the SVD rank.

    pip install -e ".[hf,treesitter]"
    python examples/calibrate_encoder.py --repo ~/code/app \
        --encoder hf:Qwen/Qwen3-Embedding-0.6B --samples 2000 --ranks 64,128,192,256,384 \
        --probes probes.json --save-projector /tmp/proj.pt --out report.json

Prints, per rank: cumulative EVR, neighbor recall@k vs the raw embeddings (what retrieval loses to
compression), probe recall@k (labelled queries, optional) and SSD bytes/node; the same for the
"prefix" baseline (first r raw dims, valid for Matryoshka models). Recommends the smallest rank
reaching --target and prints the matching `zt init` command. One encoding pass serves all ranks.

probes.json: [{"query": "retry with backoff on 503", "expected": "src/http/retry.py"}, ...]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from zt_memory.config import MemoryConfig
from zt_memory.encoders import build_encoder
from zt_memory.sysinfo import current_rss_mib, peak_rss_mib
from zt_memory.evaluation import (
    CalibrationReport, centering_ceiling, encode_corpus, load_probes, probe_hits, rank_sweep, recommend_rank, reduce, sample_corpus,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--encoder", default="hf:Qwen/Qwen3-Embedding-0.6B")
    ap.add_argument("--samples", type=int, default=2000, help="chunks to encode (>= 4x the largest rank)")
    ap.add_argument("--ranks", default="32,64,128,192,256,384,512")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--anchors", type=int, default=512)
    ap.add_argument("--target", type=float, default=0.90, help="neighbor-recall target for the recommendation")
    ap.add_argument("--drop-top", type=int, default=0, help="discard dominant components (raw LM hidden states)")
    ap.add_argument("--probes", type=Path)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--device", default="cpu", help="cpu or cuda (needs a CUDA build of torch)")
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = torch default)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-projector", type=Path, help="write the frozen basis at the recommended rank")
    ap.add_argument("--out", type=Path, help="JSON report")
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    ranks = sorted(int(r) for r in args.ranks.split(","))
    probes = load_probes(args.probes) if args.probes else []
    config = MemoryConfig(storage_dir=args.repo / ".zt" / "calibration-tmp", repo_root=args.repo)

    sample = sample_corpus(config, args.samples, seed=args.seed, include_prefixes=[p.expected for p in probes])
    print(f"[corpus] {sample.total_files} files, {sample.total_chunks} chunks; sampled {len(sample.texts)}", file=sys.stderr)
    if len(sample.texts) < 4 * max(ranks):
        print(f"[warn] {len(sample.texts)} samples < 4 x max rank: SVD estimates at high rank are unreliable", file=sys.stderr)

    hf_opts = {"max_length": args.max_length, "batch_size": args.batch_size, "dtype": args.dtype,
               "device": args.device} if args.encoder.startswith("hf:") else {}
    encoder = build_encoder(args.encoder, **hf_opts)
    after_load = current_rss_mib()
    print(f"[encoder] {encoder.fingerprint} dim={encoder.dim} device={args.device} dtype={args.dtype}"
          + (f" | RSS after load {after_load:.0f} MiB" if after_load is not None else ""), file=sys.stderr)

    def progress(done: int, total: int) -> None:
        if sys.stderr.isatty():
            print(f"\r[encode] {done}/{total}", end="", file=sys.stderr, flush=True)

    hidden, secs = encode_corpus(encoder, sample.texts, batch=max(args.batch_size * 8, 64), progress=progress)
    print(file=sys.stderr)
    query_hidden = encoder.encode([p.query for p in probes], is_query=True) if probes else None
    rows, projectors = rank_sweep(hidden, ranks, k=args.k, n_anchors=args.anchors, drop_top=args.drop_top, seed=args.seed,
                                  query_hidden=query_hidden, probes=probes, doc_paths=sample.paths)
    ceiling = next((r.ceiling for r in rows if r.method == "svd"), 1.0)
    report = CalibrationReport(encoder.fingerprint, encoder.dim, len(sample.texts), sample.total_chunks, secs,
                               len(sample.texts) / max(secs, 1e-9), args.k, rows, ceiling)
    peak = peak_rss_mib()
    peak_txt = f"{peak:.0f} MiB" if peak is not None else "n/a (pip install psutil)"
    print(f"\nencode: {report.chunks_per_second:.1f} chunks/s on {args.device} -> full repo ~{sample.total_chunks / report.chunks_per_second / 60:.1f} min; peak RSS {peak_txt}")
    print(f"centering ceiling (nbr@{args.k} of raw vs mean-centered, full dim): {ceiling:.3f}  <- max reachable by 'svd'")
    probe_col = f"probe@{args.k}"
    print(f"{'method':8s} {'rank':>5s} {'EVR':>6s} {'nbr@' + str(args.k):>7s} {'fidel.':>7s} {probe_col:>8s} {'B/node':>7s}")
    for r in rows:
        evr = f"{r.evr:.3f}" if r.evr is not None else "  -  "
        probe = f"{r.probe_recall:.3f}" if r.probe_recall is not None else "   -  "
        print(f"{r.method:8s} {r.rank:5d} {evr:>6s} {r.neighbor_recall:7.3f} {r.fidelity:7.3f} {probe:>8s} {r.bytes_per_node:7d}")
    if probes:
        print(f"note: {len(probes)} probes -> one probe = {1 / len(probes):.3f}; differences of one probe are not evidence")

    best = recommend_rank(rows, args.target)
    if best is not None:
        reached = best.neighbor_recall >= args.target
        print(f"\nrecommended: {best.method} rank {best.rank} (neighbor recall {best.neighbor_recall:.3f}"
              + (f" >= {args.target})" if reached else f"; target {args.target} not reached: best fidelity tested)"))
        prefix_same = next((r for r in rows if r.method == "prefix" and r.rank == best.rank), None)
        if prefix_same is not None and prefix_same.neighbor_recall >= best.neighbor_recall:
            print("note: Matryoshka prefix truncation matches SVD at this rank: the SVD stage buys nothing for this encoder")
        if probes and query_hidden is not None:
            details = probe_hits(reduce(best.method, projectors, query_hidden, best.rank),
                                 reduce(best.method, projectors, hidden, best.rank), sample.paths, probes, k=args.k)
            report.probe_details = details
            print(f"\nprobes @ {best.method}/{best.rank}:")
            for d in details:
                mark = f"hit@{d['hit_rank'] + 1}" if d["hit_rank"] is not None else "MISS "
                print(f"  {mark:7s} {d['query'][:60]:60s} top1={d['top1']}")
        if args.save_projector:
            projectors[best.method].truncated(best.rank).save(args.save_projector)
            print(f"saved frozen basis ({best.method}) -> {args.save_projector}")
        extra = f" --projector {args.save_projector}" if args.save_projector else ""
        print(f"next: zt init --repo {args.repo} --encoder {args.encoder} --rank {best.rank} --max-length {args.max_length}{extra}")
    if args.out:
        args.out.write_text(report.to_json())
    return 0


if __name__ == "__main__":
    sys.exit(main())
