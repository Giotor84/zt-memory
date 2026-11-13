"""Scale check: ingest a large tree, report throughput, RSS, disk, query latency.

    python examples/benchmark_scale.py --repo /usr/lib/python3.13 --store /tmp/zt_bench
"""
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

from zt_memory import HashingEncoder, MemoryConfig, NewtonianMemoryManager, route_to_compute_engine


def rss_mb() -> float:
    from zt_memory.sysinfo import peak_rss_mib

    return peak_rss_mib() or float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--store", type=Path, required=True)
    ap.add_argument("--ram-mb", type=int, default=64)
    ap.add_argument("--backend", default="scan", choices=["scan", "hnsw"])
    args = ap.parse_args()
    shutil.rmtree(args.store, ignore_errors=True)
    cfg = MemoryConfig(storage_dir=args.store, repo_root=args.repo, ram_budget_bytes=args.ram_mb * 2**20,
                       index_backend=args.backend, excluded_dirs=frozenset({"__pycache__", "site-packages"}))
    with NewtonianMemoryManager(cfg, HashingEncoder(dim=1024)) as mgr:
        t0 = time.perf_counter()
        rep = mgr.ingest_repository()
        dt = time.perf_counter() - t0
        st = mgr.stats()
        print(f"files={rep.files_seen} chunks={rep.chunks_added} ingest={dt:.1f}s ({rep.chunks_added / dt:.0f} chunks/s)")
        print(f"nodes={st.nodes_by_kind} disk={st.disk_bytes / 2**20:.1f} MiB ws={st.working_set}")
        for q in ["parse http headers from a socket", "json decoder error handling", "thread pool executor shutdown"]:
            t0 = time.perf_counter()
            ctx = route_to_compute_engine(q, mgr)
            ms = (time.perf_counter() - t0) * 1000
            print(f"{ms:7.1f} ms  blocks={len(ctx.blocks):2d} tokens={ctx.token_estimate:5d}  top={ctx.blocks[0].title if ctx.blocks else '-'}")
        print(f"peak RSS={rss_mb():.0f} MiB")


if __name__ == "__main__":
    main()
