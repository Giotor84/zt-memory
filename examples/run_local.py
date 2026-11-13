"""End-to-end loop: navigator (CPU) -> geometric router -> compute engine (VRAM, OpenAI-compatible server).

Brownfield:
    python examples/run_local.py brownfield --repo ~/code/app --store ~/.zt/app "fix retry logic in the http client"
Greenfield:
    python examples/run_local.py greenfield --repo ~/code/new --store ~/.zt/new \
        --spec "REST API for invoices with pagination" --spec "PostgreSQL repository layer" "scaffold the project"

--encoder hashing (offline default) | hf:<model>  e.g. hf:Qwen/Qwen3-Embedding-0.6B (CPU, pooling=last)
--engine  http://127.0.0.1:8080/v1   omit for a dry run that prints the routed prompt.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from zt_memory import (
    ComputeEngineClient, HashingEncoder, MemoryConfig, NewtonianMemoryManager, NodeKind, Requirement,
    RouterConfig, TransformersHiddenStateEncoder, parse_generated_files, route_to_compute_engine,
    write_generated_files,
)
from zt_memory.encoders import HiddenStateEncoder


def build_encoder(spec: str) -> HiddenStateEncoder:
    if spec == "hashing":
        return HashingEncoder(dim=1024)
    if spec.startswith("hf:"):
        return TransformersHiddenStateEncoder(
            spec[3:], pooling="last", device="cpu",
            query_prefix="Instruct: Given a code change request, retrieve relevant code\nQuery: ",
        )
    raise SystemExit(f"unknown encoder {spec!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["brownfield", "greenfield"])
    ap.add_argument("task")
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--store", type=Path, required=True)
    ap.add_argument("--spec", action="append", default=[])
    ap.add_argument("--encoder", default="hashing")
    ap.add_argument("--engine", default=None)
    ap.add_argument("--budget", type=int, default=12_000)
    ap.add_argument("--ram-mb", type=int, default=1536)
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    args = ap.parse_args()

    args.repo.mkdir(parents=True, exist_ok=True)
    cfg = MemoryConfig(storage_dir=args.store, repo_root=args.repo, ram_budget_bytes=args.ram_mb * 2**20)
    with NewtonianMemoryManager(cfg, build_encoder(args.encoder)) as mgr:
        if args.spec:
            mgr.seed_requirements([Requirement(s) for s in args.spec])
        report = mgr.ingest_repository()
        print(f"[sync] +{report.chunks_added} ~{report.chunks_updated} -{report.chunks_removed} "
              f"unchanged={report.chunks_unchanged} mean_surprise={report.mean_surprise:.3f}")

        ctx = route_to_compute_engine(args.task, mgr, config=RouterConfig(token_budget=args.budget), mode=args.mode)
        print(f"[route] mode={ctx.mode} blocks={len(ctx.blocks)} tokens~{ctx.token_estimate}/{ctx.token_budget}")
        if args.engine is None:
            print(ctx.render(line_numbers=args.mode == "brownfield"))
            return

        answer = ComputeEngineClient(args.engine).complete(ctx, max_tokens=args.max_new_tokens)
        if args.mode == "brownfield":
            print(answer)  # unified diff: review, then `git apply`; the next sync measures its surprise
            return
        written = write_generated_files(parse_generated_files(answer), args.repo)
        spec_ids = [b.node_ids[0] for b in ctx.blocks if b.kind == NodeKind.SPEC]
        for rel in written:
            mgr.register_generated_artifact(rel, spec_ids)
        print(f"[greenfield] wrote {written} bound to specs {spec_ids}")


if __name__ == "__main__":
    main()
