"""End-to-end bug fixing: BM25 whole files vs zt-memory spans, same local model (see zt_memory/bench_bugfix.py).

    # 1. mine 10-15 real, reproducible bugs from a Python repo with pytest (any LLM server state)
    python examples/benchmark_bugfix.py mine --repo C:\\src\\click --workdir C:\\zt_bench\\click --limit 15

    # 2. optional, llama-server OFF: first full index on the GPU (otherwise `run` does it on CPU, slower)
    python examples/benchmark_bugfix.py index --workdir C:\\zt_bench\\click --device cuda

    # 3. llama-server ON: run every case under every condition (resumable: rerun to continue)
    python examples/benchmark_bugfix.py run --workdir C:\\zt_bench\\click

    # 4. tables again at any time
    python examples/benchmark_bugfix.py report --workdir C:\\zt_bench\\click

The repo's test dependencies must be importable by this Python (pip install -e <repo> or its test extras);
`mine` keeps only cases whose tests fail before and pass after the real fix in *this* environment.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from zt_memory.bench_bugfix import CONDITIONS, Worktree, load_cases, load_results, mine_cases, run_benchmark, save_cases, summarize
from zt_memory.config import MemoryConfig, RouterConfig
from zt_memory.encoders import build_encoder
from zt_memory.engine_client import ComputeEngineError, make_engine_client
from zt_memory.project import BEDROCK_DEFAULTS, ENGINE_DEFAULTS
from zt_memory.manager import NewtonianMemoryManager


def _encoder(args: argparse.Namespace):  # noqa: ANN202
    opts = {"device": args.device} if args.device else {}
    return build_encoder(args.encoder, **opts)


def cmd_mine(args: argparse.Namespace) -> int:
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    cases = mine_cases(Path(args.repo), workdir, max_commits=args.max_commits, limit=args.limit, max_source_files=args.max_source_files,
                       max_loc=args.max_loc, require_bug_words=not args.any_message, test_timeout_s=args.test_timeout,
                       keep_addopts=args.keep_addopts)
    save_cases(workdir / "cases.json", Path(args.repo), cases)
    print(f"{len(cases)} valid case(s) -> {workdir / 'cases.json'} (edit the file to drop or reword cases)")
    return 0 if cases else 1


def cmd_index(args: argparse.Namespace) -> int:
    workdir = Path(args.workdir)
    repo, cases = load_cases(workdir / "cases.json")
    if not cases:
        print("no cases: run `mine` first")
        return 1
    wt = Worktree(repo, workdir / "worktree")
    wt.checkout(cases[0].base)
    t = time.perf_counter()
    encoder = _encoder(args)
    with NewtonianMemoryManager(MemoryConfig(storage_dir=workdir / "zt_store", repo_root=wt.path, rank=args.rank), encoder) as mgr:
        report = mgr.sync()
    release = getattr(encoder, "release", None)
    if callable(release):
        release()
    wt.remove()
    print(f"indexed {cases[0].base[:9]} in {time.perf_counter() - t:.0f}s: {report.summary()}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    workdir = Path(args.workdir)
    repo, cases = load_cases(workdir / "cases.json")
    if args.cases:
        wanted = set(args.cases.split(","))
        cases = [c for c in cases if c.id in wanted]
    settings = {**ENGINE_DEFAULTS, **(BEDROCK_DEFAULTS if args.backend == "bedrock" else {}),
                "backend": args.backend, "url": args.engine, "model": args.model or ("local" if args.backend == "openai" else ""),
                "timeout_s": args.timeout, "region": args.region, "proxy": args.proxy, "max_calls": args.max_calls}
    if args.backend == "bedrock" and not settings["model"]:
        print("--model is required with --backend bedrock (e.g. minimax.minimax-m2.5)")
        return 2
    client = make_engine_client(settings, usage_log=workdir / "engine_usage.jsonl")
    if not client.health():
        print(f"compute engine not reachable at {args.engine}: start llama-server first" if args.backend == "openai"
              else f"bedrock: {client.last_error}")
        return 6
    # The whole-file baseline packs files up to the served context: with a 196k-token Bedrock window that would be a
    # ~190k-token prompt per call (cost) and a different experiment than the 16k local runs. Default to 16k unless asked.
    served = args.served_ctx or (16384 if args.backend == "bedrock" else None)
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    encoder = _encoder(args)
    print(f"{len(cases)} case(s) x {conditions} | engine ctx {client.context_size()} (used {served or 'served'}) | "
          f"encoder {args.encoder} ({args.device or 'cpu'})"
          + (f" | max_calls {args.max_calls} per run, usage log {workdir / 'engine_usage.jsonl'}" if args.backend == "bedrock" else ""))
    try:
        results = run_benchmark(repo, cases, workdir, client, encoder, conditions=conditions, rank=args.rank,
                                router_config=RouterConfig(), max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                                retries=args.retries, served_ctx=served, with_failure_log=args.with_failure_log,
                                test_timeout_s=args.test_timeout, keep_addopts=args.keep_addopts)
    except ComputeEngineError as exc:
        if not getattr(exc, "stop_run", False):
            raise
        print(f"\nstopped: {exc}\nresults so far are kept in {workdir / 'results.jsonl'}; run again to continue")
        _report(workdir, load_results(workdir / "results.jsonl"), conditions)
        return 6
    return _report(workdir, results, conditions)


def cmd_report(args: argparse.Namespace) -> int:
    workdir = Path(args.workdir)
    return _report(workdir, load_results(workdir / "results.jsonl"), list(CONDITIONS))


def _report(workdir: Path, results, conditions) -> int:  # noqa: ANN001
    text = summarize(results, conditions)
    (workdir / "report.md").write_text("# Bug-fix benchmark\n\n" + text + "\n", encoding="utf-8")
    print("\n" + text + f"\n\nreport: {workdir / 'report.md'} | raw: {workdir / 'results.jsonl'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--workdir", required=True, help="benchmark folder: cases.json, zt_store/, results.jsonl, report.md")

    def encoder_opts(p: argparse.ArgumentParser) -> None:
        p.add_argument("--encoder", default="hf:Qwen/Qwen3-Embedding-0.6B")
        p.add_argument("--rank", type=int, default=128, help="must be the same for index and run")
        p.add_argument("--device", help="encoder device (default cpu; cuda only while llama-server is off)")

    def test_opts(p: argparse.ArgumentParser) -> None:
        p.add_argument("--test-timeout", type=float, default=120.0, help="seconds per pytest run (hanging tests count as failures)")
        p.add_argument("--keep-addopts", action="store_true", help="keep the repo's pytest addopts (default: cleared)")

    p = sub.add_parser("mine", help="find reproducible fix commits and write cases.json")
    common(p)
    test_opts(p)
    p.add_argument("--repo", required=True, help="local clone of a Python project with pytest tests")
    p.add_argument("--limit", type=int, default=15)
    p.add_argument("--max-commits", type=int, default=600)
    p.add_argument("--max-source-files", type=int, default=2)
    p.add_argument("--max-loc", type=int, default=40, help="max changed source lines per fix (small, single-shot bugs)")
    p.add_argument("--any-message", action="store_true", help="do not require bug words (fix, bug, error...) in the subject")
    p.set_defaults(fn=cmd_mine)

    p = sub.add_parser("index", help="first full index at the oldest case (run with llama-server off to use the GPU)")
    common(p)
    encoder_opts(p)
    p.set_defaults(fn=cmd_index)

    p = sub.add_parser("run", help="run all conditions on all cases (resumable)")
    common(p)
    encoder_opts(p)
    test_opts(p)
    p.add_argument("--engine", default="http://127.0.0.1:8080/v1", help="OpenAI-compatible URL (backend openai)")
    p.add_argument("--backend", choices=("openai", "bedrock"), default="openai")
    p.add_argument("--model", help="model name (bedrock: model ID, e.g. minimax.minimax-m2.5)")
    p.add_argument("--region", default=ENGINE_DEFAULTS["region"], help="bedrock region")
    p.add_argument("--proxy", default="", help="bedrock HTTPS proxy (default: HTTPS_PROXY or none)")
    p.add_argument("--max-calls", type=int, default=ENGINE_DEFAULTS["max_calls"],
                   help="bedrock: paid calls allowed in this run (a run stops cleanly and resumes later)")
    p.add_argument("--served-ctx", type=int, help="context the prompts must fit in (default: the server's; bedrock 16384)")
    p.add_argument("--timeout", type=float, default=900.0, help="seconds per LLM request")
    p.add_argument("--conditions", default=",".join(CONDITIONS), help=f"comma list from {CONDITIONS}")
    p.add_argument("--cases", help="comma list of case ids to run (default: all)")
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--retries", type=int, default=1)
    p.add_argument("--with-failure-log", action="store_true", help="append the failing pytest output to the bug report")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("report", help="print and save the tables from results.jsonl")
    common(p)
    p.set_defaults(fn=cmd_report)

    args = ap.parse_args()
    return int(args.fn(args))


if __name__ == "__main__":
    sys.exit(main())
