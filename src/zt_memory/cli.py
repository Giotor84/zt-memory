"""`zt` command line: init, sync, status, watch, route, code, undo, engine, mcp, hook.

Exit codes: 0 ok, 1 error, 2 usage, 3 store locked, 4 no usable edit, 5 check failed, 6 engine unreachable.
"""
from __future__ import annotations

import argparse
import json
import shlex
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

from .config import NodeKind
from .project import CONFIG_DIR, ProjectConfig, ProjectConfigError, exclude_from_git
from .engine_client import ComputeEngineError
from .storage import StoreLockedError

HOOK_MARKER = "# zt-memory managed hook"
HOOK_NAMES = ("post-commit", "post-merge", "post-checkout", "post-rewrite")


class CliError(RuntimeError):
    pass


def _project(args: argparse.Namespace) -> ProjectConfig:
    project = ProjectConfig.load(Path(args.repo)) if args.repo else ProjectConfig.find()
    if getattr(args, "lock_timeout", None) is not None:
        project.memory["lock_timeout_s"] = float(args.lock_timeout)
    return project


def _emit(args: argparse.Namespace, payload: dict[str, object], text: str) -> None:
    if getattr(args, "json", False):
        print(json.dumps(payload, indent=2, default=str))
    elif not getattr(args, "quiet", False):
        print(text)


# ------------------------------------------------------------------------------------------
# commands
# ------------------------------------------------------------------------------------------
def cmd_init(args: argparse.Namespace) -> int:
    root = Path(args.repo or ".").resolve()
    if not root.is_dir():
        raise CliError(f"{root} is not a directory")
    store = Path(args.store).expanduser() if args.store else root / CONFIG_DIR / "store"
    encoder: dict[str, object] = {"spec": args.encoder}
    if args.max_length:
        encoder["max_length"] = args.max_length
    if args.dtype:
        encoder["dtype"] = args.dtype
    memory: dict[str, object] = {"rank": args.rank, "ram_budget_mb": args.ram_mb}
    engine = {"url": args.engine_url} if args.engine_url else {}
    project = ProjectConfig(root, store if store.is_absolute() else root / store, encoder, memory, {"token_budget": args.budget}, engine)
    path = project.save(overwrite=args.force)
    if args.projector:
        _install_projector(project, Path(args.projector), args.rank)
    git = exclude_from_git(root)
    print(f"wrote {path}" + (" (added /.zt/ to .git/info/exclude)" if git else ""))
    print("next: `zt sync` (first run embeds the whole repository), then `zt hook install` or `zt watch`")
    return 0


def _install_projector(project: ProjectConfig, src: Path, rank: int) -> None:
    from .projector import SubspaceProjector

    proj = SubspaceProjector.load(src)
    if proj.rank != rank:
        raise CliError(f"projector rank {proj.rank} != --rank {rank}")
    if not proj.frozen:
        raise CliError("projector is not frozen")
    project.store.mkdir(parents=True, exist_ok=True)
    target = project.store / "projector.pt"
    if (project.store / "meta.sqlite3").exists():
        raise CliError(f"store {project.store} already initialised: the basis cannot change under stored vectors")
    shutil.copyfile(src, target)


def cmd_sync(args: argparse.Namespace) -> int:
    project = _project(args)
    for key in ("device", "dtype", "batch_size"):
        value = getattr(args, key, None)
        if value is not None:  # per-run override, e.g. first bulk sync on GPU while the LLM is not loaded
            project.encoder[key] = value
    encoder = project.build_encoder()
    try:
        with project.open_manager(encoder) as mgr:
            report = mgr.sync(full=args.full)
    finally:
        release = getattr(encoder, "release", None)
        if callable(release):
            release()  # frees VRAM immediately when the run used CUDA
    _emit(args, report.to_dict(), report.summary())
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    project = _project(args)
    with project.open_manager() as mgr:
        cs = mgr.scan_changes()
        st = mgr.stats()
        hot = mgr.hottest(args.hot, frozenset({NodeKind.CHUNK}))
        recs = mgr.records(i for i, _ in hot)
    payload = {
        "repo": str(project.repo_root), "store": str(project.store), "encoder": project.encoder.get("spec"),
        "nodes": st.nodes_by_kind, "disk_mib": round(st.disk_bytes / 2**20, 2), "clock": st.clock,
        "index": st.index_backend, "pending": {"added": cs.added, "changed": cs.changed, "removed": cs.removed},
        "touched": [p for p, _ in cs.touched],
        "hottest": [{"key": recs[i].key, "energy": round(e, 4)} for i, e in hot if i in recs],
    }
    pending = len(cs.pending)
    lines = [
        f"repo   {project.repo_root}", f"store  {project.store} ({payload['disk_mib']} MiB, index={st.index_backend})",
        f"nodes  " + ", ".join(f"{k}={v}" for k, v in st.nodes_by_kind.items() if v),
        f"sync   " + ("up to date" if not pending else f"{pending} pending (+{len(cs.added)} ~{len(cs.changed)} -{len(cs.removed)})"),
    ]
    lines += [f"hot    {h['energy']:.3f}  {h['key']}" for h in payload["hottest"]]  # type: ignore[union-attr]
    _emit(args, payload, "\n".join(lines))
    return 0


def cmd_watch(args: argparse.Namespace) -> int:
    from .watch import RepoWatcher

    project = _project(args)
    encoder = project.build_encoder()  # loaded once; the store is opened per sync
    config = project.memory_config()

    def report(r) -> None:  # noqa: ANN001
        if r.added or r.changed or r.removed or not args.quiet:
            print(r.summary(), flush=True)

    watcher = RepoWatcher(config, lambda: project.open_manager(encoder), interval_s=args.interval,
                          debounce_s=args.debounce, on_sync=report)
    print(f"watching {project.repo_root} every {args.interval}s (debounce {args.debounce}s), Ctrl-C to stop", flush=True)
    try:
        watcher.run(max_polls=args.max_polls)
    except KeyboardInterrupt:
        pass
    return 0


def cmd_route(args: argparse.Namespace) -> int:
    from .router import GeometricRouter

    project = _project(args)
    with project.open_manager() as mgr:
        if not args.no_sync:
            r = mgr.sync()
            if (r.added or r.changed or r.removed) and not args.json:
                print(f"[zt] {r.summary()}", file=sys.stderr)
        ctx = GeometricRouter(mgr, project.router_config()).route_to_compute_engine(
            args.query, mode=args.mode, token_budget=args.budget)
    if args.engine:
        from .engine_client import make_engine_client
        from .project import ENGINE_DEFAULTS

        client = make_engine_client({**ENGINE_DEFAULTS, "backend": "openai", "url": args.engine, "model": args.model})
        answer = client.complete(ctx, max_tokens=args.max_new_tokens)
        print(answer)
        return 0
    if args.json:
        print(json.dumps({
            "mode": ctx.mode, "token_estimate": ctx.token_estimate, "token_budget": ctx.token_budget,
            "file_map": list(ctx.file_map), "messages": ctx.to_chat_messages(line_numbers=args.line_numbers),
            "blocks": [{"title": b.title, "kind": b.kind.name, "score": round(b.score, 4), "geo_rank": b.geo_rank,
                        "lex_rank": b.lex_rank, "tokens": b.tokens} for b in ctx.blocks],
        }, indent=2))
    else:
        print(ctx.render(line_numbers=args.line_numbers))
    return 0


def _engine_client(project: ProjectConfig, args: argparse.Namespace):  # noqa: ANN202
    """(client, settings, where) for [engine]; --engine URL overrides the URL of the OpenAI-compatible backend."""
    from .engine_client import make_engine_client

    eng = project.engine_settings()
    if getattr(args, "engine", None):
        eng["url"] = args.engine
    client = make_engine_client(eng, usage_log=project.engine_usage_log)
    where = eng["url"] if eng["backend"] == "openai" else f"bedrock {eng['model']} ({eng['region']})"
    return client, eng, where


def _require_engine(client, eng: dict, where: str) -> None:  # noqa: ANN001
    """health() costs nothing on either backend (no model call on Bedrock)."""
    if client.health():
        return
    if eng["backend"] == "bedrock":
        raise CliError(f"{where}: {getattr(client, 'last_error', None) or 'client unavailable'}")
    raise CliError(f"compute engine not reachable at {where}: {LLAMA_HINT}")


def _cpu_encoder(project: ProjectConfig, args: argparse.Namespace):  # noqa: ANN202
    """While the LLM occupies VRAM the navigator encoder must stay on CPU (an OOM there kills generation)."""
    if str(project.encoder.get("device", "cpu")).startswith("cuda") and not getattr(args, "allow_gpu_encoder", False):
        print("[zt] encoder forced to CPU while the compute engine owns the GPU (use --allow-gpu-encoder to override)", file=sys.stderr)
        project.encoder["device"] = "cpu"
    return project.build_encoder()


LLAMA_HINT = ("start it first, e.g.:\n  llama-server -m qwen2.5-coder-14b-instruct-q4_k_m.gguf -ngl 99 -fa on "
              "-ctk q8_0 -ctv q8_0 -c 16384 -np 1 --host 127.0.0.1 --port 8080")


def cmd_engine(args: argparse.Namespace) -> int:
    project = _project(args)
    client, eng, url = _engine_client(project, args)
    if eng["backend"] == "bedrock":
        return _engine_bedrock(args, project, client, eng)
    _require_engine(client, eng, url)
    n_ctx = client.context_size()
    sample = "def apply_discount(price: float, percent: float) -> float:\n    return price * (1 - percent / 100)\n"
    n_tok = client.count_tokens(sample)
    probe = client.chat_full([{"role": "user", "content": "Write a Python function that reverses a string."}],
                             max_tokens=int(args.probe_tokens), temperature=0.0)
    budget = (n_ctx or int(eng["ctx"])) - int(eng["max_new_tokens"]) - 128
    payload = {"url": url, "healthy": True, "served_ctx": n_ctx, "chars_per_token_sample": round(len(sample) / n_tok, 2) if n_tok else None,
               "probe_tokens": probe.completion_tokens, "probe_seconds": round(probe.seconds, 2),
               "tokens_per_second": round(probe.tokens_per_second, 1) if probe.tokens_per_second else None,
               "router_budget": budget}
    _emit(args, payload, "\n".join([
        f"engine   {url}  (healthy)",
        f"context  {n_ctx} tokens per slot" + ("" if n_ctx else " (unknown: not a llama-server?)"),
        f"speed    {payload['tokens_per_second']} tok/s ({probe.completion_tokens} tokens in {probe.seconds:.1f}s, includes prompt processing)",
        f"budget   router gets up to {budget} tokens (context - max_new_tokens {eng['max_new_tokens']} - margin)",
    ]))
    return 0


def _engine_bedrock(args: argparse.Namespace, project: ProjectConfig, client, eng: dict) -> int:  # noqa: ANN001
    """Configuration and model profile; a paid call only with --probe."""
    info = client.describe()
    healthy = client.health()
    max_new = int(eng["max_new_tokens"])
    budget = min(int(project.router_config().token_budget), int(client.context_size() or 16384) - max_new - 128)
    payload = {**info, "boto3": healthy, "unavailable": None if healthy else client.last_error,
               "max_new_tokens": max_new, "max_new_tokens_sent": client.output_limit(max_new), "router_budget": budget,
               "probe": None}
    lines = [
        f"backend  bedrock  model {info['model']}",
        f"profile  {info['profile']}: context {info['context']}, max output {info['max_output_tokens']}, "
        f"reasoning {'yes' if info['reasoning'] else 'no'}"
        + (f", extra fields {info['extra_fields']}" if info["extra_fields"] else ""),
        f"aws      region {info['region']}, proxy {info['proxy'] or '(HTTPS_PROXY or none)'}, credentials from the "
        f"environment (IAM role), timeout {info['timeout_s']:.0f}s, aws_retries {info['aws_retries']}",
        f"limits   max_calls {info['max_calls']} per process, max_new_tokens {max_new} (sent: {client.output_limit(max_new)}), "
        f"usage log {info['usage_log']}",
        f"budget   router gets up to {budget} tokens ([router] token_budget, within context - max_new_tokens - margin)",
        f"boto3    {'available' if healthy else 'NOT AVAILABLE: ' + str(client.last_error)}",
    ]
    if args.probe:
        if not healthy:
            raise CliError(str(client.last_error))
        t0 = time.perf_counter()
        answer = client.probe()
        payload["probe"] = {"answer": answer, "ok": answer.strip().strip(".").upper() == "OK",
                            "seconds": round(time.perf_counter() - t0, 2)}
        lines.append(f"probe    {answer!r} in {payload['probe']['seconds']}s (1 paid call)")
    else:
        lines.append("probe    not run (zt engine --probe makes one paid call)")
    _emit(args, payload, "\n".join(lines))
    return 0 if healthy else 1


def cmd_code(args: argparse.Namespace) -> int:
    from .coding import run_code_task
    from .editing import EditPlan
    from .sysinfo import current_rss_mib

    project = _project(args)
    client, eng, url = _engine_client(project, args)
    _require_engine(client, eng, url)
    max_new = int(args.max_new_tokens or eng["max_new_tokens"])

    def confirm(plan: EditPlan) -> bool:
        print(plan.diff() or "(no textual change)", file=sys.stderr if args.json else sys.stdout)
        if args.yes:
            return True
        try:
            return input("apply these changes? [y/N] ").strip().lower() in ("y", "yes", "s", "si")
        except EOFError:
            return False

    def echo(msg: str) -> None:
        if not args.json:
            print(msg, file=sys.stderr, flush=True)

    from .need import need_settings

    need = need_settings(eng, project.router_preset)
    encoder = _cpu_encoder(project, args)
    t0 = time.perf_counter()
    with project.open_manager(encoder) as mgr:
        res = run_code_task(
            mgr, client, args.instruction, router_config=project.router_config(), max_new_tokens=max_new,
            temperature=float(args.temperature if args.temperature is not None else eng["temperature"]),
            retries=int(args.retries if args.retries is not None else eng["retries"]), budget_cap=args.budget,
            default_ctx=int(eng["ctx"] or client.context_size() or 16384), dry_run=args.dry_run, confirm=None if args.dry_run else confirm,
            check_cmd=args.check, revert_on_fail=args.revert_on_fail, echo=echo, need=need,
        )
    elapsed = time.perf_counter() - t0
    if args.dry_run and res.plan is not None:
        print(res.plan.diff(), file=sys.stderr if args.json else sys.stdout)
    if args.show_answer or res.status in ("no_edits", "edit_failed"):
        print("----- model answer -----\n" + res.answer, file=sys.stderr)
    rss = current_rss_mib()
    payload = {
        "status": res.status, "attempts": res.attempts, "blocks": res.blocks, "context_tokens": res.context_tokens,
        "context_budget": res.context_budget, "served_ctx": res.served_ctx, "llm_seconds": round(res.llm_seconds, 2),
        "prompt_tokens": sum(r.prompt_tokens or 0 for r in res.llm),
        "completion_tokens": sum(r.completion_tokens or 0 for r in res.llm), "elapsed_seconds": round(elapsed, 2),
        "engine_calls": len(res.llm), "need_rounds": res.need_rounds, "need_requests": res.need_requests,
        "changed": list(res.applied.changed) if res.applied else [], "created": list(res.applied.created) if res.applied else [],
        "backup": res.applied.backup_id if res.applied else None,
        "heated": [{"symbol": k, "energy_before": round(b, 4), "energy_after": round(a, 4)} for k, b, a in res.heated],
        "check_returncode": res.check_returncode, "errors": res.errors, "rss_mib": round(rss) if rss else None,
    }
    lines = [f"status   {res.status}  ({res.attempts} attempt(s), {elapsed:.1f}s total, LLM {res.llm_seconds:.1f}s)"]
    if eng["backend"] == "bedrock":
        lines.append(f"bedrock  {len(res.llm)} call(s), {payload['prompt_tokens']} input + {payload['completion_tokens']} "
                     f"output tokens (log: {project.engine_usage_log})")
    if res.need_rounds:
        lines.append(f"need     {res.need_rounds} extra round(s): " + "; ".join(", ".join(r) for r in res.need_requests))
    if res.applied:
        lines.append(f"files    {', '.join([*res.applied.changed, *res.applied.created])}  (undo: zt undo)")
    lines += [f"heated   {k.split('::', 1)[-1]:40s} energy {b:.3f} -> {a:.3f}" for k, b, a in res.heated[:8]]
    if res.check_returncode is not None:
        tail = "\n".join(res.check_output.strip().splitlines()[-15:])
        lines.append(f"check    exit {res.check_returncode}\n{tail}")
    lines += [f"error    {e}" for e in res.errors[-5:]]
    _emit(args, payload, "\n".join(lines))
    return {"applied": 0, "dry_run": 0, "declined": 0, "no_edits": 4, "edit_failed": 4, "check_failed": 5, "reverted": 5}[res.status]


def cmd_undo(args: argparse.Namespace) -> int:
    from .editing import undo_last

    project = _project(args)
    undone = undo_last(project.repo_root)
    if undone is None:
        print("nothing to undo")
        return 0
    with project.open_manager(_cpu_encoder(project, args)) as mgr:
        report = mgr.ingest_files(undone[1])
    print(f"restored {', '.join(undone[1])} from backup {undone[0]} | sync: ~{report.chunks_updated} +{report.chunks_added} -{report.chunks_removed}")
    return 0


def cmd_mcp(args: argparse.Namespace) -> int:
    from .mcp_server import serve_stdio

    # --repo must exist; without it the project is looked up on each call from the session folder, so the
    # server also runs (and explains how to enable zt) in folders that have no index yet.
    from .mcp_server import project_names

    if args.project and args.repo:
        raise SystemExit("zt mcp: use either --repo or --project, not both")
    try:
        projects = project_names(args.project) if args.project else None
    except ValueError as exc:
        raise SystemExit(f"zt mcp: {exc}") from None
    project = _project(args) if args.repo else None
    if project is not None and args.device:
        project.encoder["device"] = args.device
    serve_stdio(project, default_budget=args.budget, device=args.device, preload=args.preload,
                idle_unload_s=args.idle_unload, max_inline_sync_files=args.max_inline_sync, default_mode=args.mode,
                read_span=args.read_span, projects=projects)
    return 0


def _hook_script(repo_root: Path, chained: Path | None) -> str:
    # as_posix(): Git for Windows runs hooks with its own sh, which wants C:/... rather than C:\...
    python, repo = Path(sys.executable).as_posix(), Path(repo_root).as_posix()
    cmd = f"{shlex.quote(python)} -m zt_memory.cli sync --quiet --lock-timeout 300 --repo {shlex.quote(repo)}"
    lines = ["#!/bin/sh", HOOK_MARKER]
    if chained is not None:
        lines.append(f'{shlex.quote(Path(chained).as_posix())} "$@" || exit $?')
    # Background + detached: never slow down or fail a git operation; the store lock serializes runs.
    lines.append(f"( {cmd} >/dev/null 2>&1 & )")
    return "\n".join(lines) + "\n"


def _hooks_dir(repo_root: Path) -> Path:
    try:
        out = subprocess.run(["git", "-C", str(repo_root), "rev-parse", "--git-path", "hooks"],
                             capture_output=True, text=True, check=True, timeout=10).stdout.strip()
    except (subprocess.SubprocessError, OSError) as exc:
        raise CliError(f"{repo_root} is not a git work tree") from exc
    path = Path(out)
    return path if path.is_absolute() else repo_root / path


def cmd_hook(args: argparse.Namespace) -> int:
    project = _project(args)
    hooks = _hooks_dir(project.repo_root)
    hooks.mkdir(parents=True, exist_ok=True)
    done: list[str] = []
    for name in HOOK_NAMES:
        path = hooks / name
        ours = path.exists() and HOOK_MARKER in path.read_text(errors="replace")
        if args.action == "uninstall":
            if ours:
                path.unlink()
                backup = path.with_name(name + ".pre-zt")
                if backup.exists():
                    backup.rename(path)
                done.append(name)
            continue
        chained = None
        if path.exists() and not ours:
            if not args.chain:
                raise CliError(f"{path} exists and is not managed by zt: rerun with --chain to keep it and call it first")
            chained = path.with_name(name + ".pre-zt")
            path.rename(chained)
        elif ours and path.with_name(name + ".pre-zt").exists():
            chained = path.with_name(name + ".pre-zt")
        path.write_text(_hook_script(project.repo_root, chained), encoding="utf-8", newline="\n")  # LF: sh rejects CRLF
        path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        done.append(name)
    print(f"{args.action}ed hooks: {', '.join(done) or 'none'} in {hooks}")
    return 0


# ------------------------------------------------------------------------------------------
# parser
# ------------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="zt", description="Zero-Token Physics-Guided Memory for local code LLMs")
    sub = ap.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--repo", help="repository root (default: nearest ancestor with .zt/config.toml)")
        p.add_argument("--lock-timeout", type=float, help="seconds to wait for a store held by another process")

    p = sub.add_parser("init", help="write .zt/config.toml for a repository")
    p.add_argument("--repo", help="repository root (default: cwd)")
    p.add_argument("--encoder", default="hashing", help="'hashing[:dim]' or 'hf:<model>[@rev]' (e.g. hf:Qwen/Qwen3-Embedding-0.6B)")
    p.add_argument("--rank", type=int, default=128)
    p.add_argument("--ram-mb", type=int, default=1536)
    p.add_argument("--budget", type=int, default=12_000, help="router token budget")
    p.add_argument("--store", help="store directory (default: <repo>/.zt/store)")
    p.add_argument("--projector", help="frozen SVD basis from examples/calibrate_encoder.py --save-projector")
    p.add_argument("--max-length", type=int)
    p.add_argument("--dtype", choices=["float32", "bfloat16", "float16"])
    p.add_argument("--engine-url", help="compute engine base URL stored in [engine] (default http://127.0.0.1:8080/v1)")
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_init)

    p = sub.add_parser("sync", help="incremental sync of the repository into the store")
    common(p)
    p.add_argument("--full", action="store_true", help="ignore the stat cache and re-chunk every file")
    p.add_argument("--json", action="store_true")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--device", help="encoder device for this run only, e.g. cuda (needs a CUDA build of torch)")
    p.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], help="encoder dtype for this run only")
    p.add_argument("--batch-size", type=int, help="encoder batch size for this run only (GPU: 32-64)")
    p.set_defaults(fn=cmd_sync)

    p = sub.add_parser("status", help="pending changes, store size, hottest nodes")
    common(p)
    p.add_argument("--hot", type=int, default=5)
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_status)

    p = sub.add_parser("watch", help="poll the tree and sync after edits settle")
    common(p)
    p.add_argument("--interval", type=float, default=2.0)
    p.add_argument("--debounce", type=float, default=1.5)
    p.add_argument("--max-polls", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--quiet", action="store_true", help="print only syncs that changed something")
    p.set_defaults(fn=cmd_watch)

    p = sub.add_parser("route", help="sync, then print the routed prompt (or send it to --engine)")
    common(p)
    p.add_argument("query")
    p.add_argument("--mode", choices=["auto", "brownfield", "greenfield"], default="auto")
    p.add_argument("--budget", type=int, default=None)
    p.add_argument("--line-numbers", action="store_true")
    p.add_argument("--no-sync", action="store_true")
    p.add_argument("--json", action="store_true")
    p.add_argument("--engine", help="OpenAI-compatible base URL, e.g. http://127.0.0.1:8080/v1")
    p.add_argument("--model", default="local")
    p.add_argument("--max-new-tokens", type=int, default=4096)
    p.set_defaults(fn=cmd_route)

    p = sub.add_parser("code", help="navigator -> LLM -> SEARCH/REPLACE patch -> backup -> apply -> sync")
    common(p)
    p.add_argument("instruction", help="what to change, in natural language (bug report, feature request)")
    p.add_argument("--engine", help="OpenAI-compatible base URL (default: [engine] url or http://127.0.0.1:8080/v1)")
    p.add_argument("--max-new-tokens", type=int)
    p.add_argument("--temperature", type=float)
    p.add_argument("--retries", type=int, help="re-ask the model with the real file text when a SEARCH does not match")
    p.add_argument("--budget", type=int, help="cap the routed context (tokens); default fits the served context")
    p.add_argument("--dry-run", action="store_true", help="show the diff, write nothing")
    p.add_argument("--yes", "-y", action="store_true", help="apply without asking")
    p.add_argument("--check", help="command run after applying, e.g. \"python -m pytest -q\"")
    p.add_argument("--revert-on-fail", action="store_true", help="undo the edit if --check fails")
    p.add_argument("--show-answer", action="store_true")
    p.add_argument("--allow-gpu-encoder", action="store_true")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_code)

    p = sub.add_parser("undo", help="restore the files changed by the last `zt code` and resync")
    common(p)
    p.set_defaults(fn=cmd_undo)

    p = sub.add_parser("engine", help="check the compute engine: health, served context, generation speed "
                                        "(Bedrock: configuration and model profile; --probe for one paid call)")
    common(p)
    p.add_argument("--engine")
    p.add_argument("--probe-tokens", type=int, default=64)
    p.add_argument("--probe", action="store_true", help="Bedrock: make one minimal paid call (\"Reply only with: OK\")")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_engine)

    p = sub.add_parser("mcp", help="serve this repository's memory to MCP clients over stdio (Claude Code, Cursor, Cline)")
    common(p)
    p.add_argument("--budget", type=int, help="token budget of zt_query_context in code mode (default 4000)")
    p.add_argument("--mode", choices=("code", "locations", "focus"), default="locations",
                   help="default zt_query_context mode: only locations + signatures, for agents that read files themselves "
                        "(default); the code spans; or focus: 5 whole bodies plus a map of signatures")
    p.add_argument("--project", action="append", metavar="[NAME=]PATH",
                   help="serve several projects from one server (repeatable; NAME defaults to the folder name): every "
                        "tool then takes `project`. For Claude Desktop, which has one entry per server and no session folder")
    p.add_argument("--read-span", action="store_true",
                   help="also offer zt_read_span (up to 3 line ranges read from disk, never excluded files): for "
                        "clients without their own file-reading tool, e.g. Claude Desktop")
    p.add_argument("--device", help="encoder device, e.g. cuda when no local LLM occupies the GPU (default: config)")
    p.add_argument("--preload", action="store_true", help="load the encoder at launch instead of on the first call")
    p.add_argument("--idle-unload", type=float, default=900.0,
                   help="seconds without calls before the encoder is released (0 = keep it loaded)")
    p.add_argument("--max-inline-sync", type=int, default=100,
                   help="refresh the index inside a tool call only when at most this many files changed")
    p.set_defaults(fn=cmd_mcp)

    p = sub.add_parser("hook", help="install/uninstall git hooks that run `zt sync` in the background")
    common(p)
    p.add_argument("action", choices=["install", "uninstall"])
    p.add_argument("--chain", action="store_true", help="keep existing hooks (renamed *.pre-zt) and call them first")
    p.set_defaults(fn=cmd_hook)
    return ap


def _tolerant_console() -> None:
    """On a non-UTF-8 console (Windows cp1252) unencodable characters in code or paths become '?' instead of
    aborting `zt route` with UnicodeEncodeError. Output that the old encoding could write is unchanged."""
    for stream in (sys.stdout, sys.stderr):
        encoding = (getattr(stream, "encoding", "") or "").lower().replace("-", "")
        if encoding != "utf8" and hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")


def main(argv: Sequence[str] | None = None) -> int:
    _tolerant_console()
    args = build_parser().parse_args(argv)
    try:
        return int(args.fn(args))
    except StoreLockedError as exc:
        print(f"zt: {exc}", file=sys.stderr)
        return 3
    except (CliError, ProjectConfigError) as exc:
        print(f"zt: {exc}", file=sys.stderr)
        return 1
    except ComputeEngineError as exc:
        print(f"zt: {exc}", file=sys.stderr)
        return 6


def sync_main(argv: Sequence[str] | None = None) -> int:
    """`zt-sync [--full] [--json] [--quiet] [--repo PATH]` == `zt sync ...`."""
    return main(["sync", *(sys.argv[1:] if argv is None else argv)])


if __name__ == "__main__":
    sys.exit(main())
