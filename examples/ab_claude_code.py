"""A/B test: Claude Code fixing real bugs with and without the zt-memory MCP server.

Each task is a bug mined by `benchmark_bugfix.py mine` (its tests fail before the fix and pass after). The agent gets
the failing pytest output and must fix the source. Both arms run the same headless Claude Code (`claude -p`), same
model, same built-in tools, on a fresh copy of the repository at the parent commit **without git history** (so the
fix cannot be looked up). Arm `zt` also has the zt-memory MCP server and one system-prompt line saying to use it
first - what a user's memory/CLAUDE.md note would say - with answers of up to 12000 tokens of code. Arm `zt-4k`
caps them at 4000 tokens; arm `zt-loc` returns only locations (file, lines, signature, first docstring line,
~500 tokens) and lets the agent open the code itself. Arm `zt-focus` returns the 5 most relevant bodies whole plus a
map of signatures (`mode="focus"`, the router's `api` preset); arm `zt-loc-span` is `zt-loc` plus the `zt_read_span`
tool (line ranges read by the server, the NEED round over MCP). Grading restores the fix's test files and runs them.

    python examples/ab_claude_code.py run --workdir DIR --cases 12 CASES_JSON [CASES_JSON ...]
    python examples/ab_claude_code.py run --workdir DIR --descriptions examples/data/ab_vague_it.json --arms plain,zt-4k CASES...
    python examples/ab_claude_code.py report --workdir DIR

With `--descriptions` the task is a bug report in words (no log) and the fix's tests stay hidden until grading.

Measured per run: tests passing, cost (USD, as reported by Claude Code), API turns, tokens read by the model
(input + cache), output tokens, the largest single context, tool calls, wall time.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import random
import shutil
import statistics
import subprocess
import sys
import tarfile
import time
from pathlib import Path

from zt_memory.bench_bugfix import BugCase, load_cases, mcnemar_exact, run_tests

ARMS = ("plain", "zt", "zt-4k", "zt-loc", "zt-focus", "zt-loc-span")
ZT_SERVER_ARGS = {"zt": ["--budget", "12000"], "zt-4k": ["--budget", "4000"], "zt-loc": ["--mode", "locations"],
                  "zt-focus": ["--mode", "focus"], "zt-loc-span": ["--mode", "locations", "--read-span"]}
PROMPT = (
    "In questo repository alcuni test falliscono. Output di pytest:\n\n```\n{log}\n```\n\n"
    "Trova la causa e correggi il codice sorgente. Non modificare i file di test. "
    "Alla fine scrivi una sola riga con il file e la funzione che hai corretto."
)
VAGUE_PROMPT = (
    "In questo progetto c'e' un bug: {text}\n\n"
    "Trova la causa e correggi il codice sorgente. "
    "Alla fine scrivi una sola riga con il file e la funzione che hai corretto."
)
ZT_HINT = (
    "Questo progetto ha zt-memory attivo: per trovare il codice rilevante usa prima lo strumento MCP "
    "zt_query_context (passagli l'errore o la descrizione del problema), poi leggi o modifica i file indicati."
)
ZT_HINT_SPAN = " Per leggere le righe indicate da zt puoi usare zt_read_span invece di aprire file interi."
BUILTIN_TOOLS = "Bash,Read,Edit,Write,Glob,Grep"
ZT_TOOLS = ("mcp__zt-memory__zt_query_context", "mcp__zt-memory__zt_status", "mcp__zt-memory__zt_record_decision",
            "mcp__zt-memory__zt_read_span")
# Variables that would tie the child session to the calling one (session id, extra directories).
_ENV_DROP = ("CLAUDECODE", "CLAUDE_CODE_SESSION_ID", "CLAUDE_ADDITIONAL_DIRECTORIES", "CLAUDE_CODE_ADDITIONAL_DIRECTORIES_CLAUDE_MD")
MAX_LOG_CHARS = 8000


def _git(cwd: Path, *args: str, binary: bool = False) -> bytes | str:
    out = subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, check=True).stdout
    return out if binary else out.decode("utf-8", errors="replace")


def prepare(case: BugCase, repo: Path, dest: Path, *, with_zt: bool, show_tests: bool = True) -> Path:
    """Parent commit (+ the fix's test files if `show_tests`), as a one-commit repository (no history to read the fix from)."""
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    archive = _git(repo, "archive", "--format=tar", case.base, binary=True)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(dest, filter="data")
    for rel in case.test_files if show_tests else ():
        (dest / rel).parent.mkdir(parents=True, exist_ok=True)
        (dest / rel).write_bytes(_git(repo, "show", f"{case.fix}:{rel}", binary=True))
    _git(dest, "init", "-q")
    _git(dest, "add", "-A")
    _git(dest, "-c", "user.name=ab", "-c", "user.email=ab@example.invalid", "commit", "-qm", "base")
    if with_zt:
        for args in (["init", "--repo", str(dest)], ["sync", "--repo", str(dest), "--quiet"]):
            subprocess.run([sys.executable, "-m", "zt_memory.cli", *args], check=True, capture_output=True)
    return dest


def run_agent(root: Path, prompt: str, *, with_zt: bool, transcript: Path, model: str, max_turns: int,
              max_budget_usd: float, timeout_s: float, zt_args: tuple[str, ...] = (), forbidden: tuple[str, ...] = ()) -> dict:
    cmd = [shutil.which("claude") or "claude", "-p", prompt, "--output-format", "stream-json", "--verbose",
           "--model", model, "--max-turns", str(max_turns), "--max-budget-usd", str(max_budget_usd),
           "--no-session-persistence", "--setting-sources", "project", "--strict-mcp-config",
           "--tools", BUILTIN_TOOLS, "--permission-mode", "bypassPermissions",
           "--allowedTools", *BUILTIN_TOOLS.split(","), *(ZT_TOOLS if with_zt else ())]
    if with_zt:
        cfg = {"mcpServers": {"zt-memory": {"command": sys.executable,
                                            "args": ["-m", "zt_memory.cli", "mcp", "--repo", str(root), *zt_args]}}}
        cfg_path = transcript.with_suffix(".mcp.json")
        cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
        cmd += ["--mcp-config", str(cfg_path), "--append-system-prompt", ZT_HINT + (ZT_HINT_SPAN if "--read-span" in zt_args else "")]
    env = {k: v for k, v in os.environ.items() if k not in _ENV_DROP}
    env["IS_SANDBOX"] = "1"  # bypassPermissions as root inside a throwaway container
    t = time.perf_counter()
    try:
        proc = subprocess.run(cmd, cwd=str(root), capture_output=True, text=True, encoding="utf-8", errors="replace",
                              timeout=timeout_s, env=env)
        stdout, stderr, rc = proc.stdout, proc.stderr, proc.returncode
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr, rc = "timeout", 124
    transcript.write_text(stdout, encoding="utf-8")
    stats = parse_transcript(stdout, forbidden)
    stats.update(exit_code=rc, wall_s=round(time.perf_counter() - t, 1), stderr_tail=stderr.strip()[-300:])
    return stats


def parse_transcript(stream: str, forbidden: tuple[str, ...] = ()) -> dict:
    tools: dict[str, int] = {}
    zt_modes: dict[str, int] = {}  # mode passed to zt_query_context ("server" = left to the server default)
    seen_msgs: set[str] = set()
    peak = 0
    read_paths: set[str] = set()
    outside = 0
    result: dict = {}
    for line in stream.splitlines():
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("type") == "result":
            result = ev
        if ev.get("type") != "assistant":
            continue
        msg = ev.get("message", {})
        mid = msg.get("id")
        u = msg.get("usage") or {}
        if mid and mid not in seen_msgs:
            seen_msgs.add(mid)
            peak = max(peak, int(u.get("input_tokens", 0)) + int(u.get("cache_read_input_tokens", 0))
                       + int(u.get("cache_creation_input_tokens", 0)))
        for block in msg.get("content", []) or []:
            if block.get("type") != "tool_use":
                continue
            name = block.get("name", "?")
            tools[name] = tools.get(name, 0) + 1
            raw = json.dumps(block.get("input", {}))
            if name == "Read":
                read_paths.add(str(block.get("input", {}).get("file_path", "")))
            if name.endswith("zt_query_context"):
                mode = str(block.get("input", {}).get("mode") or "server")
                zt_modes[mode] = zt_modes.get(mode, 0) + 1
            if any(f in raw for f in forbidden):  # the benchmark's own clone, with the fix in its history
                outside += 1
    u = result.get("usage") or {}
    return {
        "ok": result.get("subtype") == "success" and not result.get("is_error", False),
        "cost_usd": float(result.get("total_cost_usd") or 0.0),
        "turns": int(result.get("num_turns") or 0),
        "tokens_read": int(u.get("input_tokens", 0)) + int(u.get("cache_read_input_tokens", 0)) + int(u.get("cache_creation_input_tokens", 0)),
        "tokens_out": int(u.get("output_tokens", 0)),
        "peak_context": peak,
        "tools": tools,
        "zt_modes": zt_modes,
        "files_read": len(read_paths),
        "outside_access": outside,
        "answer": str(result.get("result", ""))[-300:],
    }


def grade(case: BugCase, repo: Path, root: Path) -> tuple[bool, list[str]]:
    changed = [ln[3:] for ln in _git(root, "status", "--porcelain", "--untracked-files=all").splitlines() if ln.strip()]
    changed = [c for c in changed if not c.startswith(".zt/")]
    for rel in case.test_files:  # the agent may have edited tests: grade against the fix's own tests
        (root / rel).write_bytes(_git(repo, "show", f"{case.fix}:{rel}", binary=True))
    rc, _ = run_tests(root, case.test_files, timeout_s=300)
    return rc == 0, changed


def pick(case_files: list[Path], n: int, seed: int) -> list[tuple[str, Path, BugCase]]:
    pools = []
    for f in case_files:
        repo, cases = load_cases(f)
        pools.append([(f.parent.name, repo, c) for c in cases])
    rng = random.Random(seed)
    for pool in pools:
        rng.shuffle(pool)
    out: list[tuple[str, Path, BugCase]] = []
    while len(out) < n and any(pools):  # round-robin across repositories
        for pool in pools:
            if pool and len(out) < n:
                out.append(pool.pop())
    return out


def cmd_run(args: argparse.Namespace) -> int:
    wd: Path = args.workdir
    wd.mkdir(parents=True, exist_ok=True)
    out = wd / "ab_results.jsonl"
    done = {(r["repo"], r["case"], r["arm"]) for r in load(out)}
    arms_run = tuple(a for a in args.arms.split(",") if a)
    unknown = [a for a in arms_run if a not in ARMS]
    if unknown:
        raise SystemExit(f"unknown arms {unknown}; choose from {ARMS}")
    descriptions = {}
    if args.descriptions:
        descriptions = {k: v for k, v in json.loads(args.descriptions.read_text(encoding="utf-8")).items() if not k.startswith("_")}
    tasks = pick(args.cases, 10_000 if descriptions else args.n, args.seed)
    if descriptions:
        tasks = [t for t in tasks if f"{t[0]}/{t[2].id}" in descriptions][: args.n]
    shard, shards = (int(x) for x in args.shard.split("/"))
    tasks = [t for i, t in enumerate(tasks) if i % shards == shard]  # parallel workers on one workdir
    spent = sum(r["cost_usd"] for r in load(out))
    for i, (name, repo, case) in enumerate(tasks):
        k = (i * shards + shard) % len(arms_run)
        arms = arms_run[k:] + arms_run[:k]  # rotate which arm runs first
        for arm in arms:
            if (name, case.id, arm) in done:
                continue
            if spent >= args.total_budget_usd:
                print(f"[ab] total budget {args.total_budget_usd} USD reached, stopping", flush=True)
                return 0
            run_dir = wd / "runs" / f"{name}_{case.id}_{arm}"
            with_zt = arm in ZT_SERVER_ARGS
            text = descriptions.get(f"{name}/{case.id}")
            root = prepare(case, repo, run_dir / "repo", with_zt=with_zt, show_tests=text is None)
            prompt = VAGUE_PROMPT.format(text=text) if text else PROMPT.format(log=case.failure_excerpt[-MAX_LOG_CHARS:])
            forbidden = tuple(dict.fromkeys([str(repo), str(repo.resolve())]))
            stats = run_agent(root, prompt, with_zt=with_zt, transcript=run_dir / "transcript.jsonl", model=args.model,
                              max_turns=args.max_turns, max_budget_usd=args.max_run_usd, timeout_s=args.timeout,
                              zt_args=tuple(ZT_SERVER_ARGS.get(arm, ())), forbidden=forbidden)
            resolved, changed = grade(case, repo, root)
            gold_edited = any(c in case.source_files for c in changed)
            row = {"repo": name, "case": case.id, "arm": arm, "resolved": resolved, "gold_edited": gold_edited,
                   "changed": changed, **stats}
            with out.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")
            spent += row["cost_usd"]
            print(f"[ab {name} {case.id} {arm}] resolved={resolved} cost=${row['cost_usd']:.3f} turns={row['turns']} "
                  f"read={row['tokens_read']} peak={row['peak_context']} tools={row['tools']} total=${spent:.2f}", flush=True)
            if not args.keep:
                shutil.rmtree(run_dir / "repo", ignore_errors=True)
    print(report(load(out)))
    return 0


def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def report(rows: list[dict]) -> str:
    by = {(r["repo"], r["case"], r["arm"]): r for r in rows}
    present = [a for a in ARMS if any(r["arm"] == a for r in rows)]
    cases = sorted({(r["repo"], r["case"]) for r in rows if all((r["repo"], r["case"], a) in by for a in present)})
    if "plain" not in present or len(present) < 2 or not cases:
        return "no complete pairs"
    n = len(cases)
    arms = present
    head = " | ".join(arms)
    lines = [f"## Claude Code A/B - {n} bugs, same model, with and without zt-memory", "",
             "Each cell: mean / median per bug. Ratio: median over bugs of (zt arm / plain).", "",
             f"| measure | {head} | " + " | ".join(f"{a} / plain" for a in arms[1:]) + " |",
             "|---|" + "---|" * (2 * len(arms) - 1)]
    resolved = {a: sum(by[(*c, a)]["resolved"] for c in cases) for a in arms}
    lines.append("| bugs fixed (tests pass) | " + " | ".join(f"{resolved[a]}/{n}" for a in arms) + " |" + " |" * (len(arms) - 1))
    for key, label in (("cost_usd", "cost per bug (USD)"), ("tokens_read", "tokens read by the model"),
                       ("tokens_out", "output tokens"), ("peak_context", "largest single context"),
                       ("turns", "API turns"), ("files_read", "files opened with Read"), ("wall_s", "wall time (s)")):
        vals = {a: [float(by[(*c, a)][key]) for c in cases] for a in arms}
        cells = [f"{statistics.mean(vals[a]):.3g} / {statistics.median(vals[a]):.3g}" for a in arms]
        for a in arms[1:]:
            ratios = [z / p for p, z in zip(vals["plain"], vals[a]) if p > 0]
            cells.append(f"{statistics.median(ratios):.2f}" if ratios else "-")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.append("")
    for a in arms[1:]:
        cheaper = sum(by[(*c, a)]["cost_usd"] < by[(*c, "plain")]["cost_usd"] for c in cases)
        wins = sum(by[(*c, a)]["resolved"] and not by[(*c, "plain")]["resolved"] for c in cases)
        losses = sum(by[(*c, "plain")]["resolved"] and not by[(*c, a)]["resolved"] for c in cases)
        calls = [by[(*c, a)]["tools"].get("mcp__zt-memory__zt_query_context", 0) for c in cases]
        spans = [by[(*c, a)]["tools"].get("mcp__zt-memory__zt_read_span", 0) for c in cases]
        lines.append(f"- {a}: cheaper than plain on {cheaper}/{n} bugs (sign test p = {mcnemar_exact(cheaper, n - cheaper):.3g}); "
                     f"fixed only with {a}: {wins}, only without: {losses}; zt_query_context calls per bug "
                     f"{statistics.mean(calls):.1f} (never called on {calls.count(0)})"
                     + (f"; zt_read_span calls per bug {statistics.mean(spans):.1f} (used on {n - spans.count(0)})" if any(spans) else ""))
    lines.append(f"- runs that touched the benchmark's own clone (contamination): {sum(r['outside_access'] > 0 for r in rows)}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("cases", nargs="+", type=Path)
    r.add_argument("--workdir", type=Path, required=True)
    r.add_argument("--cases", dest="n", type=int, default=12, help="number of bugs (round-robin across the files)")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--model", default="claude-sonnet-5-5")
    r.add_argument("--max-turns", type=int, default=40)
    r.add_argument("--max-run-usd", type=float, default=2.0)
    r.add_argument("--total-budget-usd", type=float, default=25.0)
    r.add_argument("--timeout", type=float, default=1200.0)
    r.add_argument("--keep", action="store_true", help="keep each run's repository copy")
    r.add_argument("--arms", default=",".join(ARMS), help=f"comma-separated subset of {ARMS}")
    r.add_argument("--descriptions", type=Path, help="JSON {'repo/case_id': bug report in words}: no log, tests hidden")
    r.add_argument("--shard", default="0/1", help="I/N: run only every N-th bug starting at I (parallel workers)")
    p = sub.add_parser("report")
    p.add_argument("--workdir", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.cmd == "report":
        print(report(load(args.workdir / "ab_results.jsonl")))
        return 0
    return cmd_run(args)


if __name__ == "__main__":
    sys.exit(main())
