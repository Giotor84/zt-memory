"""Phase 2: does a large model use the map and NEED instead of guessing? (focused context + NEED vs today's context)

Three steps, because the cases come from public repositories and the model runs in an environment without internet:

  select  (with the clones, no model): for each localization case, index a snapshot of the parent commit, route it with
          the api preset (anti-clone, 5 bodies, 20 more signatures) and keep the cases where the changed lines are in the
          map but not in a body ("map-only") plus a few where they are in a body ("control"). Writes a self-contained
          pack: snapshots (no git history) + questions + changed lines.
  run     (where the model is, e.g. an air-gapped SageMaker environment + Bedrock): per case and arm, one task, max `--max-calls` paid calls:
            A  today's context, one call
            E  focused context + map, NEED offered (one extra round with a short new prompt: same for every model)
          Nothing is written to the snapshots: the edit plan is computed in memory and compared with the real fix.
  report  tables: edit on the changed lines, NEED asked / hit, tokens, calls, weighted cost (input + 4 x output);
          McNemar A vs E. The same command works for every model, so runs of different models compare directly.

    python examples/benchmark_need.py select --repos-dir DIR --loc-en EN.json --loc examples/data/loc_cases_it.json \\
        --pack need_pack.tar.gz --controls 6
    python examples/benchmark_need.py run --pack need_pack.tar.gz --workdir DIR --backend bedrock \\
        --model minimax.minimax-m2.5 --proxy http://... --max-calls 40 --limit 12 --sets commit
    python examples/benchmark_need.py report --workdir DIR
"""
from __future__ import annotations

import argparse
import difflib
import io
import json
import random
import shutil
import statistics
import subprocess
import sys
import tarfile
import tempfile
from dataclasses import replace
from pathlib import Path

from zt_memory.bench_bugfix import mcnemar_exact
from zt_memory.bench_localize import LocCase, _coverage
from zt_memory.coding import CODE_SYSTEM_PROMPT, request_edits
from zt_memory.config import MemoryConfig, NodeKind, RouterConfig
from zt_memory.editing import SEARCH_REPLACE_CONTRACT
from zt_memory.encoders import build_encoder
from zt_memory.engine_client import ComputeEngineError, make_engine_client
from zt_memory.manager import NewtonianMemoryManager
from zt_memory.need import NEED_CONTRACT, NeedSettings, NeedState, offer_need
from zt_memory.project import BEDROCK_DEFAULTS, ENGINE_DEFAULTS, ROUTER_PRESETS
from zt_memory.router import GeometricRouter

ENCODER = "hashing:1024"
RANK = 64
BUDGET = 12_000
ARM_CONFIGS = {"A": RouterConfig(reinforce=False), "E": replace(RouterConfig(reinforce=False), **ROUTER_PRESETS["api"])}
TOLERANCE = 2  # an edit within 2 lines of a changed line counts as "on the fix spot"
OUTPUT_WEIGHT = 4  # weighted cost = input + 4 x output tokens (output, reasoning included, is priced higher)


def _spans(ctx, *, body_only: bool) -> dict[str, list[tuple[int, int]]]:  # noqa: ANN001
    out: dict[str, list[tuple[int, int]]] = {}
    for b in ctx.blocks:
        if b.path and b.start_line is not None and b.kind in (NodeKind.CHUNK, NodeKind.FILE) and (b.full or not body_only):
            out.setdefault(b.path, []).append((int(b.start_line), int(b.end_line or b.start_line)))
    return out


def _category(ctx, gold) -> str:  # noqa: ANN001
    _, _, body = _coverage(_spans(ctx, body_only=True), gold)
    _, _, seen = _coverage(_spans(ctx, body_only=False), gold)
    return "body" if body else ("map-only" if seen else "missing")


def _manager(snapshot: Path, store: Path, encoder) -> NewtonianMemoryManager:  # noqa: ANN001
    return NewtonianMemoryManager(MemoryConfig(storage_dir=store, repo_root=snapshot, rank=RANK, calibration_samples=1024), encoder)


# -- select ---------------------------------------------------------------------------------------
def cmd_select(args: argparse.Namespace) -> int:
    items: dict[tuple[str, str], dict] = {}
    for path, qset in ((args.loc_en, "commit"), (args.loc, "italiano")):
        if not path:
            continue
        for name, cases in json.loads(Path(path).read_text(encoding="utf-8")).items():
            for d in cases:
                c = LocCase.from_dict(d)
                item = items.setdefault((name, c.id), {"repo": name, "id": c.id, "base": c.base, "gold": c.gold, "queries": {}})
                item["queries"][qset] = c.query
    encoder = build_encoder(ENCODER)
    chosen: list[dict] = []
    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp) / "pack"
        (staging / "snapshots").mkdir(parents=True)
        for (name, cid), item in sorted(items.items()):
            snap = staging / "snapshots" / f"{name}_{cid}"
            archive = subprocess.run(["git", "-C", str(args.repos_dir / name), "archive", "--format=tar", item["base"]],
                                     capture_output=True, check=True).stdout
            with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
                tar.extractall(snap, filter="data")
            store = Path(tmp) / "stores" / f"{name}_{cid}"
            with _manager(snap, store, encoder) as mgr:
                mgr.sync()
                for qset, query in item["queries"].items():
                    ctx = GeometricRouter(mgr, ARM_CONFIGS["E"]).route_to_compute_engine(query, token_budget=BUDGET)
                    item.setdefault("category", {})[qset] = _category(ctx, item["gold"])
            print(f"[select] {name} {cid}: {item['category']}", flush=True)
        rng = random.Random(args.seed)
        keep_snaps: set[str] = set()
        for qset in ("commit", "italiano"):
            pool = [it for it in items.values() if qset in it.get("category", {})]
            map_only = [it for it in pool if it["category"][qset] == "map-only"]
            controls = [it for it in pool if it["category"][qset] == "body"]
            rng.shuffle(controls)
            for it in map_only + controls[: args.controls]:
                chosen.append({"set": qset, "repo": it["repo"], "id": it["id"], "query": it["queries"][qset], "gold": it["gold"],
                               "selected_as": it["category"][qset], "snapshot": f"snapshots/{it['repo']}_{it['id']}"})
                keep_snaps.add(f"{it['repo']}_{it['id']}")
            print(f"[select] {qset}: {len(map_only)} map-only + {min(len(controls), args.controls)} controls")
        for snap in (staging / "snapshots").iterdir():
            if snap.name not in keep_snaps:
                shutil.rmtree(snap)
        rng.shuffle(chosen)  # --limit then takes a mixed sample
        chosen.sort(key=lambda c: c["selected_as"] != "map-only")  # map-only cases first
        (staging / "cases.json").write_text(json.dumps(chosen, indent=1, ensure_ascii=False), encoding="utf-8")
        with tarfile.open(args.pack, "w:gz") as tar:
            tar.add(staging, arcname="need_pack")
    print(f"[select] {len(chosen)} task(s) -> {args.pack} ({Path(args.pack).stat().st_size / 2**20:.1f} MiB)")
    return 0


# -- run ------------------------------------------------------------------------------------------
def touched_lines(snapshot: Path, plan) -> dict[str, list[int]]:  # noqa: ANN001
    """Base-side lines changed by an edit plan (insertions mark the line they follow)."""
    out: dict[str, list[int]] = {}
    for rel, new in plan.new_contents.items():
        old_path = snapshot / rel
        old = old_path.read_text(encoding="utf-8", errors="replace").splitlines() if old_path.exists() else []
        lines: set[int] = set()
        for tag, i1, i2, _, _ in difflib.SequenceMatcher(None, old, new.splitlines(), autojunk=False).get_opcodes():
            if tag != "equal":
                lines.update(range(i1 + 1, max(i2, i1 + 1) + 1))
        out[rel] = sorted(lines)
    return out


def on_gold(touched: dict[str, list[int]], gold: dict[str, list[int]]) -> bool:
    return all(any(abs(t - g) <= TOLERANCE for t in touched.get(rel, []) for g in lines) for rel, lines in gold.items())


def cmd_run(args: argparse.Namespace) -> int:
    wd = Path(args.workdir)
    wd.mkdir(parents=True, exist_ok=True)
    pack_dir = wd / "need_pack"
    if not pack_dir.exists():
        with tarfile.open(args.pack) as tar:
            tar.extractall(wd, filter="data")
    tasks = pick_tasks(json.loads((pack_dir / "cases.json").read_text(encoding="utf-8")), args.sets.split(","), args.limit)
    out = wd / "need_results.jsonl"
    model = args.model or "local"
    done = {(r.get("model", ""), r["set"], r["id"], r["arm"]) for r in _load(out)}
    settings = {**ENGINE_DEFAULTS, **(BEDROCK_DEFAULTS if args.backend == "bedrock" else {}), "backend": args.backend,
                "url": args.engine, "model": args.model or "local", "region": args.region, "proxy": args.proxy,
                "max_calls": args.max_calls, "timeout_s": args.timeout}
    client = make_engine_client(settings, usage_log=wd / "engine_usage.jsonl")
    if not client.health():
        print(f"engine not available: {getattr(client, 'last_error', None) or args.engine}")
        return 6
    need = NeedSettings(rounds=1)
    by_kind = {k: sum(t["selected_as"] == k for t in tasks) for k in ("map-only", "body")}
    print(f"{len(tasks)} task(s) ({by_kind['map-only']} map-only, {by_kind['body']} control) x arms {args.arms} | "
          f"model {args.model} | max_calls {args.max_calls}")
    encoder = build_encoder(ENCODER)
    max_new = int(args.max_new_tokens or settings["max_new_tokens"])
    try:
        for t in tasks:
            arms = [a for a in args.arms.split(",") if (model, t["set"], t["id"], a) not in done]
            if not arms:
                continue
            snap = pack_dir / t["snapshot"]
            with _manager(snap, wd / "stores" / Path(t["snapshot"]).name, encoder) as mgr:
                mgr.sync()
                for arm in arms:
                    ctx = GeometricRouter(mgr, ARM_CONFIGS[arm]).route_to_compute_engine(t["query"], token_budget=BUDGET)
                    state = None
                    contract = SEARCH_REPLACE_CONTRACT
                    if arm == "E" and offer_need(ctx, need):
                        contract += "\n\n" + NEED_CONTRACT
                        state = NeedState(NeedSettings(**vars(need)), snap, ctx, t["query"], CODE_SYSTEM_PROMPT, SEARCH_REPLACE_CONTRACT)
                    messages = ctx.to_chat_messages(CODE_SYSTEM_PROMPT, contract=contract)
                    got = request_edits(client, messages, snap, max_new_tokens=max_new, temperature=0.0, retries=0, need=state)
                    touched = touched_lines(snap, got.plan) if got.plan is not None and got.plan.ok else {}
                    asked = [s for r in got.need_requests for s in r]
                    row = {
                        "model": model, "set": t["set"], "repo": t["repo"], "id": t["id"], "arm": arm, "selected_as": t["selected_as"],
                        "category": _category(ctx, t["gold"]), "context_tokens": ctx.token_estimate,
                        "calls": got.attempts, "input_tokens": sum(r.prompt_tokens or 0 for r in got.llm),
                        "output_tokens": sum(r.completion_tokens or 0 for r in got.llm),
                        "seconds": round(sum(r.seconds for r in got.llm), 1), "plan_ok": bool(got.plan and got.plan.ok),
                        "edit_on_gold": bool(touched) and on_gold(touched, t["gold"]), "need_offered": state is not None,
                        "need_rounds": got.need_rounds, "need_asked": asked, "need_hit": _need_hit(asked, t["gold"]),
                        "errors": got.errors[-2:],
                    }
                    with out.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(row) + "\n")
                    print(f"[{t['set']} {t['repo']} {t['id']} {arm}] {row['category']:8s} calls {row['calls']} "
                          f"in {row['input_tokens']} out {row['output_tokens']} | plan {'ok' if row['plan_ok'] else 'no'} | "
                          f"on fix {'YES' if row['edit_on_gold'] else 'no'} | need {asked or '-'}", flush=True)
    except ComputeEngineError as exc:
        if not getattr(exc, "stop_run", False):
            raise
        print(f"\nstopped: {exc}\nresults kept in {out}; run again to continue")
    print("\n" + report(_load(out)))
    return 0


def pick_tasks(tasks: list[dict], sets: list[str], limit: int) -> list[dict]:
    """Per question set: all tasks, or `limit` of them with 2/3 map-only and 1/3 controls (8 + 4 for 12), in pack order
    (shuffled with a fixed seed when the pack was made), so every run of every model sees the same tasks."""
    out: list[dict] = []
    for qset in sets:
        mine = [t for t in tasks if t["set"] == qset]
        if not limit:
            out += mine
            continue
        maps = [t for t in mine if t["selected_as"] == "map-only"]
        controls = [t for t in mine if t["selected_as"] == "body"]
        n_ctrl = min(len(controls), limit // 3)
        n_map = min(len(maps), limit - n_ctrl)
        n_ctrl = min(len(controls), limit - n_map)  # fewer map-only cases than asked: fill with controls
        out += maps[:n_map] + controls[:n_ctrl]
    return out


def _need_hit(asked: list[str], gold: dict[str, list[int]]) -> bool:
    for label in asked:
        path, _, rng = label.rpartition(":L")
        start, _, end = rng.partition("-")
        if path in gold and any(int(start) - TOLERANCE <= g <= int(end) + TOLERANCE for g in gold[path]):
            return True
    return False


def _load(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()] if path.exists() else []


def report(rows: list[dict]) -> str:
    """One section per model and question set (runs of several models can share a workdir)."""
    lines: list[str] = []
    for model in sorted({r.get("model", "") for r in rows}):
        lines += [f"## Modello {model or '(senza nome)'}", ""]
        lines.append(_report_model([r for r in rows if r.get("model", "") == model]))
    return "\n".join(lines) or "no complete A/E pairs yet"


def _report_model(rows: list[dict]) -> str:
    by = {(r["set"], r["id"], r["arm"]): r for r in rows}
    lines = []
    for qset in sorted({r["set"] for r in rows}):
        ids = sorted({r["id"] for r in rows if r["set"] == qset and (qset, r["id"], "A") in by and (qset, r["id"], "E") in by})
        if not ids:
            continue
        lines += [f"### {qset} — {len(ids)} compiti (categoria di E al momento della prova)", "",
                  "| gruppo | casi | braccio | modifica sul punto giusto | patch applicabile | NEED chiesto / centrato | chiamate | "
                  f"token in / out (medi) | costo pesato in + {OUTPUT_WEIGHT}×out (medio) |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for group in ("map-only", "body", "missing"):
            gids = [i for i in ids if by[(qset, i, "E")]["category"] == group]
            if not gids:
                continue
            for arm in ("A", "E"):
                rs = [by[(qset, i, arm)] for i in gids]
                need = (f"{sum(r['need_rounds'] > 0 for r in rs)} / {sum(r['need_hit'] for r in rs)}" if arm == "E" else "-")
                lines.append(f"| {group} | {len(gids)} | {arm} | {sum(r['edit_on_gold'] for r in rs)} | {sum(r['plan_ok'] for r in rs)} | "
                             f"{need} | {statistics.mean(r['calls'] for r in rs):.2f} | "
                             f"{statistics.mean(r['input_tokens'] for r in rs):.0f} / {statistics.mean(r['output_tokens'] for r in rs):.0f} | "
                             f"{statistics.mean(_cost(r) for r in rs):.0f} |")
        a = [by[(qset, i, "A")] for i in ids]
        e = [by[(qset, i, "E")] for i in ids]
        won = sum(y["edit_on_gold"] and not x["edit_on_gold"] for x, y in zip(a, e))
        lost = sum(x["edit_on_gold"] and not y["edit_on_gold"] for x, y in zip(a, e))
        tin = [sum(r["input_tokens"] for r in g) for g in (a, e)]
        tout = [sum(r["output_tokens"] for r in g) for g in (a, e)]
        cost = [sum(_cost(r) for r in g) for g in (a, e)]
        lines += ["", f"- E contro A sulla modifica nel punto giusto: {won} vinti, {lost} persi (McNemar p = {mcnemar_exact(won, lost):.3g})",
                  f"- token totali in ingresso: A {tin[0]}, E {tin[1]} ({100 * (tin[1] / tin[0] - 1):+.0f}%); "
                  f"in uscita: A {tout[0]}, E {tout[1]} ({100 * (tout[1] / tout[0] - 1) if tout[0] else 0:+.0f}%)",
                  f"- costo pesato (in + {OUTPUT_WEIGHT}×out): A {cost[0]}, E {cost[1]} ({100 * (cost[1] / cost[0] - 1) if cost[0] else 0:+.0f}%)", ""]
    return "\n".join(lines) or "no complete A/E pairs yet"


def _cost(r: dict) -> int:
    return int(r["input_tokens"]) + OUTPUT_WEIGHT * int(r["output_tokens"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--repos-dir", type=Path, required=True)
    s.add_argument("--loc", type=Path, help="Italian questions (examples/data/loc_cases_it.json)")
    s.add_argument("--loc-en", type=Path, help="commit-message questions")
    s.add_argument("--pack", type=Path, required=True)
    s.add_argument("--controls", type=int, default=6, help="cases with the fix spot already in a body, per question set")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_select)
    r = sub.add_parser("run")
    r.add_argument("--pack", type=Path, required=True)
    r.add_argument("--workdir", type=Path, required=True)
    r.add_argument("--sets", default="commit,italiano")
    r.add_argument("--arms", default="A,E")
    r.add_argument("--limit", type=int, default=0, help="per question set: N tasks, 2/3 map-only + 1/3 controls; 0 = all")
    r.add_argument("--backend", choices=("openai", "bedrock"), default="bedrock")
    r.add_argument("--engine", default="http://127.0.0.1:8080/v1")
    r.add_argument("--model")
    r.add_argument("--region", default="eu-central-1")
    r.add_argument("--proxy", default="")
    r.add_argument("--max-calls", type=int, default=40)
    r.add_argument("--max-new-tokens", type=int)
    r.add_argument("--timeout", type=float, default=900.0)
    r.set_defaults(fn=cmd_run)
    p = sub.add_parser("report")
    p.add_argument("--workdir", type=Path, required=True)
    p.set_defaults(fn=lambda a: print(report(_load(Path(a.workdir) / "need_results.jsonl"))) or 0)
    args = ap.parse_args()
    return int(args.fn(args))


if __name__ == "__main__":
    sys.exit(main())
