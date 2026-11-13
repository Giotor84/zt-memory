"""Focus phase 1b: where is the right block in today's ranking, and can an adaptive focus keep it whole?

Same cases and metric as benchmark_focus.py (body / map coverage of the lines the real fix changed), one pass, every
variant on the same index state. For A (today) and B (anti-clone) it also records, per case, the position of the
first chosen block covering each changed file (raw, and counting only "production" chunks as the focus counter
does), the score of that block relative to the best, and how flat the top-5 scores are.

    python examples/benchmark_focus_adaptive.py --workdir DIR --repos-dir DIR --loc ... --loc-en ... --traceback ...
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import replace
from pathlib import Path

from benchmark_focus import BASE, LAMBDA, MAX_QUERY_CHARS, spans  # noqa: E402 - examples folder on sys.path
from zt_memory.bench_bugfix import Worktree, load_cases, mcnemar_exact
from zt_memory.bench_localize import LocCase, _added_file, _coverage, changed_lines
from zt_memory.config import MemoryConfig, NodeKind
from zt_memory.encoders import build_encoder
from zt_memory.manager import NewtonianMemoryManager
from zt_memory.router import GeometricRouter, _is_stub_path, _is_test_path

ANTI = dict(dedup=True, diversity_lambda=LAMBDA)
VARIANTS = {
    "A": BASE,
    "B": replace(BASE, **ANTI),
    "D": replace(BASE, **ANTI, focus_blocks=5, focus_ratio=0.6),
    "F15": replace(BASE, **ANTI, focus_blocks=15, focus_ratio=0.6),
    "M50": replace(BASE, **ANTI, focus_blocks=3, focus_ratio=0.6, focus_mass=0.5),
    "M70": replace(BASE, **ANTI, focus_blocks=3, focus_ratio=0.6, focus_mass=0.7),
    "M85": replace(BASE, **ANTI, focus_blocks=3, focus_ratio=0.6, focus_mass=0.85),
    "G20": replace(BASE, **ANTI, focus_blocks=5, focus_ratio=0.6, focus_flat_drop=0.2, focus_flat_blocks=15),
    "G35": replace(BASE, **ANTI, focus_blocks=5, focus_ratio=0.6, focus_flat_drop=0.35, focus_flat_blocks=15),
    # module glue (<header>, <gap:...>) takes no focus slot
    "Dx": replace(BASE, **ANTI, focus_blocks=5, focus_ratio=0.6, focus_skip_glue=True),
    "F15x": replace(BASE, **ANTI, focus_blocks=15, focus_ratio=0.6, focus_skip_glue=True),
    "M70x": replace(BASE, **ANTI, focus_blocks=3, focus_ratio=0.6, focus_mass=0.7, focus_skip_glue=True),
    "M85x": replace(BASE, **ANTI, focus_blocks=3, focus_ratio=0.6, focus_mass=0.85, focus_skip_glue=True),
}
INSTRUMENT = ("A", "B")


def ranks(ctx, mgr, gold: dict[str, list[int]]) -> dict:  # noqa: ANN001
    """Positions (1-based) in the selection order of the first chosen chunk covering each gold file."""
    pinned = len(ctx.pinned)
    raw_pos = prod_pos = 0
    first_raw: dict[str, int] = {}
    first_prod: dict[str, int | None] = {}
    score_of: dict[str, float] = {}
    prod_scores: list[float] = []
    nonsymbol_in_top10 = 0
    for k, (nid, score) in enumerate(ctx.selection):
        rec = mgr.record(nid)
        if rec is None or rec.kind != NodeKind.CHUNK:
            continue
        raw_pos += 1
        is_prod = k >= pinned and not _is_test_path(rec.path) and not _is_stub_path(rec.path)
        if is_prod:
            prod_pos += 1
            prod_scores.append(score)
            if prod_pos <= 10 and (rec.symbol or "<").startswith("<"):
                nonsymbol_in_top10 += 1
        lines = gold.get(rec.path or "")
        if lines and rec.start_line is not None and any(rec.start_line <= ln <= (rec.end_line or rec.start_line) for ln in lines):
            if rec.path not in first_raw:
                first_raw[rec.path] = raw_pos
                first_prod[rec.path] = prod_pos if is_prod else None
                score_of[rec.path] = score
    covered = len(first_raw) == len(gold)
    best = max(prod_scores) if prod_scores else 0.0
    top5 = sorted(prod_scores, reverse=True)[:5]
    return {
        "n_chunks": raw_pos, "n_prod": prod_pos, "nonsymbol_top10": nonsymbol_in_top10,
        "rank_raw": max(first_raw.values()) if covered else None,
        "rank_prod": (max(v for v in first_prod.values()) if covered and all(v is not None for v in first_prod.values()) else None),
        "score_ratio": (min(score_of.values()) / best if covered and best > 0 else None),
        "drop_1_5": ((top5[0] - top5[4]) / top5[0] if len(top5) == 5 and top5[0] > 0 else None),
    }


def run_repo(repo: Path, name: str, items: list[dict], wd: Path, encoder, *, rank: int, budget: int, out: Path) -> None:  # noqa: ANN001
    done = set()
    if out.exists():
        done = {(r["set"], r["repo"], r["case"], r["variant"]) for r in map(json.loads, out.read_text().splitlines())}
    wt = Worktree(repo, wd / "worktree")
    cfg = MemoryConfig(storage_dir=wd / "zt_store", repo_root=wt.path, rank=rank, calibration_samples=1024)
    try:
        with NewtonianMemoryManager(cfg, encoder) as mgr:
            for it in items:
                todo = [(s, v) for s in it["queries"] for v in VARIANTS if (s, name, it["id"], v) not in done]
                if not todo:
                    continue
                wt.checkout(it["base"])
                if it["tests"]:
                    wt.take_files(it["fix"], it["tests"])
                mgr.sync()
                rows = []
                for qset, variant in todo:
                    ctx = GeometricRouter(mgr, VARIANTS[variant]).route_to_compute_engine(it["queries"][qset], token_budget=budget)
                    _, _, body = _coverage(spans(ctx, body_only=True), it["gold"])
                    _, _, seen = _coverage(spans(ctx, body_only=False), it["gold"])
                    row = {"set": qset, "repo": name, "case": it["id"], "variant": variant, "tokens": ctx.token_estimate,
                           "body": body, "map": seen, "gold_in_tests": any(_is_test_path(p) for p in it["gold"])}
                    if variant in INSTRUMENT:
                        row.update(ranks(ctx, mgr, it["gold"]))
                    rows.append(row)
                with out.open("a", encoding="utf-8") as fh:
                    fh.writelines(json.dumps(r) + "\n" for r in rows)
                print(f"[{name} {it['id'][:10]}] {len(rows)} rows", flush=True)
    finally:
        wt.remove()


def _bucket(r: int) -> str:
    for lo, hi in ((1, 1), (2, 2), (3, 3), (4, 5), (6, 10), (11, 15), (16, 20), (21, 30), (31, 999)):
        if lo <= r <= hi:
            return f"{lo}" if lo == hi else (f"{lo}-{hi}" if hi < 999 else f"{lo}+")
    return "?"


def report(rows: list[dict]) -> str:
    by = {(r["set"], r["repo"], r["case"], r["variant"]): r for r in rows}
    out = []
    for qset in sorted({r["set"] for r in rows}):
        cases = sorted({(r["repo"], r["case"]) for r in rows if r["set"] == qset
                        and all((qset, r["repo"], r["case"], v) in by for v in VARIANTS)})
        n = len(cases)
        a = [by[(qset, *c, "A")] for c in cases]
        hits = [r for r in a if r["body"]]
        out += [f"### {qset} — {n} casi; A trova il punto nel corpo in {len(hits)}", "",
                f"- ground truth in un file di test: {sum(r['gold_in_tests'] for r in a)} casi",
                f"- blocchi (chunk) scelti da A: media {statistics.mean(r['n_chunks'] for r in a):.1f}, "
                f"di produzione {statistics.mean(r['n_prod'] for r in a):.1f}",
                f"- chunk senza simbolo (intestazioni, blocchi di modulo) tra i primi 10 di produzione: media "
                f"{statistics.mean(r['nonsymbol_top10'] for r in a):.1f}", ""]
        prods = [r["rank_prod"] for r in hits if r["rank_prod"] is not None]
        raws = [r["rank_raw"] for r in hits if r["rank_raw"] is not None]
        out += ["| posizione del blocco giusto (A, quando lo trova) | casi (contando solo produzione) | casi (contando tutti i chunk) |",
                "|---|---|---|"]
        for label in ("1", "2", "3", "4-5", "6-10", "11-15", "16-20", "21-30", "31+"):
            out.append(f"| {label} | {sum(_bucket(x) == label for x in prods)} | {sum(_bucket(x) == label for x in raws)} |")
        if prods:
            out += ["", f"- posizione mediana {statistics.median(prods):.0f} (produzione), {statistics.median(raws):.0f} (tutti); "
                    f"entro la 5ª: {sum(x <= 5 for x in prods)}/{len(prods)}; oltre la 10ª: {sum(x > 10 for x in prods)}/{len(prods)}"]
        ratios = [r["score_ratio"] for r in hits if r["score_ratio"] is not None]
        if ratios:
            out.append(f"- punteggio del blocco giusto rispetto al migliore: mediana {statistics.median(ratios):.2f}; "
                       f"sotto 0,6: {sum(x < 0.6 for x in ratios)}/{len(ratios)}")
        drops = [r["drop_1_5"] for r in a if r["drop_1_5"] is not None]
        if drops:
            out.append(f"- calo di punteggio dal 1° al 5° candidato: mediana {statistics.median(drops):.2f} "
                       f"(quote: <0,2 {sum(x < 0.2 for x in drops)}, <0,35 {sum(x < 0.35 for x in drops)} su {len(drops)})")
        tok_a = statistics.mean(r["tokens"] for r in a)
        body_a = sum(r["body"] for r in a)
        out += ["", "| variante | token medi | vs A | punto nel corpo | persi / vinti contro A | McNemar p | entro il criterio (≥35% token in meno, ≤2 punti persi) |",
                "|---|---|---|---|---|---|---|"]
        for v in VARIANTS:
            rs = [by[(qset, *c, v)] for c in cases]
            tok = statistics.mean(r["tokens"] for r in rs)
            body = sum(r["body"] for r in rs)
            lost = sum(x["body"] and not y["body"] for x, y in zip(a, rs))
            won = sum(y["body"] and not x["body"] for x, y in zip(a, rs))
            cut = 1 - tok / tok_a
            ok = "sì" if cut >= 0.35 and (body_a - body) / n <= 0.02 else "no"
            out.append(f"| {v} | {tok:.0f} | {-100 * cut:+.0f}% | {body}/{n} ({100 * body / n:.0f}%) | {lost} / {won} | "
                       f"{mcnemar_exact(won, lost):.3g} | {ok if v not in ('A', 'B') else '-'} |")
        out.append("")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--workdir", type=Path, required=True)
    ap.add_argument("--repos-dir", type=Path, required=True)
    ap.add_argument("--loc", type=Path)
    ap.add_argument("--loc-en", type=Path)
    ap.add_argument("--traceback", type=Path, nargs="*", default=[])
    ap.add_argument("--encoder", default="hashing:1024")
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--budget", type=int, default=12_000)
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    args.workdir.mkdir(parents=True, exist_ok=True)
    out = args.workdir / "adaptive_results.jsonl"
    if not args.report_only:
        per_repo: dict[str, dict[str, dict]] = {}
        for path, qset in ((args.loc, "italiano"), (args.loc_en, "commit")):
            if not path:
                continue
            for name, cases in json.loads(path.read_text(encoding="utf-8")).items():
                for d in cases:
                    c = LocCase.from_dict(d)
                    item = per_repo.setdefault(name, {}).setdefault(c.id, {"id": c.id, "base": c.base, "fix": c.fix, "tests": [],
                                                                           "queries": {}, "gold": c.gold})
                    item["queries"][qset] = c.query
        tb: dict[str, list[dict]] = {}
        for path in args.traceback:
            repo, cases = load_cases(path)
            for c in cases:
                gold = {rel: changed_lines(repo, c.base, c.fix, rel) for rel in c.source_files if not _added_file(repo, c.base, rel)}
                if gold:
                    tb.setdefault(path.parent.name, []).append({"id": c.id, "base": c.base, "fix": c.fix, "tests": list(c.test_files),
                                                                "queries": {"traceback": c.failure_excerpt[-MAX_QUERY_CHARS:]},
                                                                "gold": gold})
        encoder = build_encoder(args.encoder)
        for name in sorted(set(per_repo) | set(tb)):
            repo = (args.repos_dir / name).resolve()
            print(f"== {name}", flush=True)
            if name in per_repo:
                run_repo(repo, name, list(per_repo[name].values()), args.workdir / name / "loc", encoder, rank=args.rank,
                         budget=args.budget, out=out)
            if name in tb:
                run_repo(repo, name, tb[name], args.workdir / name / "tb", encoder, rank=args.rank, budget=args.budget, out=out)
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
    text = report(rows)
    (args.workdir / "adaptive_report.md").write_text("# Contesto a fuoco — fase 1b\n\n" + text + "\n", encoding="utf-8")
    print("\n" + text)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.exit(main())
