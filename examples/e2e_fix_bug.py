"""End-to-end check on a real bug: question -> navigator -> LLM (GPU) -> patch applied -> physics sync -> tests.

    1. start llama-server (see docs/ENGINE_WINDOWS.md), then:
    python examples\\e2e_fix_bug.py --engine http://127.0.0.1:8080/v1
    python examples\\e2e_fix_bug.py --encoder hf:Qwen/Qwen3-Embedding-0.6B   # with the real navigator encoder

Builds a small shop project whose `apply_discount` treats a percentage as a fraction (3 failing tests,
plus distractor modules), proves the tests fail, runs `zt code` with --check/--revert-on-fail through the
real CLI, and verifies: patch applied to the right file, tests green, edited symbol heated by the sync.
Exit code 0 only if every stage passes.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path

FILES = {
    "shop/__init__.py": "",
    "shop/pricing.py": '''
        """Price computations for the shop."""
        from __future__ import annotations

        VAT_RATE = 0.22


        def apply_discount(price: float, percent: float) -> float:
            """Return the price after a percentage discount; `percent` is between 0 and 100."""
            if not 0 <= percent <= 100:
                raise ValueError("percent must be between 0 and 100")
            return round(price - price * percent, 2)


        def with_vat(price: float) -> float:
            """Gross price including VAT."""
            return round(price * (1 + VAT_RATE), 2)
    ''',
    "shop/cart.py": '''
        from __future__ import annotations

        from .pricing import apply_discount, with_vat


        class Cart:
            def __init__(self) -> None:
                self.lines: list[tuple[str, float, int]] = []

            def add(self, sku: str, unit_price: float, qty: int = 1) -> None:
                self.lines.append((sku, unit_price, qty))

            def subtotal(self) -> float:
                return round(sum(price * qty for _, price, qty in self.lines), 2)

            def total(self, discount_percent: float = 0.0) -> float:
                return with_vat(apply_discount(self.subtotal(), discount_percent))
    ''',
    "shop/inventory.py": '''
        from __future__ import annotations

        _STOCK: dict[str, int] = {}


        def restock(sku: str, qty: int) -> None:
            _STOCK[sku] = _STOCK.get(sku, 0) + qty


        def reserve(sku: str, qty: int) -> bool:
            if _STOCK.get(sku, 0) < qty:
                return False
            _STOCK[sku] -= qty
            return True
    ''',
    "shop/users.py": '''
        from __future__ import annotations

        import hashlib


        def password_hash(password: str, salt: str) -> str:
            return hashlib.sha256((salt + password).encode()).hexdigest()


        def is_admin(email: str) -> bool:
            return email.endswith("@shop.example")
    ''',
    "tests/test_pricing.py": '''
        import pytest

        from shop.pricing import apply_discount


        def test_ten_percent():
            assert apply_discount(200, 10) == 180


        def test_zero_and_full():
            assert apply_discount(80, 0) == 80
            assert apply_discount(80, 100) == 0


        def test_rejects_out_of_range():
            with pytest.raises(ValueError):
                apply_discount(10, 120)
    ''',
    "README.md": "# Shop\n\nTiny shop backend: pricing, cart, inventory, users.\n",
}
BUG_REPORT = ("Customers report wrong prices: apply_discount(200, 10) returns -1800 instead of 180. "
              "The percent argument is a percentage between 0 and 100. Fix the discount computation.")


def run(cmd: list[str], cwd: Path, timeout: float = 3600) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--workdir", type=Path, default=Path("zt_e2e"))
    ap.add_argument("--encoder", default="hashing:1024", help="navigator encoder; hf:Qwen/Qwen3-Embedding-0.6B for the real one")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--rank", type=int, help="SVD rank (default 128 for hf: encoders, 64 otherwise; must match --projector)")
    ap.add_argument("--projector", help="reuse a frozen SVD basis (calibrate_encoder.py --save-projector) and skip calibration")
    args = ap.parse_args()

    py = sys.executable
    zt = [py, "-m", "zt_memory.cli"]
    stages: list[tuple[str, bool, str]] = []

    def stage(name: str, ok: bool, detail: str = "") -> bool:
        stages.append((name, ok, detail))
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""), flush=True)
        return ok

    def finish() -> int:
        print("\n" + "\n".join(f"  {'PASS' if ok else 'FAIL'}  {name:28s} {detail}" for name, ok, detail in stages))
        all_ok = all(ok for _, ok, _ in stages)
        print(f"\nE2E {'PASSED' if all_ok else 'FAILED'}  (project left in {args.workdir.resolve()})")
        return 0 if all_ok else 1

    # 0. engine
    from zt_memory.engine_client import ComputeEngineClient

    client = ComputeEngineClient(args.engine)
    if not stage("engine reachable", client.health(), args.engine):
        print("start llama-server first (docs/ENGINE_WINDOWS.md)")
        return finish()
    stage("engine context", True, f"{client.context_size()} tokens per slot")

    # 1. project with a real bug
    root = args.workdir.resolve() / "shop_repo"
    rank = args.rank or (128 if args.encoder.startswith("hf:") else 64)
    cached = args.workdir.resolve() / "projector_cache" / f"{re.sub(r'[^A-Za-z0-9.-]+', '_', args.encoder)}-r{rank}.pt"
    old_basis, old_cfg = root / ".zt" / "store" / "projector.pt", root / ".zt" / "config.toml"
    if not cached.exists() and old_basis.exists() and old_cfg.exists():  # keep the basis of a previous run's store
        cfg = old_cfg.read_text(encoding="utf-8")
        if args.encoder.split(":", 1)[-1] in cfg and re.search(rf"^rank\s*=\s*{rank}\s*$", cfg, re.MULTILINE):
            cached.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(old_basis, cached)
    if root.exists():
        shutil.rmtree(root, onerror=lambda f, p, e: (Path(p).chmod(0o700), f(p)))
    for rel, body in FILES.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(textwrap.dedent(body).lstrip("\n"), encoding="utf-8")
    if shutil.which("git"):
        run(["git", "init", "-q"], root)
        run(["git", "add", "-A"], root)
        run(["git", "-c", "user.name=e2e", "-c", "user.email=e2e@local", "commit", "-qm", "shop with discount bug"], root)
    before = run([py, "-m", "pytest", "-q"], root)
    if not stage("bug reproduced", before.returncode != 0, "tests fail before the fix"):
        return finish()

    # 2. navigator index; the SVD basis depends only on encoder + rank, so it is cached across runs
    projector = Path(args.projector).resolve() if args.projector else (cached if cached.exists() else None)
    init = run([*zt, "init", "--repo", str(root), "--encoder", args.encoder, "--rank", str(rank),
                "--engine-url", args.engine, *(["--projector", str(projector)] if projector else [])], root)
    if not stage("zt init", init.returncode == 0, init.stderr.strip()[-200:]):
        return finish()
    if projector:
        print(f"[....] reusing the SVD basis {projector}", flush=True)
    elif args.encoder.startswith("hf:"):
        print("[....] zt sync: the project has ~15 chunks, so the SVD is calibrated on ~1000 filler chunks from the stdlib;"
              " on CPU this takes several minutes, only on the first run (the basis is then cached)", flush=True)
    t = time.perf_counter()
    sync = run([*zt, "sync", "--repo", str(root), "--json"], root)
    ok = sync.returncode == 0
    detail = (f"{json.loads(sync.stdout)['chunks_added']} chunks in {time.perf_counter() - t:.1f}s" if ok else sync.stderr[-300:])
    if not stage("zt sync", ok, detail):
        return finish()
    if not projector:
        cached.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / ".zt" / "store" / "projector.pt", cached)

    # 3. the full loop through the real CLI
    print("[....] zt code: encoder load, routing, LLM answer, patch, tests (progress below)", flush=True)
    t = time.perf_counter()
    # stdout carries the JSON result; stderr (progress lines and diff) streams live to the console
    code = subprocess.run([*zt, "code", BUG_REPORT, "--repo", str(root), "--engine", args.engine, "--yes", "--json",
                           "--max-new-tokens", str(args.max_new_tokens), "--check", f'"{py}" -m pytest -q', "--revert-on-fail"],
                          cwd=str(root), stdout=subprocess.PIPE, text=True, encoding="utf-8", errors="replace", timeout=3600)
    try:
        res = json.loads(code.stdout)
    except json.JSONDecodeError:
        stage("zt code", False, f"exit {code.returncode}: {code.stdout[-500:]}")
        return finish()
    stage("context fitted", bool(res.get("context_tokens") is None or res["context_tokens"] <= (res["served_ctx"] or 10**9)),
          f"{res.get('context_tokens')} tokens (exact) / ctx {res.get('served_ctx')}")
    stage("LLM answered", res["attempts"] >= 1, f"{res['completion_tokens']} tokens in {res['llm_seconds']}s, {res['attempts']} attempt(s)")
    stage("patch applied", res["status"] in ("applied",) or bool(res["changed"]), f"status={res['status']} files={res['changed']}")
    stage("right file", res["changed"] == ["shop/pricing.py"], "")
    stage("tests green", res["check_returncode"] == 0, f"exit {res['check_returncode']}")
    heated = {h["symbol"]: h for h in res["heated"]}
    hot = heated.get("chunk:shop/pricing.py::apply_discount")
    stage("physics sync", bool(hot and hot["energy_after"] > hot["energy_before"]),
          f"apply_discount energy {hot['energy_before']:.3f} -> {hot['energy_after']:.3f}" if hot else "edited symbol not heated")
    stage("total time", True, f"{time.perf_counter() - t:.1f}s for zt code (incl. encoder load, LLM, tests)")
    if res["status"] != "applied":
        print("errors:", res["errors"])
    return finish()


if __name__ == "__main__":
    sys.exit(main())
