"""`zt code` end to end against a fake llama-server speaking the real protocol (/health, /props, /tokenize, chat)."""
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from zt_memory.cli import main
from zt_memory.project import ProjectConfig

BUGGY = "def apply_discount(price: float, percent: float) -> float:\n    \"\"\"Price after a percentage discount (0-100).\"\"\"\n    return price - price * percent\n"
TEST = "from shop.pricing import apply_discount\n\n\ndef test_discount():\n    assert apply_discount(200, 10) == 180\n"
FIX = ("shop/pricing.py\n<<<<<<< SEARCH\n    return price - price * percent\n=======\n"
       "    return price - price * percent / 100\n>>>>>>> REPLACE\n")
WRONG = "shop/pricing.py\n<<<<<<< SEARCH\n    return price * percent\n=======\n    return 0\n>>>>>>> REPLACE\n"


class FakeLlama:
    def __init__(self, answers, n_ctx=8192):
        self.answers, self.n_ctx, self.requests = list(answers), n_ctx, []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):  # quiet
                pass

            def _send(self, obj):
                body = json.dumps(obj).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path == "/health":
                    self._send({"status": "ok"})
                elif self.path == "/props":
                    self._send({"default_generation_settings": {"n_ctx": outer.n_ctx}})
                else:
                    self.send_error(404)

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.path == "/tokenize":
                    self._send({"tokens": [0] * (len(payload["content"]) // 3)})
                elif self.path == "/v1/chat/completions":
                    outer.requests.append(payload)
                    text = outer.answers.pop(0) if outer.answers else "no idea"
                    self._send({"choices": [{"message": {"content": text}, "finish_reason": "stop"}],
                                "usage": {"prompt_tokens": 100, "completion_tokens": 20}})
                else:
                    self.send_error(404)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()


@pytest.fixture
def shop(tmp_path, capsys):
    root = tmp_path / "shop_repo"
    (root / "shop").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "shop/__init__.py").write_text("")
    (root / "shop/pricing.py").write_text(BUGGY)
    (root / "shop/inventory.py").write_text("def stock_level(sku: str) -> int:\n    return 0\n")
    (root / "tests/test_pricing.py").write_text(TEST)
    assert main(["init", "--repo", str(root), "--rank", "32"]) == 0
    cfg = ProjectConfig.load(root)
    cfg.memory.update(calibration_samples=128, lock_timeout_s=0.2)
    cfg.save(overwrite=True)
    capsys.readouterr()
    return root


def _code(capsys, root, url, *extra):
    code = main(["code", "apply_discount gives wrong prices: a 10% discount on 200 must give 180", "--repo", str(root),
                 "--engine", url, "--json", "--yes", *extra])
    out = capsys.readouterr()
    return code, json.loads(out.out), out.err


def test_code_applies_patch_syncs_and_heats(shop, capsys):
    llama = FakeLlama([FIX])
    try:
        code, res, _ = _code(capsys, shop, llama.url, "--check", f'"{sys.executable}" -m pytest -q', "--max-new-tokens", "1024")
    finally:
        llama.close()
    assert code == 0 and res["status"] == "applied" and res["changed"] == ["shop/pricing.py"]
    assert res["check_returncode"] == 0
    assert "percent / 100" in (shop / "shop/pricing.py").read_text()
    heated = {h["symbol"]: h for h in res["heated"]}
    hot = heated["chunk:shop/pricing.py::apply_discount"]
    assert hot["energy_after"] > hot["energy_before"]  # the edit impulse heated the touched symbol
    prompt = llama.requests[0]["messages"][1]["content"]
    assert "<<<<<<< SEARCH" in prompt and "def apply_discount" in prompt  # navigator put the right code in context
    assert res["context_tokens"] <= res["served_ctx"] - 1024


def test_retry_with_feedback_then_undo(shop, capsys):
    llama = FakeLlama([WRONG, FIX])
    try:
        code, res, _ = _code(capsys, shop, llama.url)
    finally:
        llama.close()
    assert code == 0 and res["attempts"] == 2
    feedback = llama.requests[1]["messages"][-1]["content"]
    assert "could not be applied" in feedback and "return price - price * percent" in feedback  # real text shown
    assert main(["undo", "--repo", str(shop)]) == 0
    assert (shop / "shop/pricing.py").read_text() == BUGGY


def test_failed_check_reverts(shop, capsys):
    breaks = FIX.replace("percent / 100", "percent / 1000")
    llama = FakeLlama([breaks])
    try:
        code, res, _ = _code(capsys, shop, llama.url, "--check", f'"{sys.executable}" -m pytest -q', "--revert-on-fail")
    finally:
        llama.close()
    assert code == 5 and res["status"] == "reverted" and res["check_returncode"] != 0
    assert (shop / "shop/pricing.py").read_text() == BUGGY


def test_unusable_answer_changes_nothing(shop, capsys):
    llama = FakeLlama(["I think you should rewrite everything.", "Still prose."])
    try:
        code, res, err = _code(capsys, shop, llama.url, "--retries", "1")
    finally:
        llama.close()
    assert code == 4 and res["status"] == "no_edits" and res["attempts"] == 2
    assert (shop / "shop/pricing.py").read_text() == BUGGY and "model answer" in err


def test_engine_down_is_a_clear_error(shop, capsys):
    code = main(["code", "x", "--repo", str(shop), "--engine", "http://127.0.0.1:9/v1", "--yes"])
    assert code == 1 and "llama-server" in capsys.readouterr().err


def test_engine_command_reports_budget(shop, capsys):
    llama = FakeLlama(["def rev(s):\n    return s[::-1]"], n_ctx=12288)
    try:
        code = main(["engine", "--repo", str(shop), "--engine", llama.url, "--json"])
    finally:
        llama.close()
    info = json.loads(capsys.readouterr().out)
    assert code == 0 and info["served_ctx"] == 12288 and info["router_budget"] == 12288 - 3072 - 128
