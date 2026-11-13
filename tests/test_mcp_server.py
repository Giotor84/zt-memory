import io
import json
import subprocess
import sys
import time

import pytest

from zt_memory.cli import main
from zt_memory.mcp_server import SUPPORTED_PROTOCOL_VERSIONS, McpStdioServer, ZtTools
from zt_memory.project import ProjectConfig

PRICING = '''VAT_RATE = 0.22


def apply_discount(price: float, percent: float) -> float:
    """Price after a percentage discount."""
    return round(price - price * percent, 2)


def with_vat(price: float) -> float:
    return round(price * (1 + VAT_RATE), 2)
'''


@pytest.fixture
def repo(tmp_path, capsys):
    root = tmp_path / "shop_repo"
    (root / "shop").mkdir(parents=True)
    (root / "shop/__init__.py").write_text("")
    (root / "shop/pricing.py").write_text(PRICING)
    (root / "shop/inventory.py").write_text("def stock_level(sku: str) -> int:\n    return 0\n")
    assert main(["init", "--repo", str(root), "--rank", "32"]) == 0
    cfg = ProjectConfig.load(root)
    cfg.memory.update(calibration_samples=128, lock_timeout_s=0.2)
    cfg.save(overwrite=True)
    assert main(["sync", "--repo", str(root), "--quiet"]) == 0
    capsys.readouterr()
    return root


def _rpc(i, method, params=None):
    msg = {"jsonrpc": "2.0", "id": i, "method": method}
    if params is not None:
        msg["params"] = params
    return msg


def _run(root, messages, tools=None):
    raw = b"".join((m if isinstance(m, bytes) else json.dumps(m).encode()) + b"\n" for m in messages)
    out = io.BytesIO()
    tools = tools or ZtTools(ProjectConfig.load(root))
    McpStdioServer(tools, inp=io.BytesIO(raw), out=out, log=lambda s: None).serve()
    replies = [json.loads(line) for line in out.getvalue().splitlines()]
    return {r.get("id"): r for r in replies}, replies


def _text(reply):
    assert reply["result"]["isError"] is False, reply
    return reply["result"]["content"][0]["text"]


def test_lifecycle_and_tool_listing(repo):
    by_id, replies = _run(repo, [
        _rpc(1, "initialize", {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "test"}}),
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        _rpc(2, "tools/list"),
        _rpc(3, "ping"),
        _rpc(4, "initialize", {"protocolVersion": "1999-01-01"}),
    ])
    assert len(replies) == 4  # the notification is never answered
    init = by_id[1]["result"]
    assert init["protocolVersion"] == "2025-06-18" and init["capabilities"] == {"tools": {"listChanged": False}}
    assert init["serverInfo"]["name"] == "zt-memory" and "zt_query_context" in init["instructions"]
    assert by_id[4]["result"]["protocolVersion"] == SUPPORTED_PROTOCOL_VERSIONS[0]
    tools = {t["name"]: t for t in by_id[2]["result"]["tools"]}
    assert set(tools) == {"zt_query_context", "zt_record_decision", "zt_status"}
    assert tools["zt_query_context"]["inputSchema"]["required"] == ["query"]
    assert by_id[3]["result"] == {}


def test_query_decision_status_round_trip(repo):
    by_id, _ = _run(repo, [
        _rpc(1, "tools/call", {"name": "zt_query_context", "arguments": {"query": "apply_discount returns a wrong price", "token_budget": 2000}}),
        _rpc(2, "tools/call", {"name": "zt_record_decision", "arguments": {
            "summary": "Discounts are percentages in [0, 100]; rounding to cents happens once, at the end.",
            "paths": ["shop/pricing.py::apply_discount", "shop\\pricing.py", "shop/missing.py"]}}),
        _rpc(3, "tools/call", {"name": "zt_query_context", "arguments": {"query": "discount rounding percentages policy", "line_numbers": True}}),
        _rpc(4, "tools/call", {"name": "zt_status", "arguments": {"limit": 5}}),
    ])
    ctx = _text(by_id[1])
    assert ctx.startswith("zt-memory: ") and "shop/pricing.py" in ctx and "def apply_discount" in ctx
    assert "Output contract" not in ctx and "## Task" not in ctx  # the client brings its own task
    rec = _text(by_id[2])
    assert "recorded, linked to shop/pricing.py::apply_discount, shop/pricing.py" in rec and "shop/missing.py" in rec
    assert "rounding to cents happens once" in _text(by_id[3])  # the decision comes back with the code
    status = _text(by_id[4])
    assert "hottest symbols" in status and "shop/pricing.py::apply_discount" in status


def test_protocol_and_tool_errors(repo):
    by_id, replies = _run(repo, [
        b"{not json",
        _rpc(1, "tools/call", {"name": "rm_rf", "arguments": {}}),
        _rpc(2, "resources/list"),
        _rpc(3, "tools/call", {"name": "zt_query_context", "arguments": {"query": "  "}}),
        _rpc(4, "tools/call", {"name": "zt_query_context", "arguments": {"query": "x", "token_budget": 10}}),
        _rpc(5, "tools/call", {"name": "zt_record_decision", "arguments": {"summary": "s", "paths": "shop/pricing.py"}}),
    ])
    assert replies[0]["error"]["code"] == -32700
    assert by_id[1]["error"]["code"] == -32602 and by_id[2]["error"]["code"] == -32601
    for i in (3, 4, 5):  # bad arguments are tool errors the model can read and fix, not protocol errors
        assert by_id[i]["result"]["isError"] is True and "invalid arguments" in by_id[i]["result"]["content"][0]["text"]


def test_store_lock_is_released_between_calls(repo):
    _run(repo, [_rpc(1, "tools/call", {"name": "zt_status", "arguments": {}})])
    with ProjectConfig.load(repo).open_manager() as mgr:  # would raise StoreLockedError if the server kept it
        assert mgr.stats().nodes_by_kind


def test_cli_stdout_carries_only_protocol_messages(repo):
    msgs = [
        _rpc(1, "initialize", {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t"}}),
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        _rpc(2, "tools/call", {"name": "zt_query_context", "arguments": {"query": "with_vat VAT_RATE"}}),
    ]
    proc = subprocess.run([sys.executable, "-m", "zt_memory.cli", "mcp", "--repo", str(repo)],
                          input="".join(json.dumps(m) + "\n" for m in msgs).encode(), capture_output=True, timeout=120)
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")
    lines = proc.stdout.decode("utf-8").splitlines()
    replies = [json.loads(line) for line in lines]  # every stdout line is JSON-RPC
    assert [r["id"] for r in replies] == [1, 2]
    assert "with_vat" in replies[1]["result"]["content"][0]["text"]
    assert "location(s)" in replies[1]["result"]["content"][0]["text"]  # `zt mcp` defaults to locations
    assert b"[zt-mcp] serving" in proc.stderr


def _call(i, name, **arguments):
    return _rpc(i, "tools/call", {"name": name, "arguments": arguments})


def test_no_project_explains_activation_then_works_without_restart(tmp_path, capsys):
    root = tmp_path / "new_project"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg/core.py").write_text("def answer():\n    return 42\n")
    tools = ZtTools(start=root)  # what `zt mcp` without --repo does in a folder with no .zt
    by_id, _ = _run(root, [_rpc(1, "initialize", {"protocolVersion": "2025-06-18"}), _rpc(2, "tools/list"),
                           _call(3, "zt_query_context", query="answer")], tools)
    assert by_id[1]["result"]["serverInfo"]["name"] == "zt-memory" and len(by_id[2]["result"]["tools"]) == 3
    res = by_id[3]["result"]
    assert res["isError"] and "not enabled for this project" in res["content"][0]["text"]
    assert "exclude_globs" in res["content"][0]["text"] and not tools.encoder_loaded

    assert main(["init", "--repo", str(root), "--rank", "32"]) == 0
    cfg = ProjectConfig.load(root)
    cfg.memory.update(calibration_samples=128)
    cfg.save(overwrite=True)
    by_id, _ = _run(root, [_call(4, "zt_query_context", query="answer")], tools)
    assert by_id[4]["result"]["isError"] and "never built" in by_id[4]["result"]["content"][0]["text"]

    assert main(["sync", "--repo", str(root), "--quiet"]) == 0
    capsys.readouterr()
    by_id, _ = _run(root, [_call(5, "zt_query_context", query="answer")], tools)  # same server instance
    assert "def answer" in _text(by_id[5])


def test_encoder_is_lazy_and_released_when_idle(repo):
    loads = []

    def factory(pc):
        loads.append(1)
        return pc.build_encoder()

    tools = ZtTools(ProjectConfig.load(repo), encoder_factory=factory, idle_unload_s=0.3)
    _run(repo, [_rpc(1, "initialize", {}), _rpc(2, "tools/list")], tools)
    assert loads == [] and not tools.encoder_loaded  # handshake and listing never load the model
    _run(repo, [_call(3, "zt_status")], tools)
    assert loads == [1] and tools.encoder_loaded
    deadline = time.monotonic() + 10
    while tools.encoder_loaded and time.monotonic() < deadline:
        time.sleep(0.1)
    assert not tools.encoder_loaded  # released after idle_unload_s
    by_id, _ = _run(repo, [_call(4, "zt_query_context", query="with_vat")], tools)
    assert loads == [1, 1] and "with_vat" in _text(by_id[4])  # reloaded on demand
    tools.close()


def test_large_change_is_not_synced_inside_a_call(repo):
    (repo / "shop/pricing.py").write_text(PRICING.replace("1 + VAT_RATE", "1.0 + VAT_RATE"))
    tools = ZtTools(ProjectConfig.load(repo), max_inline_sync_files=0)
    by_id, _ = _run(repo, [_call(1, "zt_query_context", query="with_vat")], tools)
    text = _text(by_id[1])
    assert "index NOT refreshed: 1 files changed" in text and "with_vat" in text
    (repo / "shop/inventory.py").write_text("def stock_level(sku: str) -> int:\n    return 1\n")
    by_id, _ = _run(repo, [_call(2, "zt_query_context", query="stock_level")], ZtTools(ProjectConfig.load(repo)))
    assert "; index refreshed" in _text(by_id[2])  # default limit: a small change is synced inline


def test_cli_starts_in_a_folder_without_zt(tmp_path):
    msgs = [_rpc(1, "initialize", {"protocolVersion": "2025-06-18"}), _call(2, "zt_status")]
    proc = subprocess.run([sys.executable, "-m", "zt_memory.cli", "mcp"], cwd=tmp_path,
                          input="".join(json.dumps(m) + "\n" for m in msgs).encode(), capture_output=True, timeout=120)
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")
    replies = [json.loads(line) for line in proc.stdout.decode("utf-8").splitlines()]
    assert replies[0]["result"]["protocolVersion"] == "2025-06-18"
    assert replies[1]["result"]["isError"] and "not enabled" in replies[1]["result"]["content"][0]["text"]


def test_locations_mode_returns_signatures_not_bodies(repo):
    by_id, _ = _run(repo, [
        _call(1, "zt_query_context", query="apply_discount returns a wrong price", mode="locations"),
        _call(2, "zt_query_context", query="apply_discount returns a wrong price", mode="locations", token_budget=100),
        _call(3, "zt_query_context", query="x", mode="everything"),
        _call(4, "zt_query_context", query="x", token_budget=150),
    ])
    text = _text(by_id[1])
    assert "location(s)" in text and "read the lines you need" in text
    assert "shop/pricing.py  L" in text and "def apply_discount(price: float, percent: float) -> float:" in text
    assert "Price after a percentage discount." in text  # first docstring line
    assert "return round(" not in text and "```" not in text  # no bodies
    first = next(line for line in text.splitlines() if line.startswith("- "))
    assert "apply_discount" in first  # best match first
    import re
    assert int(re.search(r"~(\d+) tokens", _text(by_id[2])).group(1)) <= 100  # the list respects token_budget
    for i in (3, 4):  # unknown mode; code mode keeps its 256-token minimum
        assert by_id[i]["result"]["isError"] is True


def test_server_default_mode(repo):
    tools = ZtTools(ProjectConfig.load(repo), default_mode="locations")
    by_id, _ = _run(repo, [_call(1, "zt_query_context", query="with_vat"), _call(2, "zt_query_context", query="with_vat", mode="code")], tools)
    assert "location(s)" in _text(by_id[1]) and "def with_vat(price: float) -> float:" in _text(by_id[1])
    assert "```python" in _text(by_id[2])
