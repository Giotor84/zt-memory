import io
import json
import os
import shutil
import subprocess

import pytest

from zt_memory.cli import main
from zt_memory.mcp_server import McpStdioServer, ZtTools
from zt_memory.need import NeedSettings, NeedState, SpanRequest
from zt_memory.project import ProjectConfig
from zt_memory.spans import SpanPolicy, SpanRefused, parse_span

PRICING = '''VAT_RATE = 0.22


def apply_discount(price: float, percent: float) -> float:
    """Price after a percentage discount."""
    return round(price - price * percent, 2)


def with_vat(price: float) -> float:
    return round(price * (1 + VAT_RATE), 2)
'''
SECRET = "DEBTOR_NAME = 'Mario Rossi'\n"


@pytest.fixture
def repo(tmp_path, capsys):
    root = tmp_path / "shop_repo"
    (root / "shop").mkdir(parents=True)
    (root / "shop/__init__.py").write_text("")
    (root / "shop/pricing.py").write_text(PRICING)
    (root / "shop/debtors.py").write_text(SECRET)
    (root / "shop/long.py").write_text("".join(f"x{i} = {i}\n" for i in range(1, 701)))
    (root / ".env").write_text("TOKEN=abc\n")
    assert main(["init", "--repo", str(root), "--rank", "32"]) == 0
    cfg = ProjectConfig.load(root)
    cfg.memory.update(calibration_samples=128, lock_timeout_s=0.2, exclude_globs="debtors.py")
    cfg.save(overwrite=True)
    assert main(["sync", "--repo", str(root), "--quiet"]) == 0
    capsys.readouterr()
    return root


def _call(root, name, args, *, read_span=True):
    msgs = [{"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": name, "arguments": args}}]
    out = io.BytesIO()
    tools = ZtTools(ProjectConfig.load(root), read_span=read_span)
    McpStdioServer(tools, inp=io.BytesIO(json.dumps(msgs[0]).encode() + b"\n"), out=out, log=lambda s: None).serve()
    return json.loads(out.getvalue().splitlines()[0])


def _listing(root, *, read_span):
    raw = b"".join(json.dumps(m).encode() + b"\n" for m in (
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18"}},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}))
    out = io.BytesIO()
    McpStdioServer(ZtTools(ProjectConfig.load(root), read_span=read_span), inp=io.BytesIO(raw), out=out,
                   log=lambda s: None).serve()
    init, listing = (json.loads(x) for x in out.getvalue().splitlines())
    return init["result"]["instructions"], {t["name"]: t for t in listing["result"]["tools"]}


def test_read_span_is_offered_only_when_enabled(repo):
    instructions, tools = _listing(repo, read_span=False)
    assert "zt_read_span" not in tools and "zt_read_span" not in instructions
    reply = _call(repo, "zt_read_span", {"spans": ["shop/pricing.py:L1-2"]}, read_span=False)
    assert reply["error"]["code"] == -32602
    instructions, tools = _listing(repo, read_span=True)
    assert "zt_read_span" in tools and "zt_read_span" in instructions
    assert "zt_read_span" in tools["zt_query_context"]["description"]
    assert tools["zt_read_span"]["inputSchema"]["properties"]["spans"]["maxItems"] == 3


def test_reads_exact_lines_from_disk(repo):
    reply = _call(repo, "zt_read_span", {"spans": ["shop/pricing.py:L4-6", "./shop/pricing.py:9-10"], "line_numbers": True})
    text = reply["result"]["content"][0]["text"]
    assert reply["result"]["isError"] is False
    assert "### shop/pricing.py  L4-6 (of 10)" in text and "4 | def apply_discount(price: float, percent: float) -> float:" in text
    assert "### shop/pricing.py  L9-10 (of 10)" in text and "10 |     return round(price * (1 + VAT_RATE), 2)" in text
    assert "refused" not in text


def test_excluded_hidden_and_outside_files_are_refused_without_content(repo, tmp_path):
    outside = tmp_path / "outside.py"
    outside.write_text("LEAK = 1\n")
    links = {}
    try:
        os.symlink(outside, repo / "shop/link_out.py")
        os.symlink(repo / "shop/debtors.py", repo / "shop/alias.py")
        links = {"shop/link_out.py:L1-1": "outside the project", "shop/alias.py:L1-1": "excluded"}
    except (OSError, NotImplementedError):
        pass  # no symlinks on this platform
    cases = {
        "shop/debtors.py:L1-1": "excluded",
        "SHOP/DEBTORS.PY:L1-1": "excluded",
        "shop/../shop/debtors.py:L1-1": "outside the project",
        "../outside.py:L1-1": "outside the project",
        str(outside) + ":L1-1": "outside the project",
        "C:\\Windows\\win.ini:L1-1": "outside the project",
        ".env:L1-1": "hidden",
        ".zt/config.toml:L1-5": "hidden",
        "shop/missing.py:L1-1": "no such file",
        **links,
    }
    for span, reason in cases.items():
        reply = _call(repo, "zt_read_span", {"spans": [span]})
        text = reply["result"]["content"][0]["text"]
        assert reply["result"]["isError"] is True, span
        assert reason in text, (span, text)
        assert "Mario Rossi" not in text and "LEAK" not in text and "TOKEN" not in text
    mixed = _call(repo, "zt_read_span", {"spans": ["shop/debtors.py:L1-1", "shop/pricing.py:L1-1"]})
    text = mixed["result"]["content"][0]["text"]
    assert mixed["result"]["isError"] is False and "VAT_RATE = 0.22" in text
    assert "refused:\n- shop/debtors.py:L1-1: excluded" in text and "Mario Rossi" not in text


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_git_ignored_files_are_refused(repo):
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / ".gitignore").write_text("generated.py\n")
    (repo / "generated.py").write_text("X = 1\n")
    reply = _call(repo, "zt_read_span", {"spans": ["generated.py:L1-1"]})
    assert reply["result"]["isError"] is True and "ignored by git" in reply["result"]["content"][0]["text"]
    assert _call(repo, "zt_read_span", {"spans": ["shop/pricing.py:L1-1"]})["result"]["isError"] is False


def test_line_limits(repo):
    text = _call(repo, "zt_read_span", {"spans": ["shop/long.py:L1-700"]})["result"]["content"][0]["text"]
    assert "L1-300 (of 700) (cut" in text and "x300 = 300" in text and "x301 = 301" not in text
    text = _call(repo, "zt_read_span", {"spans": ["shop/long.py:L1-300", "shop/long.py:L301-600",
                                                   "shop/long.py:L601-700"]})["result"]["content"][0]["text"]
    assert "L301-600 (of 700)" in text and "shop/long.py:L601-700: not read, the 600-line limit" in text
    end = _call(repo, "zt_read_span", {"spans": ["shop/long.py:L690-900"]})["result"]["content"][0]["text"]
    assert "L690-700 (of 700)\n" in end and "(cut" not in end
    too_many = _call(repo, "zt_read_span", {"spans": ["shop/pricing.py:L1-1"] * 4})
    assert too_many["result"]["isError"] is True and "at most 3" in too_many["result"]["content"][0]["text"]
    past = _call(repo, "zt_read_span", {"spans": ["shop/pricing.py:L50-60"]})
    assert past["result"]["isError"] is True and "only 10 lines" in past["result"]["content"][0]["text"]


def test_focus_mode_points_to_read_span(repo):
    reply = _call(repo, "zt_query_context", {"query": "discount on the price", "mode": "focus"})
    text = reply["result"]["content"][0]["text"]
    assert reply["result"]["isError"] is False
    assert text.startswith("zt-memory: ") and " whole, " in text and "def apply_discount" in text
    bad = _call(repo, "zt_query_context", {"query": "x", "mode": "focus", "token_budget": 100})
    assert bad["result"]["isError"] is True


def test_need_round_of_zt_code_uses_the_same_rule(repo):
    cfg = ProjectConfig.load(repo).memory_config()
    st = NeedState(NeedSettings(), repo, None, "t", "S", "C", policy=SpanPolicy(repo, cfg))
    assert st.read(SpanRequest("NEED", "shop/debtors.py", 1, 1), 600) is None
    assert st.read(SpanRequest("NEED", "shop/pricing.py", 1, 1), 600) == ("VAT_RATE = 0.22", 1, 1)


def test_parse_span():
    assert parse_span("src/a.py:L10-40") == ("src/a.py", 10, 40)
    assert parse_span("`src/a.py:10-40`") == ("src/a.py", 10, 40)
    assert parse_span("src/a.py:L12") == ("src/a.py", 12, 12)
    assert parse_span("C:\\repo\\a.py:L1-2") == ("C:\\repo\\a.py", 1, 2)
    with pytest.raises(SpanRefused):
        parse_span("src/a.py")


def test_several_projects_behind_one_server(repo, tmp_path, capsys):
    from zt_memory.mcp_server import MultiProjectTools, project_names

    other = tmp_path / "other"
    (other / "lib").mkdir(parents=True)
    (other / "lib/stock.py").write_text("def stock_level(sku: str) -> int:\n    return 7\n")
    bare = tmp_path / "bare"
    bare.mkdir()
    names = project_names([str(repo), f"inv={other}", f"none={bare}"])
    assert list(names) == ["shop_repo", "inv", "none"]
    with pytest.raises(ValueError):
        project_names([str(repo), str(repo)])
    tools = MultiProjectTools({
        "shop_repo": ZtTools(ProjectConfig.load(repo), read_span=True),
        "inv": ZtTools(None, start=other, read_span=True),
        "none": ZtTools(None, start=bare, read_span=True)})
    defs = {t["name"]: t for t in tools.tool_definitions}
    assert all(t["inputSchema"]["required"][0] == "project" for t in defs.values())
    assert defs["zt_read_span"]["inputSchema"]["properties"]["project"]["enum"] == ["shop_repo", "inv", "none"]
    assert "shop_repo, inv, none" in tools.instructions
    assert "VAT_RATE" in tools.call("zt_read_span", {"project": "shop_repo", "spans": ["shop/pricing.py:L1-1"]})
    with pytest.raises(Exception, match="must be one of"):
        tools.call("zt_read_span", {"spans": ["shop/pricing.py:L1-1"]})
    from zt_memory.mcp_server import ToolUnavailable
    with pytest.raises(ToolUnavailable, match="not enabled"):
        tools.call("zt_read_span", {"project": "none", "spans": ["a.py:L1-1"]})
    # the single-project tool list is unchanged: no `project` argument
    assert "project" not in ZtTools(ProjectConfig.load(repo)).tool_definitions[0]["inputSchema"]["properties"]
    tools.close()
