"""NEED protocol: parsing, when it is offered, round 2 (append with cache point / lean), and the zt code flow."""
import json
import sys
import types

import pytest

from zt_memory.cli import main
from zt_memory.coding import request_edits
from zt_memory.engine_bedrock import CALLS, BedrockEngineClient
from zt_memory.engine_client import ChatResult, ComputeEngineClient, ComputeEngineError
from zt_memory.need import END_NEED, NEED_CONTRACT, NeedSettings, NeedState, need_settings, offer_need, parse_need
from zt_memory.project import ProjectConfig, ProjectConfigError

PRICING = '''VAT = 0.22


def apply_discount(price: float, percent: float) -> float:
    """Price after a percentage discount (0-100)."""
    total = price - price * percent
    total = round(total, 2)
    return total


def with_vat(price: float) -> float:
    """Price including VAT."""
    if price < 0:
        raise ValueError("negative price")
    gross = price * (1 + VAT)
    cents = round(gross * 100)
    if cents % 5:
        cents += 5 - cents % 5
    result = cents / 100
    assert result >= price
    return result
'''
FIX_VAT = ("shop/pricing.py\n<<<<<<< SEARCH\n    gross = price * (1 + VAT)\n=======\n"
           "    gross = price * (1.0 + VAT)\n>>>>>>> REPLACE\n")
ASK_VAT = "NEED shop/pricing.py:L11-21\nKEEP shop/pricing.py:L4-8\nNOTE: apply_discount looks fine, the VAT rounding is in with_vat\n"


def test_parse_need_lines():
    ans = parse_need("I need more.\n- NEED `src/a.py:L10-20`\nNEED ./src/b.py:30-12\nKEEP src/c.py:L1-5\nNOTE: the bug is in b\n"
                     "NEED x.py:L1-2\nNEED y.py:L1-2\nNEED z.py:L1-2")
    assert [r.label() for r in ans.needs] == ["src/a.py:L10-20", "src/b.py:L12-30", "x.py:L1-2"]  # at most 3
    assert [r.label() for r in ans.keeps] == ["src/c.py:L1-5"] and ans.note == "the bug is in b"
    assert parse_need("shop/pricing.py\n<<<<<<< SEARCH\n...").needs == ()
    capped = parse_need("NEED a.py:L1-2\nNOTE: one\nNOTE: two\nNOTE: three\nEND_NEED\nNEED after.py:L1-2")
    assert capped.note == "one / two" and [r.path for r in capped.needs] == ["a.py"]  # 2 NOTE lines, nothing after END_NEED
    assert END_NEED in NEED_CONTRACT and "at most 2 lines" in NEED_CONTRACT and "no draft edit" in NEED_CONTRACT


def test_need_settings_resolution():
    assert need_settings({"need_rounds": -1}, "api").rounds == 1
    assert need_settings({"need_rounds": -1}, "") is None  # off without the api preset
    assert need_settings({"need_rounds": 2}, "").rounds == 2
    assert need_settings({"need_rounds": 0}, "api") is None


class _Block:
    def __init__(self, path, start, end, text, *, full=True, kind="CHUNK"):
        self.path, self.start_line, self.end_line, self.text, self.full = path, start, end, text, full
        self.kind = types.SimpleNamespace(name=kind)


class _Ctx:
    def __init__(self, blocks, pinned=()):
        self.blocks, self.pinned = blocks, pinned


def _repo(tmp_path):
    (tmp_path / "shop").mkdir()
    (tmp_path / "shop/pricing.py").write_text(PRICING)
    return tmp_path


def test_offer_only_with_a_map_and_no_traceback():
    sig = _Block("a.py", 1, 5, "def f()", full=False)
    assert offer_need(_Ctx([sig]), NeedSettings())
    assert not offer_need(_Ctx([_Block("a.py", 1, 5, "x")]), NeedSettings())  # everything already whole
    assert not offer_need(_Ctx([sig], pinned=("a.py:3",)), NeedSettings())  # traceback: answer already pinned
    assert not offer_need(_Ctx([sig]), NeedSettings(rounds=0))


def test_read_is_confined_to_the_repository(tmp_path):
    repo = _repo(tmp_path)
    st = NeedState(NeedSettings(max_lines=5), repo, _Ctx([]), "task", "sys", "contract")
    from zt_memory.need import SpanRequest
    assert st.read(SpanRequest("NEED", "shop/pricing.py", 11, 40), 600)[1:] == (11, 15)  # clamped to max_lines
    assert st.read(SpanRequest("NEED", "../etc/passwd", 1, 2), 600) is None
    assert st.read(SpanRequest("NEED", "/etc/passwd", 1, 2), 600) is None
    assert st.read(SpanRequest("NEED", "shop/missing.py", 1, 2), 600) is None


def test_lean_round_drops_round1_bodies(tmp_path):
    repo = _repo(tmp_path)
    header = _Block("shop/pricing.py", 1, 1, "VAT = 0.22", kind="FILE")
    st = NeedState(NeedSettings(), repo, _Ctx([header]), "fix the VAT rounding", "SYS", "CONTRACT")
    assert st.stop == [END_NEED]
    ans = parse_need(ASK_VAT)
    needed, kept = st.resolve(ans)
    msgs = st.next_messages(ans, needed, kept)
    assert st.stop is None  # last round: no stop sequence, no NEED
    assert [m["role"] for m in msgs] == ["system", "user"]
    text = msgs[1]["content"]
    assert "fix the VAT rounding" in text and "the VAT rounding is in with_vat" in text
    assert "cents += 5 - cents % 5" in text and "def apply_discount" in text and "VAT = 0.22" in text
    assert "BIG ROUND 1 CONTEXT" not in text and NEED_CONTRACT not in text  # last round: no more NEED offered


def test_two_rounds_keep_offering_need(tmp_path):
    st = NeedState(NeedSettings(rounds=2), _repo(tmp_path), _Ctx([]), "t", "SYS", "C")
    ans = parse_need(ASK_VAT)
    msgs = st.next_messages(ans, *st.resolve(ans))
    assert NEED_CONTRACT in msgs[1]["content"] and "ask once more" in msgs[1]["content"] and st.stop == [END_NEED]


class FakeEngine:
    def __init__(self, *answers):
        self.answers, self.seen = list(answers), []

    def chat_full(self, messages, *, max_tokens=2048, temperature=0.2, stop=None):
        self.seen.append([dict(m) for m in messages])
        self.stops = [*getattr(self, "stops", []), stop]
        return ChatResult(self.answers.pop(0), 100, 10, "stop", 0.1)


def test_request_edits_grants_one_round_then_edits(tmp_path):
    repo = _repo(tmp_path)
    st = NeedState(NeedSettings(), repo, _Ctx([]), "fix the VAT rounding", "SYS", "CONTRACT")
    eng = FakeEngine(ASK_VAT, FIX_VAT)
    got = request_edits(eng, [{"role": "user", "content": "round 1"}], repo, max_new_tokens=500, temperature=0, retries=0, need=st)
    assert got.ok and got.attempts == 2 and got.need_rounds == 1 and got.need_requests == [["shop/pricing.py:L11-21"]]
    assert "round 1" not in eng.seen[1][-1]["content"]  # round 2 is a new short prompt
    assert eng.stops == [[END_NEED], None]  # END_NEED stops round 1 only
    again = FakeEngine(ASK_VAT, ASK_VAT)  # asks twice: the second NEED is not granted (one round) and is no edit
    got = request_edits(again, [{"role": "user", "content": "r1"}], repo, max_new_tokens=500, temperature=0, retries=0,
                        need=NeedState(NeedSettings(), repo, _Ctx([]), "t", "S", "C"))
    assert not got.ok and got.attempts == 2 and got.need_rounds == 1
    bogus = FakeEngine("NEED shop/nothing.py:L1-5", FIX_VAT)  # nothing readable: normal feedback/retry path
    got = request_edits(bogus, [{"role": "user", "content": "r1"}], repo, max_new_tokens=500, temperature=0, retries=1,
                        need=NeedState(NeedSettings(), repo, _Ctx([]), "t", "S", "C"))
    assert got.ok and got.need_rounds == 0 and got.attempts == 2


def test_stop_sequence_is_sent_and_dropped_when_rejected():
    class Rejecting:
        def __init__(self):
            self.calls = []

        def converse(self, **kw):
            self.calls.append(kw)
            if "stopSequences" in kw["inferenceConfig"]:
                err = Exception("An error occurred (ValidationException): stopSequences is not supported for this model")
                err.response = {"Error": {"Code": "ValidationException"}}
                raise err
            return {"output": {"message": {"content": [{"text": "NEED a.py:L1-2"}]}}, "stopReason": "end_turn",
                    "usage": {"inputTokens": 10, "outputTokens": 3}}

    CALLS.reset()
    fake = Rejecting()
    c = BedrockEngineClient("qwen.qwen3-235b-a22b-2507-v1:0", client=fake)
    assert c.chat([{"role": "user", "content": "x"}], stop=[END_NEED]) == "NEED a.py:L1-2"
    assert c.supports_stop is False and "stopSequences" not in fake.calls[-1]["inferenceConfig"]
    c.chat([{"role": "user", "content": "y"}], stop=[END_NEED])
    assert len(fake.calls) == 3  # later calls go straight without stop sequences
    ok = BedrockEngineClient("minimax.minimax-m2.5", client=object())
    assert ok.request([{"role": "user", "content": "x"}], max_tokens=10, temperature=0, stop=[END_NEED])["inferenceConfig"]["stopSequences"] == [END_NEED]
    CALLS.reset()


def test_openai_stop_fallback(monkeypatch):
    bodies = []
    client = ComputeEngineClient("http://127.0.0.1:9/v1")

    def fake(url, body=None, timeout=None):
        bodies.append(dict(body))
        if "stop" in body:
            raise ComputeEngineError("HTTP 400 from compute engine: Unsupported parameter: 'stop' is not supported with this model")
        return {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}], "usage": {}}

    monkeypatch.setattr(client, "_request", fake)
    assert client.chat([{"role": "user", "content": "x"}], stop=[END_NEED]) == "ok"
    assert client.supports_stop is False and "stop" not in bodies[-1]


def test_openai_client_never_sends_zt_keys(monkeypatch):
    sent = {}
    client = ComputeEngineClient("http://127.0.0.1:9/v1")
    monkeypatch.setattr(client, "_request", lambda url, body=None, timeout=None: sent.update(body or {}) or
                        {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}], "usage": {}})
    client.chat_full([{"role": "user", "content": "x", "extra": True}])
    assert sent["messages"] == [{"role": "user", "content": "x"}]


# -- presets and zt code end to end ---------------------------------------------------------------
@pytest.fixture
def shop(tmp_path, capsys):
    root = tmp_path / "shop_repo"
    (root / "shop").mkdir(parents=True)
    (root / "shop/__init__.py").write_text("")
    (root / "shop/pricing.py").write_text(PRICING)
    assert main(["init", "--repo", str(root), "--rank", "32"]) == 0
    cfg = ProjectConfig.load(root)
    cfg.memory.update(calibration_samples=128)
    cfg.router.update(preset="api", focus_blocks=1, focus_ratio=0.99)  # tiny repo: force a map
    cfg.engine.update(backend="bedrock", model="minimax.minimax-m2.5", max_calls=5)
    cfg.save(overwrite=True)
    assert main(["sync", "--repo", str(root), "--quiet"]) == 0
    capsys.readouterr()
    return root


@pytest.fixture
def fake_boto3(monkeypatch):
    class Bedrock:
        def __init__(self):
            self.answers, self.calls = [], []

        def converse(self, **kw):
            self.calls.append(kw)
            text = self.answers.pop(0)
            return {"output": {"message": {"content": [{"reasoningContent": {"reasoningText": {"text": "hm"}}}, {"text": text}]}},
                    "stopReason": "end_turn", "usage": {"inputTokens": 1000, "outputTokens": 50}}

    bed = Bedrock()
    mod = types.ModuleType("boto3")
    mod.client = lambda service, **kw: bed
    cfg_mod = types.ModuleType("botocore.config")
    cfg_mod.Config = lambda **kw: kw
    monkeypatch.setitem(sys.modules, "boto3", mod)
    monkeypatch.setitem(sys.modules, "botocore", types.ModuleType("botocore"))
    monkeypatch.setitem(sys.modules, "botocore.config", cfg_mod)
    CALLS.reset()
    yield bed
    CALLS.reset()


def test_presets_are_validated_and_applied(shop):
    rc = ProjectConfig.load(shop).router_config()
    assert rc.dedup and rc.signature_extra == 20 and rc.traceback_focus_blocks == 10
    assert rc.focus_blocks == 1  # explicit keys win over the preset
    cfg = ProjectConfig.load(shop)
    cfg.router["preset"] = "turbo"
    cfg.save(overwrite=True)
    with pytest.raises(ProjectConfigError, match="preset"):
        ProjectConfig.load(shop)


def test_code_with_need_round(shop, fake_boto3, capsys):
    fake_boto3.answers = [ASK_VAT, FIX_VAT]
    code = main(["code", "prices with VAT are not rounded to 5 cents correctly", "--repo", str(shop), "--dry-run", "--json"])
    res = json.loads(capsys.readouterr().out)
    assert code == 0 and res["status"] == "dry_run" and res["need_rounds"] == 1 and res["engine_calls"] == 2
    first, second = fake_boto3.calls
    assert "NEED <path>" in first["messages"][0]["content"][0]["text"]  # offered in round 1
    assert first["inferenceConfig"]["stopSequences"] == [END_NEED] and "stopSequences" not in second["inferenceConfig"]
    lean = second["messages"]
    assert len(lean) == 1 and "Code you asked for" in lean[0]["content"][0]["text"]  # same short round 2 for every model


def test_benchmark_sample_is_balanced():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("benchmark_need", Path(__file__).parents[1] / "examples" / "benchmark_need.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    tasks = [{"set": "commit", "id": f"m{i}", "selected_as": "map-only"} for i in range(21)]
    tasks += [{"set": "commit", "id": f"c{i}", "selected_as": "body"} for i in range(6)]
    tasks += [{"set": "italiano", "id": f"i{i}", "selected_as": "map-only"} for i in range(3)]
    tasks += [{"set": "italiano", "id": f"j{i}", "selected_as": "body"} for i in range(6)]
    got = mod.pick_tasks(tasks, ["commit"], 12)
    assert [t["selected_as"] for t in got].count("map-only") == 8 and len(got) == 12
    few = mod.pick_tasks(tasks, ["italiano"], 12)  # only 3 map-only: the rest are controls
    assert [t["selected_as"] for t in few].count("map-only") == 3 and len(few) == 9
    assert len(mod.pick_tasks(tasks, ["commit", "italiano"], 0)) == len(tasks)
    assert mod._cost({"input_tokens": 1000, "output_tokens": 50}) == 1200
