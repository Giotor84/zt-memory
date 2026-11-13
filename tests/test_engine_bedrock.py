"""Bedrock backend (Converse API) with an injected fake boto3 client: no network, boto3 not required."""
import json
import sys
import types

import pytest

from zt_memory.cli import main
from zt_memory.engine_bedrock import CALLS, BedrockEngineClient, EngineBudgetExceeded, model_profile
from zt_memory.engine_client import ComputeEngineClient, ComputeEngineError, make_engine_client
from zt_memory.project import ENGINE_DEFAULTS, ProjectConfig, ProjectConfigError

MINIMAX = "minimax.minimax-m2.5"
QWEN = "qwen.qwen3-235b-a22b-2507-v1:0"


@pytest.fixture(autouse=True)
def _fresh_counter():
    CALLS.reset()
    yield
    CALLS.reset()


def reply(text=None, *, reasoning=None, stop="end_turn", usage=(120, 30)):
    blocks = []
    if reasoning is not None:
        blocks.append({"reasoningContent": {"reasoningText": {"text": reasoning, "signature": "sig"}}})
    if text is not None:
        blocks.append({"text": text})
    return {"output": {"message": {"role": "assistant", "content": blocks}}, "stopReason": stop,
            "usage": {"inputTokens": usage[0], "outputTokens": usage[1], "totalTokens": sum(usage)}}


class FakeBedrock:
    """Records every converse() call; answers from a queue (an Exception instance is raised)."""

    def __init__(self, *answers):
        self.answers, self.calls = list(answers), []

    def converse(self, **kwargs):
        self.calls.append(kwargs)
        answer = self.answers.pop(0) if self.answers else reply("no idea")
        if isinstance(answer, BaseException):
            raise answer
        return answer


def client(model=MINIMAX, *answers, **kw):
    fake = FakeBedrock(*answers)
    return BedrockEngineClient(model, client=fake, **kw), fake


# -- request mapping ------------------------------------------------------------------------------
def test_request_mapping_system_order_limits_stop():
    c, fake = client(QWEN, reply("done"))
    msgs = [{"role": "system", "content": "You fix bugs."}, {"role": "system", "content": "Use SEARCH/REPLACE."},
            {"role": "user", "content": "task"}, {"role": "assistant", "content": "answer 1"},
            {"role": "user", "content": "feedback"}, {"role": "user", "content": "more"}]
    res = c.chat_full(msgs, max_tokens=50_000, temperature=0.3, stop=["</end>"])
    req = fake.calls[0]
    assert req["modelId"] == QWEN
    assert req["system"] == [{"text": "You fix bugs.\n\nUse SEARCH/REPLACE."}]
    assert [m["role"] for m in req["messages"]] == ["user", "assistant", "user"]  # consecutive users merged
    assert req["messages"][2]["content"] == [{"text": "feedback\n\nmore"}]
    assert req["inferenceConfig"] == {"maxTokens": 8192, "temperature": 0.3, "stopSequences": ["</end>"]}
    assert res.text == "done"


def test_profiles_and_extra_fields():
    assert model_profile(MINIMAX).name == "minimax-m2.5" and model_profile(MINIMAX).reasoning
    assert model_profile(QWEN).name == "qwen3" and model_profile("eu.qwen.qwen3-32b").name == "qwen3"
    assert model_profile("mistral.large").name == "generic"
    q, qfake = client(QWEN, reply("a"))
    q.chat([{"role": "user", "content": "hi"}])
    assert qfake.calls[0]["additionalModelRequestFields"] == {"enable_thinking": False}
    m, mfake = client(MINIMAX, reply("a"))
    m.chat([{"role": "user", "content": "hi"}])
    assert "additionalModelRequestFields" not in mfake.calls[0]  # nothing meant for Qwen goes to MiniMax
    assert m.context_size() == 196_000 and q.context_size() == 256_000
    assert BedrockEngineClient("mistral.large", ctx=32_000, client=FakeBedrock()).context_size() == 32_000
    assert m.count_tokens("abc") is None and m.estimate_tokens("x" * 300) == 110


# -- answers --------------------------------------------------------------------------------------
def test_reasoning_is_ignored_and_usage_mapped():
    c, _ = client(MINIMAX, reply("shop/a.py\n<<<<<<< SEARCH", reasoning="<<<<<<< SEARCH fake edit in the thoughts", usage=(900, 400)))
    res = c.chat_full([{"role": "user", "content": "fix"}])
    assert res.text == "shop/a.py\n<<<<<<< SEARCH"
    assert (res.prompt_tokens, res.completion_tokens, res.finish_reason) == (900, 400, "stop")
    assert res.seconds >= 0
    c2, _ = client(QWEN, reply("cut", stop="max_tokens"), reply("x", stop="guardrail_intervened"))
    assert c2.chat_full([{"role": "user", "content": "a"}]).finish_reason == "length"
    assert c2.chat_full([{"role": "user", "content": "a"}]).finish_reason == "guardrail_intervened"


def test_empty_answer_with_end_turn_is_retried_once():
    c, fake = client(MINIMAX, reply(None, reasoning="thinking..."), reply("OK"))
    assert c.chat([{"role": "user", "content": "x"}]) == "OK" and len(fake.calls) == 2
    c, fake = client(MINIMAX, reply(""), reply(""))
    with pytest.raises(ComputeEngineError) as err:
        c.chat([{"role": "user", "content": "x"}])
    assert err.value.empty_response and not err.value.max_tokens_exhausted and len(fake.calls) == 2
    q, qfake = client(QWEN, reply(""))  # the Qwen profile does not retry
    with pytest.raises(ComputeEngineError):
        q.chat([{"role": "user", "content": "x"}])
    assert len(qfake.calls) == 1


def test_empty_answer_with_max_tokens_fails_at_once():
    c, fake = client(MINIMAX, reply(None, reasoning="long thoughts", stop="max_tokens"), reply("never reached"))
    with pytest.raises(ComputeEngineError) as err:
        c.chat([{"role": "user", "content": "x"}], max_tokens=16)
    assert err.value.empty_response and err.value.max_tokens_exhausted
    assert "aumenta max_new_tokens" in str(err.value) and len(fake.calls) == 1


def test_aws_errors_become_engine_errors_and_bugs_propagate():
    class ClientError(Exception):  # shape of botocore.exceptions.ClientError
        def __init__(self):
            super().__init__("An error occurred (ThrottlingException): Rate exceeded")
            self.response = {"Error": {"Code": "ThrottlingException", "Message": "Rate exceeded"}}

    c, _ = client(MINIMAX, ClientError())
    with pytest.raises(ComputeEngineError) as err:
        c.chat([{"role": "user", "content": "x"}])
    assert err.value.aws_code == "ThrottlingException" and "ThrottlingException" in str(err.value)
    c, _ = client(MINIMAX, TypeError("bug in the caller"))
    with pytest.raises(TypeError):
        c.chat([{"role": "user", "content": "x"}])


# -- client construction, health, limits ----------------------------------------------------------
class FakeConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeBoto3:
    def __init__(self):
        self.made, self.client_obj = [], FakeBedrock()

    def client(self, service, **kwargs):
        self.made.append((service, kwargs))
        return self.client_obj


def test_proxy_timeout_retries_reach_botocore_config():
    boto = FakeBoto3()
    c = BedrockEngineClient(MINIMAX, region="eu-central-1", proxy="http://proxy.example:3128", timeout_s=600, aws_retries=5,
                            boto3_module=boto, config_class=FakeConfig)
    assert c.health()
    service, kwargs = boto.made[0]
    assert service == "bedrock-runtime" and kwargs["region_name"] == "eu-central-1"
    assert kwargs["config"].kwargs == {"proxies": {"http": "http://proxy.example:3128", "https": "http://proxy.example:3128"},
                                       "connect_timeout": 10, "read_timeout": 600.0,
                                       "retries": {"max_attempts": 5, "mode": "standard"}}
    no_proxy = FakeBoto3()
    BedrockEngineClient(MINIMAX, boto3_module=no_proxy, config_class=FakeConfig).health()
    assert no_proxy.made[0][1]["config"].kwargs["proxies"] is None  # HTTPS_PROXY or direct


def test_health_never_calls_the_model_and_reports_missing_boto3(monkeypatch):
    c, fake = client(MINIMAX)
    assert c.health() and c.health() and fake.calls == []
    monkeypatch.setitem(sys.modules, "boto3", None)  # import boto3 -> ImportError
    bare = BedrockEngineClient(MINIMAX)
    assert not bare.health() and "bedrock extra" in bare.last_error


def test_max_calls_and_usage_log(tmp_path):
    log = tmp_path / ".zt" / "engine_usage.jsonl"
    c, fake = client(QWEN, reply("a", usage=(10, 2)), reply("b", usage=(11, 3)), reply("c"), max_calls=2, usage_log=log)
    c.chat([{"role": "user", "content": "1"}])
    c.chat([{"role": "user", "content": "2"}])
    with pytest.raises(EngineBudgetExceeded, match="max_calls reached"):
        c.chat([{"role": "user", "content": "3"}])
    assert len(fake.calls) == 2  # the third call never left the process
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert [(r["model"], r["input_tokens"], r["output_tokens"], r["stop_reason"]) for r in rows] == [
        (QWEN, 10, 2, "end_turn"), (QWEN, 11, 3, "end_turn")]
    assert all("time" in r and r["seconds"] >= 0 for r in rows)


def test_factory_and_probe(tmp_path):
    openai = make_engine_client(dict(ENGINE_DEFAULTS))
    assert isinstance(openai, ComputeEngineClient)
    bed = make_engine_client({**ENGINE_DEFAULTS, "backend": "bedrock", "model": MINIMAX, "ctx": None, "max_calls": 7,
                              "proxy": "http://p:3128"}, usage_log=tmp_path / "u.jsonl")
    assert isinstance(bed, BedrockEngineClient) and bed.max_calls == 7 and bed.proxy == "http://p:3128"
    assert bed.context_size() == 196_000
    with pytest.raises(ComputeEngineError, match="unknown"):
        make_engine_client({**ENGINE_DEFAULTS, "backend": "vertex"})
    c, fake = client(MINIMAX, reply("OK", reasoning="The user wants OK."))
    assert c.probe() == "OK" and fake.calls[0]["inferenceConfig"]["maxTokens"] == 128


# -- config and CLI -------------------------------------------------------------------------------
BUGGY = "def apply_discount(price: float, percent: float) -> float:\n    \"\"\"Price after a percentage discount (0-100).\"\"\"\n    return price - price * percent\n"
FIX = ("shop/pricing.py\n<<<<<<< SEARCH\n    return price - price * percent\n=======\n"
       "    return price - price * percent / 100\n>>>>>>> REPLACE\n")


@pytest.fixture
def shop(tmp_path, capsys):
    root = tmp_path / "shop_repo"
    (root / "shop").mkdir(parents=True)
    (root / "shop/__init__.py").write_text("")
    (root / "shop/pricing.py").write_text(BUGGY)
    assert main(["init", "--repo", str(root), "--rank", "32"]) == 0
    cfg = ProjectConfig.load(root)
    cfg.memory.update(calibration_samples=128)
    cfg.engine.update(backend="bedrock", model=MINIMAX, region="eu-central-1", proxy="http://proxy.example:3128", max_calls=5)
    cfg.save(overwrite=True)
    capsys.readouterr()
    return root


@pytest.fixture
def fake_boto3(monkeypatch):
    boto = FakeBoto3()
    mod = types.ModuleType("boto3")
    mod.client = boto.client
    botocore, config_mod = types.ModuleType("botocore"), types.ModuleType("botocore.config")
    config_mod.Config = FakeConfig
    monkeypatch.setitem(sys.modules, "boto3", mod)
    monkeypatch.setitem(sys.modules, "botocore", botocore)
    monkeypatch.setitem(sys.modules, "botocore.config", config_mod)
    return boto


def test_config_accepts_new_keys_and_rejects_unknown_backend(shop):
    cfg = ProjectConfig.load(shop)
    eng = cfg.engine_settings()
    assert eng["backend"] == "bedrock" and eng["ctx"] is None and eng["max_new_tokens"] == 8192 and eng["max_calls"] == 5
    assert ProjectConfig.load(shop).engine_usage_log == shop / ".zt" / "engine_usage.jsonl"
    cfg.engine["backend"] = "sagemaker"
    cfg.save(overwrite=True)
    with pytest.raises(ProjectConfigError, match="backend"):
        ProjectConfig.load(shop)


def test_engine_command_shows_config_and_probes_only_on_request(shop, fake_boto3, capsys):
    assert main(["engine", "--repo", str(shop), "--json"]) == 0
    info = json.loads(capsys.readouterr().out)
    assert info["profile"] == "minimax-m2.5" and info["boto3"] and info["probe"] is None
    assert fake_boto3.client_obj.calls == []  # no paid call without --probe
    fake_boto3.client_obj.answers = [reply("OK", reasoning="Reply OK.")]
    assert main(["engine", "--repo", str(shop), "--probe", "--json"]) == 0
    info = json.loads(capsys.readouterr().out)
    assert info["probe"]["ok"] and len(fake_boto3.client_obj.calls) == 1
    assert (shop / ".zt" / "engine_usage.jsonl").read_text().count("\n") == 1


def test_engine_command_without_boto3_explains(shop, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "boto3", None)
    assert main(["engine", "--repo", str(shop)]) == 1
    assert "bedrock extra" in capsys.readouterr().out


def test_code_dry_run_with_bedrock(shop, fake_boto3, capsys):
    fake_boto3.client_obj.answers = [reply(FIX, reasoning="The bug is the missing / 100.", usage=(1500, 200))]
    code = main(["code", "apply_discount gives wrong prices: 10% of 200 must give 180", "--repo", str(shop),
                 "--dry-run", "--json"])
    out = capsys.readouterr()
    res = json.loads(out.out)
    assert code == 0 and res["status"] == "dry_run" and res["engine_calls"] == 1
    assert res["prompt_tokens"] == 1500 and res["completion_tokens"] == 200
    assert (shop / "shop/pricing.py").read_text() == BUGGY  # dry run writes nothing
    req = fake_boto3.client_obj.calls[0]
    assert req["system"][0]["text"] and req["messages"][0]["role"] == "user"
    assert "apply_discount" in req["messages"][0]["content"][0]["text"]


def test_code_maps_aws_errors_to_exit_6(shop, fake_boto3, capsys):
    class ClientError(Exception):
        response = {"Error": {"Code": "AccessDeniedException"}}

    fake_boto3.client_obj.answers = [ClientError("not allowed")]
    assert main(["code", "fix discount", "--repo", str(shop), "--dry-run"]) == 6
    assert "AccessDeniedException" in capsys.readouterr().err
