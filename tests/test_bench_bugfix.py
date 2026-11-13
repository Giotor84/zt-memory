import os
import subprocess
import sys
from pathlib import Path

import pytest

from zt_memory.bench_bugfix import (
    bm25_rank_files, file_pack_messages, load_results, mcnemar_exact, mine_cases, run_benchmark, summarize,
)
from zt_memory.encoders import HashingEncoder
from zt_memory.engine_client import ComputeEngineClient

sys.path.insert(0, str(Path(__file__).parent))
from test_code_command import FakeLlama  # noqa: E402

OPS_BUGGY = '''def add(a, b):
    return a + b


def discount(price, percent):
    """Price after a percentage discount (percent in 0..100)."""
    return price - price * percent
'''
OPS_FIXED = OPS_BUGGY.replace("price - price * percent", "price - price * percent / 100")
UTILS_BUGGY = '''def clamp(value, low, high):
    """Clamp value into [low, high]."""
    return max(low, min(value, high - 1))


def mean(xs):
    return sum(xs) / len(xs)
'''
UTILS_FIXED = UTILS_BUGGY.replace("min(value, high - 1)", "min(value, high)")


def _git(root, *args, date=None):
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    if date:
        env["GIT_AUTHOR_DATE"] = env["GIT_COMMITTER_DATE"] = date
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, env=env)


def _commit(root, files, message, n):
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, newline="\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", message, date=f"2026-01-0{n}T12:00:00")


@pytest.fixture(scope="module")
def history(tmp_path_factory):
    root = tmp_path_factory.mktemp("calc_repo")
    _git(root, "init", "-q")
    _commit(root, {"calc/__init__.py": "", "calc/ops.py": OPS_BUGGY, "calc/utils.py": UTILS_BUGGY,
                   "tests/test_ops.py": "from calc.ops import add\n\n\ndef test_add():\n    assert add(1, 2) == 3\n"}, "initial", 1)
    _commit(root, {"calc/ops.py": OPS_FIXED, "tests/test_ops.py": "from calc.ops import add, discount\n\n\ndef test_add():\n"
                   "    assert add(1, 2) == 3\n\n\ndef test_discount():\n    assert discount(200, 10) == 180\n"},
            "Fix discount: percent is a percentage, not a fraction", 2)
    _commit(root, {"README.md": "# calc\n"}, "Add readme", 3)
    _commit(root, {"calc/utils.py": UTILS_FIXED, "tests/test_utils.py": "from calc.utils import clamp\n\n\ndef test_upper_bound():\n"
                   "    assert clamp(15, 0, 10) == 10\n"}, "Fix clamp returning high - 1 at the upper bound", 4)
    _commit(root, {"calc/utils.py": UTILS_FIXED.replace('"""Clamp value into [low, high]."""', '"""Clamp."""'),
                   "tests/test_utils.py": "from calc.utils import clamp, mean\n\n\ndef test_upper_bound():\n    assert clamp(15, 0, 10) == 10\n\n\n"
                   "def test_mean():\n    assert mean([1, 3]) == 2\n"}, "Fix docstring wording", 5)  # tests pass before: not a bug case
    return root


def _fix(path, search, replace):
    return f"{path}\n<<<<<<< SEARCH\n{search}\n=======\n{replace}\n>>>>>>> REPLACE\n"


DISCOUNT_FIX = _fix("calc/ops.py", "    return price - price * percent", "    return price - price * percent / 100")
CLAMP_FIX = _fix("calc/utils.py", "    return max(low, min(value, high - 1))", "    return max(low, min(value, high))")


def test_mining_keeps_only_reproducible_fixes(history, tmp_path):
    cases = mine_cases(history, tmp_path, echo=lambda s: None)
    assert [c.instruction for c in cases] == ["Fix discount: percent is a percentage, not a fraction",
                                              "Fix clamp returning high - 1 at the upper bound"]
    assert cases[0].source_files == ("calc/ops.py",) and cases[0].test_files == ("tests/test_ops.py",)
    assert cases[1].test_files == ("tests/test_utils.py",) and "assert" in cases[1].failure_excerpt
    assert subprocess.run(["git", "worktree", "list"], cwd=history, capture_output=True, text=True).stdout.count("\n") == 1


def test_bm25_ranking_and_whole_file_pack(tmp_path):
    (tmp_path / "a.py").write_text("def discount(price, percent):\n    return price\n" * 40)
    (tmp_path / "b.py").write_text("def unrelated():\n    pass\n")
    (tmp_path / "c.py").write_text("PERCENT = 1\n" + "x = 1\n" * 400)
    ranked = bm25_rank_files(tmp_path, ["a.py", "b.py", "c.py"], "discount uses the percent wrongly")
    assert ranked[0][0] == "a.py" and dict(ranked)["b.py"] == 0
    count = lambda s: len(s) // 4  # noqa: E731
    messages, tokens, files = file_pack_messages(tmp_path, ranked, "fix discount", budget=900, count=count)
    assert tokens + 32 <= 900 and files == ["a.py", "c.py"] and "[truncated]" in messages[1]["content"]
    assert "b.py" not in messages[1]["content"]  # score 0: never packed


def test_mcnemar_exact():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(0, 5) == pytest.approx(2 / 32)
    assert mcnemar_exact(3, 3) == 1.0


def test_end_to_end_run_report_and_resume(history, tmp_path):
    cases = mine_cases(history, tmp_path, echo=lambda s: None)
    # per case, fixed condition order: bm25-files, zt, bm25-matched (retries=0: one answer each)
    llama = FakeLlama([DISCOUNT_FIX, DISCOUNT_FIX, "no idea", "I would change clamp", CLAMP_FIX, CLAMP_FIX], n_ctx=8192)
    try:
        client = ComputeEngineClient(llama.url)
        log: list[str] = []
        results = run_benchmark(history, cases, tmp_path, client, HashingEncoder(dim=256), rank=32, max_new_tokens=512,
                                retries=0, memory_overrides={"calibration_samples": 128}, echo=log.append)
        got = {(r.case, r.condition): r for r in results}
        c1, c2 = cases[0].id, cases[1].id
        assert got[(c1, "bm25-files")].resolved and got[(c1, "zt")].resolved
        assert got[(c1, "bm25-matched")].status == "no_edits"
        assert got[(c2, "bm25-files")].status == "no_edits" and got[(c2, "zt")].resolved and got[(c2, "bm25-matched")].resolved
        assert got[(c1, "zt")].gold_in_context == 1.0 and got[(c1, "zt")].patched_files == ["calc/ops.py"]
        # matched control never exceeds the tokens zt used; the fill-the-budget baseline uses at least as many
        for c in (c1, c2):
            assert got[(c, "bm25-matched")].context_tokens <= got[(c, "zt")].context_tokens + 32
        report = summarize(results)
        assert "| zt | 2/2 (100%)" in report and "| bm25-files | 1/2 (50%)" in report
        assert "| bm25-files vs zt | 0 | 1 | 1 | 0 |" in report
        n_requests = len(llama.requests)
        again = run_benchmark(history, cases, tmp_path, client, HashingEncoder(dim=256), rank=32, max_new_tokens=512,
                              retries=0, memory_overrides={"calibration_samples": 128}, echo=log.append)
        assert len(llama.requests) == n_requests and len(again) == 6  # resumed: nothing re-asked
        assert len(load_results(tmp_path / "results.jsonl")) == 6
        assert (history / "calc/utils.py").read_text().startswith(UTILS_FIXED.split("\n")[0])  # main tree untouched
    finally:
        llama.close()


def test_bedrock_call_limit_stops_the_run_and_resume_continues(history, tmp_path):
    from zt_memory.engine_bedrock import CALLS, BedrockEngineClient, EngineBudgetExceeded

    class FakeBedrock:
        def __init__(self, answers):
            self.answers, self.calls = list(answers), 0

        def converse(self, **kwargs):
            self.calls += 1
            text = self.answers.pop(0) if self.answers else "no idea"
            return {"output": {"message": {"content": [{"reasoningContent": {"reasoningText": {"text": "hm"}}}, {"text": text}]}},
                    "stopReason": "end_turn", "usage": {"inputTokens": 100, "outputTokens": 20}}

    cases = mine_cases(history, tmp_path, echo=lambda s: None)
    CALLS.reset()
    try:
        fake = FakeBedrock([DISCOUNT_FIX, DISCOUNT_FIX, "no idea", "I would change clamp", CLAMP_FIX, CLAMP_FIX])
        client = BedrockEngineClient("qwen.qwen3-235b-a22b-2507-v1:0", client=fake, max_calls=4)
        kw = dict(rank=32, max_new_tokens=512, retries=0, served_ctx=8192, memory_overrides={"calibration_samples": 128},
                  echo=lambda s: None)
        with pytest.raises(EngineBudgetExceeded):
            run_benchmark(history, cases, tmp_path, client, HashingEncoder(dim=256), **kw)
        partial = load_results(tmp_path / "results.jsonl")
        assert len(partial) == 4 and all(r.status != "error" for r in partial)  # the 5th call was refused, not recorded
        CALLS.reset()  # a new process
        results = run_benchmark(history, cases, tmp_path, client, HashingEncoder(dim=256), **kw)
        assert len(results) == 6 and fake.calls == 6
    finally:
        CALLS.reset()
