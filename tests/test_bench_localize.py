import sys
from pathlib import Path

from zt_memory.bench_localize import (
    _coverage, changed_lines, is_test_path, load_loc_results, mine_localization_cases, run_localization, summarize_localization,
)
from zt_memory.encoders import HashingEncoder

sys.path.insert(0, str(Path(__file__).parent))
from test_bench_bugfix import history  # noqa: E402,F401  (module-scoped git repo with two real fixes)


def test_test_paths_across_languages():
    for rel in ("tests/test_x.py", "pkg/x_test.go", "src/a.test.ts", "src/__tests__/a.ts", "spec/a.spec.js", "conftest.py"):
        assert is_test_path(rel), rel
    for rel in ("src/click/core.py", "command.go", "source/core/Ky.ts", "testing_utils_helper.py"):
        assert not is_test_path(rel), rel


def test_changed_lines_and_coverage(history):
    cases = mine_localization_cases(history, min_words=2)
    by_query = {c.query: c for c in cases}
    disc = by_query["Fix discount: percent is a percentage, not a fraction"]
    assert disc.gold == {"calc/ops.py": [7]} and disc.has_tests
    assert changed_lines(history, disc.base, disc.fix, "calc/ops.py") == [7]
    assert "Fix docstring wording" in by_query  # localization needs no failing test: docstring fixes count
    assert _coverage({"calc/ops.py": [(5, 7)]}, disc.gold) == (1.0, 1.0, True)
    assert _coverage({"calc/ops.py": [(1, 3)]}, disc.gold) == (1.0, 0.0, False)
    assert _coverage({}, disc.gold) == (0.0, 0.0, False)


def test_run_report_and_resume(history, tmp_path):
    cases = mine_localization_cases(history, min_words=2)
    log = []
    results = run_localization(history, cases, tmp_path, HashingEncoder(dim=256), rank=32,
                               memory_overrides={"calibration_samples": 128}, echo=log.append)
    assert len(results) == 5 * len(cases)
    zt = {r.case: r for r in results if r.method == "zt"}
    disc = next(c for c in cases if "discount" in c.query)
    assert zt[disc.id].hit1 and zt[disc.id].spot_in_ctx and zt[disc.id].tokens > 0
    report = summarize_localization(results)
    assert f"{len(cases)} cases" in report and "| bm25-chunks |" in report and "| grep |" in report
    again = run_localization(history, cases, tmp_path, HashingEncoder(dim=256), rank=32,
                             memory_overrides={"calibration_samples": 128}, echo=log.append)
    assert len(again) == len(results) == len(load_loc_results(tmp_path / "loc_results.jsonl"))
