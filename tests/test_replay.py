import shutil
import subprocess

import pytest

from zt_memory import HashingEncoder
from zt_memory.replay import clean_message, default_configs, paired_bootstrap, replay

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")


def _commit(repo, files, message):
    for rel, body in files.items():
        p = repo / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body, encoding="utf-8")
    env = ["-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(repo), *env, "commit", "-qm", message], check=True)


@pytest.fixture
def history(tmp_path):
    repo = tmp_path / "src_repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    pool = "def acquire_connection(pool):\n    return pool.pop()\n\n\ndef release_connection(pool, c):\n    pool.append(c)\n"
    slug = "def slugify_title(t):\n    return t.lower().replace(' ', '-')\n"
    _commit(repo, {"app/pool.py": pool, "app/slug.py": slug, "README.md": "# demo\n"}, "initial import of the demo app")
    for i in range(1, 6):
        pool = pool.replace("return pool.pop()", f"return pool.pop()  # v{i}")
        _commit(repo, {"app/pool.py": pool}, f"fix acquire connection when the pool is empty, round {i}")
        slug = slug.replace("lower()", "lower().strip()") if i == 3 else slug + f"\n# note {i}\n"
        _commit(repo, {"app/slug.py": slug}, f"improve slugify title handling of spaces, step {i}")
    return repo


def test_replay_reports_paired_metrics(history, tmp_path):
    report = replay(str(history), tmp_path / "work", HashingEncoder(dim=256), commits=50,
                    memory_overrides={"rank": 16, "calibration_samples": 64, "lock_timeout_s": 0.2},
                    cutoff_alphas=(0.0, 0.5))
    assert report.cases == 10  # every non-root commit modifies an existing indexable file
    assert set(default_configs()) | {"R_recency_only"} == set(report.configs)
    for summary in report.configs.values():
        assert 0.0 <= summary["mrr"] <= 1.0 and summary["cases"] == 10
    assert report.configs["L_lexical_only"]["hit@1"] >= 0.8  # messages name the touched symbols
    assert any("physics prior" in k for k in report.comparisons)
    assert report.cutoff[1]["avg_tokens"] <= report.cutoff[0]["avg_tokens"]
    assert (history / "app/pool.py").read_text().count("# v5") == 1  # source repo untouched


def test_message_cleaning_and_bootstrap():
    msg = clean_message("Fix retry on 503 (#1234)", "Retries now back off.\n\nSigned-off-by: x <x@y>")
    assert msg == "Fix retry on 503 Retries now back off."
    same = paired_bootstrap([0.5, 0.2, 0.9], [0.5, 0.2, 0.9])
    assert same == {"diff": 0.0, "lo": 0.0, "hi": 0.0}
    better = paired_bootstrap([0.0] * 30, [1.0] * 30)
    assert better["lo"] > 0
