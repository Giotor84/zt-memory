import json
import os
import shutil
import subprocess
import sys

import pytest

from zt_memory.cli import HOOK_MARKER, main
from zt_memory.project import ProjectConfig, ProjectConfigError


def _run(capsys, *argv):
    code = main(list(argv))
    out = capsys.readouterr()
    return code, out.out, out.err


@pytest.fixture
def project(repo, capsys):
    code, out, _ = _run(capsys, "init", "--repo", str(repo), "--rank", "32")
    assert code == 0 and "wrote" in out
    # keep the suite fast: small calibration corpus
    cfg = ProjectConfig.load(repo)
    cfg.memory["calibration_samples"] = 128
    cfg.memory["lock_timeout_s"] = 0.2
    cfg.save(overwrite=True)
    return repo


def test_init_writes_valid_config_and_refuses_overwrite(project, capsys):
    cfg = ProjectConfig.load(project)
    assert cfg.encoder["spec"] == "hashing" and cfg.memory["rank"] == 32
    assert cfg.store == project / ".zt" / "store"
    assert cfg.memory_config().ram_budget_bytes == 1536 * 2**20
    code, _, err = _run(capsys, "init", "--repo", str(project))
    assert code == 1 and "exists" in err


def test_unknown_config_keys_are_rejected(project):
    path = project / ".zt" / "config.toml"
    path.write_text(path.read_text() + "\n[bogus]\nx = 1\n")
    with pytest.raises(ProjectConfigError):
        ProjectConfig.load(project)


def test_old_router_keys_still_load(project):
    cfg = ProjectConfig.load(project)
    cfg.router.update(geodesic_samples=3, geodesic_k=4, geodesic_seeds=5)  # written by versions with the geodesic tree
    cfg.save(overwrite=True)
    with pytest.warns(UserWarning, match="no longer used"):
        rc = ProjectConfig.load(project).router_config()
    assert rc.expansion_seeds == 5


def test_sync_status_route_roundtrip(project, capsys, monkeypatch):
    code, out, _ = _run(capsys, "sync", "--repo", str(project), "--json")
    first = json.loads(out)
    assert code == 0 and first["files_added"] == 5 and first["chunks_added"] > 0
    code, out, _ = _run(capsys, "sync", "--repo", str(project), "--json")
    assert json.loads(out)["files_added"] == 0
    (project / "src/app/config.py").write_text((project / "src/app/config.py").read_text() + "\n\ndef extra_hook():\n    pass\n")
    monkeypatch.chdir(project / "src")  # project discovered from a subdirectory
    code, out, _ = _run(capsys, "status", "--json")
    status = json.loads(out)
    assert status["pending"]["changed"] == ["src/app/config.py"]
    code, out, err = _run(capsys, "route", "where is extra_hook defined")  # auto-sync first
    assert code == 0 and "extra_hook" in out and "## Task" in out and "[zt]" in err
    code, out, _ = _run(capsys, "route", "parse json config", "--json", "--budget", "2000")
    payload = json.loads(out)
    assert payload["token_estimate"] <= 2000 and payload["messages"][0]["role"] == "system"


def test_locked_store_exit_code(project, capsys):
    holder = ProjectConfig.load(project).open_manager()
    try:
        code, _, err = _run(capsys, "sync", "--repo", str(project))
        assert code == 3 and "locked" in err
    finally:
        holder.close()


def test_missing_project_is_an_error(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    code, _, err = _run(capsys, "sync")
    assert code == 1 and "zt init" in err


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_hook_install_chain_uninstall(project, capsys):
    subprocess.run(["git", "init", "-q", str(project)], check=True)
    hooks = project / ".git" / "hooks"
    foreign = hooks / "post-commit"
    foreign.write_text("#!/bin/sh\necho mine\n")
    code, _, err = _run(capsys, "hook", "install", "--repo", str(project))
    assert code == 1 and "--chain" in err  # never clobber a user hook silently
    code, _, _ = _run(capsys, "hook", "install", "--repo", str(project), "--chain")
    assert code == 0
    for name in ("post-commit", "post-merge", "post-checkout", "post-rewrite"):
        text = (hooks / name).read_text()
        assert HOOK_MARKER in text and "zt_memory.cli sync --quiet" in text
        if sys.platform != "win32":  # no exec bits on NTFS; Git for Windows runs hooks regardless
            assert (hooks / name).stat().st_mode & 0o111
    assert "post-commit.pre-zt" in (hooks / "post-commit").read_text()
    code, _, _ = _run(capsys, "hook", "uninstall", "--repo", str(project))
    assert code == 0 and foreign.read_text() == "#!/bin/sh\necho mine\n"
    assert not (hooks / "post-merge").exists()
    exclude = (project / ".git" / "info" / "exclude")
    _run(capsys, "init", "--repo", str(project), "--force", "--rank", "32")
    assert "/.zt/" in exclude.read_text()


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
@pytest.mark.skipif(sys.platform == "win32", reason="detached background hooks are verified on POSIX only")
def test_hook_actually_syncs_after_commit(project, capsys):
    import time

    subprocess.run(["git", "init", "-q", str(project)], check=True)
    _run(capsys, "sync", "--repo", str(project))
    _run(capsys, "hook", "install", "--repo", str(project))
    (project / "src/app/committed.py").write_text("def from_commit():\n    return 42\n")
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    subprocess.run(["git", "-C", str(project), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(project), "commit", "-qm", "x"], check=True, env=env)
    cfg = ProjectConfig.load(project)
    deadline = time.time() + 60
    while time.time() < deadline:  # the hook runs detached in the background
        try:
            with cfg.open_manager() as mgr:
                if mgr.record_by_key("chunk:src/app/committed.py::from_commit") is not None:
                    return
        except Exception:
            pass
        time.sleep(0.5)
    pytest.fail("post-commit hook did not sync the new file")


def test_sync_accepts_per_run_encoder_overrides(project, capsys):
    code, out, _ = _run(capsys, "sync", "--repo", str(project), "--json", "--device", "cpu", "--dtype", "float32", "--batch-size", "2")
    assert code == 0 and json.loads(out)["files_added"] == 5
    assert "device" not in ProjectConfig.load(project).encoder  # overrides are not persisted
