import os
import shutil
import subprocess
import time

import pytest

from zt_memory.config import MemoryConfig
from zt_memory.storage import StoreLockedError
from zt_memory.watch import RepoWatcher


def _bump_mtime(path):
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))


@pytest.fixture
def counted(make_manager, monkeypatch):
    mgr = make_manager()
    calls = []
    original = mgr._read_source_stat

    def spy(rel):
        calls.append(rel)
        return original(rel)

    monkeypatch.setattr(mgr, "_read_source_stat", spy)
    return mgr, calls


def test_noop_sync_reads_nothing(repo, counted):
    mgr, calls = counted
    first = mgr.sync()
    assert first.added == 5 and first.ingest.chunks_added > 0
    calls.clear()
    again = mgr.sync()
    assert (again.added, again.changed, again.removed, again.touched) == (0, 0, 0, 0)
    assert calls == [] and again.ingest.chunks_added == again.ingest.chunks_updated == 0


def test_only_edited_files_are_ingested(repo, counted):
    mgr, calls = counted
    mgr.sync()
    calls.clear()
    path = repo / "src/util/strings.py"
    path.write_text(path.read_text().replace("url slug", "URL-safe slug"))
    _bump_mtime(path)
    rep = mgr.sync()
    assert rep.changed == 1 and calls == ["src/util/strings.py"]
    assert rep.ingest.chunks_updated == 1


def test_touch_without_edit_refreshes_stat_only(repo, counted):
    mgr, calls = counted
    mgr.sync()
    calls.clear()
    _bump_mtime(repo / "src/app/db.py")
    rep = mgr.sync()
    assert (rep.touched, rep.changed) == (1, 0) and calls == []
    assert mgr.sync().touched == 0  # refreshed stat is persisted


def test_added_and_removed_files(repo, counted):
    mgr, _ = counted
    mgr.sync()
    (repo / "src/app/new_mod.py").write_text("def brand_new():\n    return 1\n")
    (repo / "src/util/strings.py").unlink()
    rep = mgr.sync()
    assert (rep.added, rep.removed) == (1, 1)
    assert mgr.record_by_key("chunk:src/app/new_mod.py::brand_new") is not None
    assert mgr.record_by_key("file:src/util/strings.py") is None
    assert mgr.scan_changes().pending == []


def test_chunker_change_forces_full_rescan(repo, counted):
    mgr, _ = counted
    mgr.sync()
    with mgr.meta.transaction():
        mgr.meta.set_meta("file_state_chunker", "old-chunker")
    assert mgr.scan_changes().full
    rep = mgr.sync()
    assert rep.full and rep.changed == 5 and rep.ingest.chunks_updated == 0  # re-chunked, nothing actually changed


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_gitignore_is_respected(repo, make_manager):
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / ".gitignore").write_text("generated/\n")
    (repo / "generated").mkdir()
    (repo / "generated/huge_client.py").write_text("def autogen():\n    pass\n")
    mgr = make_manager()
    mgr.sync()
    assert mgr.record_by_key("file:generated/huge_client.py") is None
    assert mgr.record_by_key("file:src/app/db.py") is not None  # untracked but not ignored: indexed
    plain = make_manager(storage_dir=repo.parent / "store2", respect_gitignore=False)
    plain.sync()
    assert plain.record_by_key("file:generated/huge_client.py") is not None


def test_exclude_globs_hide_files(repo, make_manager):
    mgr = make_manager(exclude_globs="strings.py, src/app/db.py")
    mgr.sync()
    assert mgr.record_by_key("file:src/util/strings.py") is None
    assert mgr.record_by_key("file:src/app/db.py") is None
    assert mgr.record_by_key("file:src/app/config.py") is not None
    assert mgr.scan_changes().pending == []


def test_exclude_globs_are_case_insensitive_and_block_explicit_ingest(repo, make_manager):
    from zt_memory.discovery import is_excluded

    mgr = make_manager(exclude_globs="STRINGS.PY, fixtures/*.json")
    cfg = mgr.config
    assert is_excluded("src/util/strings.py", cfg) and is_excluded("tests\\fixtures\\case1.json", cfg)
    assert not is_excluded("fixtures/sub/case1.json", cfg)  # `*` stays within one path segment
    mgr.sync()
    mgr.ingest_files(["src/util/strings.py"])  # e.g. `zt code` touching the file: still never indexed
    assert mgr.record_by_key("file:src/util/strings.py") is None


def test_glob_added_later_removes_already_indexed_file(repo, make_manager):
    first = make_manager()
    first.sync()
    assert first.record_by_key("file:src/util/strings.py") is not None
    first.close()
    later = make_manager(exclude_globs="strings.py")
    report = later.sync()
    assert report.removed == 1 and later.record_by_key("file:src/util/strings.py") is None
    assert not any(r.path == "src/util/strings.py" for r in later.records(range(later.meta.high_water())).values())


def test_store_lock_is_exclusive(repo, make_manager):
    first = make_manager()
    with pytest.raises(StoreLockedError):
        make_manager()
    first.close()
    first.close()  # idempotent
    make_manager().close()


def test_watcher_debounces_and_syncs(repo, make_manager):
    t = [100.0]
    config = MemoryConfig(storage_dir=repo.parent / "store", repo_root=repo)
    syncs = []
    w = RepoWatcher(config, lambda: make_manager(), debounce_s=1.0, clock=lambda: t[0], on_sync=syncs.append)
    assert w.poll_once() is not None  # first poll: catch up with whatever changed while not watching
    assert w.poll_once() is None  # quiet tree: nothing to do
    path = repo / "src/app/config.py"
    path.write_text(path.read_text() + "\n\ndef late_addition():\n    return 2\n")
    _bump_mtime(path)
    assert w.poll_once() is None  # change seen, waiting for quiet
    t[0] += 0.5
    assert w.poll_once() is None  # still inside the debounce window
    t[0] += 0.6
    rep = w.poll_once()
    assert rep is not None and rep.changed == 1 and rep.ingest.chunks_added == 1
    assert len(syncs) == 2
