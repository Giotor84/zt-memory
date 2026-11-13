import sqlite3

import pytest

from zt_memory import NodeKind
from zt_memory.lexical import LexicalQuery, build_query, lexical_tokens


def exact(token: str) -> LexicalQuery:
    return LexicalQuery(f'"{token.lower()}"', (token.lower(),), (), True)

ERR_REPO = {
    "src/pool.py": '''
        class PoolExhausted(RuntimeError):
            """Raised when every slot of the connection pool is busy."""


        def borrow(pool):
            if not pool.free:
                raise PoolExhausted("E_POOL_4012: no free connection in pool")
            return pool.free.pop()
    ''',
    "src/report.py": '''
        def monthly_report(rows):
            """Aggregate invoice totals per month."""
            return sum(r.total for r in rows)
    ''',
}


def test_tokens_keep_whole_identifiers_and_subtokens():
    toks = lexical_tokens("raise PoolExhausted('E_POOL_4012') # parse_json_config 0x80070005")
    for t in ["poolexhausted", "pool", "exhausted", "e_pool_4012", "4012", "parse_json_config", "parse", "json", "config", "0x80070005"]:
        assert t in toks


def test_query_builder_flags_code_like_queries():
    assert build_query("why does `ECONNRESET` happen").code_like
    assert build_query("fix PoolExhausted in borrow").code_like
    assert not build_query("aggregate monthly invoice totals").code_like
    q = build_query('error "no free connection" in the pool')
    assert q.phrases == ("no free connection",) and "the" not in q.terms


@pytest.fixture
def err_mgr(tmp_path, make_manager):
    from conftest import write_repo

    write_repo(tmp_path / "repo", ERR_REPO)
    mgr = make_manager()
    mgr.ingest_repository()
    return mgr


def _hit_symbols(mgr, query, k=5):
    recs = mgr.records(h.node_id for h in mgr.lexical_search(query, k))
    return [recs[h.node_id].symbol or recs[h.node_id].path for h in mgr.lexical_search(query, k)]


def test_exact_identifier_and_error_code_hits(err_mgr):
    assert _hit_symbols(err_mgr, "E_POOL_4012")[0] == "borrow"
    assert _hit_symbols(err_mgr, "PoolExhausted")[0] == "PoolExhausted"
    assert _hit_symbols(err_mgr, '"no free connection"')[0] == "borrow"
    assert _hit_symbols(err_mgr, "monthly invoice")[0] == "monthly_report"
    files = err_mgr.lexical_search("report.py", 3, frozenset({NodeKind.FILE}))
    assert err_mgr.record(files[0].node_id).path == "src/report.py"


def test_index_follows_edits_and_deletions(err_mgr, tmp_path):
    path = tmp_path / "repo/src/pool.py"
    path.write_text(path.read_text().replace("E_POOL_4012", "E_POOL_5000"))
    err_mgr.ingest_files(["src/pool.py"])
    assert err_mgr.lexical_search(exact("E_POOL_4012"), 5) == []
    assert _hit_symbols(err_mgr, "E_POOL_5000")[0] == "borrow"
    (tmp_path / "repo/src/report.py").unlink()
    err_mgr.ingest_repository()
    assert err_mgr.lexical_search(exact("monthly_report"), 5) == []


def test_rollback_keeps_lexicon_consistent(err_mgr, tmp_path, monkeypatch):
    before = err_mgr.lexicon.count()
    path = tmp_path / "repo/src/report.py"
    path.write_text(path.read_text() + "\n\ndef quarterly_report(rows):\n    return 0\n")

    def boom(*a, **k):
        raise RuntimeError("crash during ingest")

    monkeypatch.setattr(err_mgr, "_recompute_structural", boom)
    with pytest.raises(RuntimeError):
        err_mgr.ingest_files(["src/report.py"])
    assert err_mgr.lexicon.count() == before
    assert err_mgr.lexical_search(exact("quarterly_report"), 5) == []
    monkeypatch.undo()
    err_mgr.ingest_files(["src/report.py"])
    assert _hit_symbols(err_mgr, "quarterly_report")[0] == "quarterly_report"


def test_missing_lexicon_is_rebuilt_on_open(err_mgr, make_manager):
    n = err_mgr.lexicon.count()
    with err_mgr.meta.transaction():
        err_mgr.meta.connection.execute("DROP TABLE lexicon")
        err_mgr.meta.set_meta("lexicon", "0")
    err_mgr.close()
    again = make_manager()
    assert again.lexicon.count() == n
    assert _hit_symbols(again, "E_POOL_4012")[0] == "borrow"


def test_contentless_index_does_not_duplicate_code(err_mgr, tmp_path):
    if sqlite3.sqlite_version_info < (3, 43, 0):
        pytest.skip("contentless_delete needs SQLite >= 3.43")
    assert err_mgr.lexicon.contentless
    err_mgr.flush()
    blob = b"".join(p.read_bytes() for p in (tmp_path / "store").glob("meta.sqlite3*"))
    assert b"Raised when every slot of the connection pool is busy" not in blob
