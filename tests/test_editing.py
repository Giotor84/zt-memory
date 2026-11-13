import pytest

from zt_memory.editing import EditError, parse_edit_blocks, plan_edits, undo_last, write_plan

SRC = "def apply_discount(price, percent):\n    return price - price * percent\n\n\ndef total(items):\n    return sum(items)\n"


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "shop").mkdir()
    (tmp_path / "shop/pricing.py").write_text(SRC, newline="\n")  # LF on every OS: CRLF has its own test
    return tmp_path


def _answer(path="shop/pricing.py", search="    return price - price * percent", replace="    return price - price * percent / 100"):
    return f"Fix the percentage.\n\n{path}\n<<<<<<< SEARCH\n{search}\n=======\n{replace}\n>>>>>>> REPLACE\n"


def test_parse_variants():
    fenced = "```python\nshop/pricing.py\n<<<<<<< SEARCH\na\n=======\nb\n>>>>>>> REPLACE\n```"
    path_inside_fence = "`shop/pricing.py`:\n```python\n<<<<<<< SEARCH\na\n=======\nb\n>>>>>>> REPLACE\n```\n<<<<<<< SEARCH\nc\n=======\nd\n>>>>>>> REPLACE"
    for text in (fenced, path_inside_fence):
        blocks = parse_edit_blocks(text)
        assert blocks[0].path == "shop/pricing.py" and (blocks[0].search, blocks[0].replace) == ("a", "b")
    assert parse_edit_blocks(path_inside_fence)[1].path == "shop/pricing.py"  # path carried over
    with pytest.raises(EditError):
        parse_edit_blocks("<<<<<<< SEARCH\na\n=======\nb\n")
    assert parse_edit_blocks("no blocks here") == []


def test_exact_apply_backup_and_undo(repo):
    plan = plan_edits(repo, parse_edit_blocks(_answer()))
    assert plan.ok and plan.match_modes == ["exact"] and "+    return price - price * percent / 100" in plan.diff()
    applied = write_plan(repo, plan, label="fix")
    assert applied.changed == ("shop/pricing.py",)
    assert "percent / 100" in (repo / "shop/pricing.py").read_text()
    undone = undo_last(repo)
    assert undone and undone[1] == ["shop/pricing.py"] and (repo / "shop/pricing.py").read_text() == SRC
    assert undo_last(repo) is None


def test_tolerant_matching(repo):
    trailing = _answer(search="    return price - price * percent   ")
    assert plan_edits(repo, parse_edit_blocks(trailing)).match_modes == ["rstrip"]
    dedented = _answer(search="return price - price * percent", replace="return price - price * percent / 100")
    plan = plan_edits(repo, parse_edit_blocks(dedented))
    assert plan.match_modes == ["indent"] and "    return price - price * percent / 100\n" in plan.new_contents["shop/pricing.py"]
    numbered = _answer(search="2 |     return price - price * percent", replace="2 |     return price * (1 - percent / 100)")
    assert "price * (1 - percent / 100)" in plan_edits(repo, parse_edit_blocks(numbered)).new_contents["shop/pricing.py"]


def test_ambiguous_missing_and_all_or_nothing(repo):
    (repo / "shop/pricing.py").write_text(SRC + "\n\ndef again(items):\n    return sum(items)\n", newline="\n")
    ambiguous = _answer(search="    return sum(items)", replace="    return 0")
    assert "matches 2 places" in plan_edits(repo, parse_edit_blocks(ambiguous)).failures[0].reason
    good_then_bad = _answer() + "\n" + _answer(search="    this line does not exist", replace="x")
    plan = plan_edits(repo, parse_edit_blocks(good_then_bad))
    assert not plan.ok and len(plan.failures) == 1
    with pytest.raises(EditError):
        write_plan(repo, plan)
    assert "percent / 100" not in (repo / "shop/pricing.py").read_text()  # nothing written


def test_create_files_and_reject_unsafe_paths(repo):
    plan = plan_edits(repo, parse_edit_blocks("shop/tax.py\n<<<<<<< SEARCH\n=======\nRATE = 0.22\n>>>>>>> REPLACE"))
    applied = write_plan(repo, plan)
    assert applied.created == ("shop/tax.py",) and (repo / "shop/tax.py").read_text() == "RATE = 0.22\n"
    undo_last(repo)
    assert not (repo / "shop/tax.py").exists()
    for bad in ("../evil.py", "C:/Windows/x.py", ".git/config", ".zt/config.toml"):
        assert not plan_edits(repo, parse_edit_blocks(_answer(path=bad))).ok
    assert "empty SEARCH" in plan_edits(repo, parse_edit_blocks(_answer(search=""))).failures[0].reason


def test_crlf_files_keep_windows_newlines(repo):
    (repo / "shop/pricing.py").write_bytes(SRC.replace("\n", "\r\n").encode())
    plan = plan_edits(repo, parse_edit_blocks(_answer()))
    write_plan(repo, plan)
    data = (repo / "shop/pricing.py").read_bytes()
    assert b"percent / 100\r\n" in data and b"\n" not in data.replace(b"\r\n", b"")


def test_fenced_sections_from_chat_models(repo):
    # verbatim shape of a Qwen2.5-Coder-14B answer: each section wrapped in its own ```python fence
    answer = ("shop/pricing.py\n<<<<<<< SEARCH\n```python\ndef apply_discount(price, percent):\n"
              "    return price - price * percent\n```\n=======\n```python\ndef apply_discount(price, percent):\n"
              "    return price * (1 - percent / 100)\n```\n>>>>>>> REPLACE\n")
    plan = plan_edits(repo, parse_edit_blocks(answer))
    assert plan.ok and plan.match_modes == ["unfenced+exact"]
    new = plan.new_contents["shop/pricing.py"]
    assert "```" not in new and "    return price * (1 - percent / 100)\n" in new and new.endswith("return sum(items)\n")
    created = plan_edits(repo, parse_edit_blocks("shop/tax.py\n<<<<<<< SEARCH\n=======\n```python\nRATE = 0.22\n```\n>>>>>>> REPLACE\n"))
    assert created.new_contents["shop/tax.py"] == "RATE = 0.22\n"
    doc = plan_edits(repo, parse_edit_blocks("docs/usage.md\n<<<<<<< SEARCH\n=======\n```bash\nzt sync\n```\n>>>>>>> REPLACE\n"))
    assert doc.new_contents["docs/usage.md"] == "```bash\nzt sync\n```\n"  # fences are real content in prose files
