"""`zt` must not crash on a console that cannot encode every character (Windows cp1252)."""
import io
import sys

from zt_memory import cli


def test_non_utf8_console_replaces_unencodable_characters(monkeypatch):
    out = io.TextIOWrapper(io.BytesIO(), encoding="cp1252", newline="\n")
    monkeypatch.setattr(sys, "stdout", out)
    cli._tolerant_console()
    print("arrow → and 中")
    out.flush()
    assert out.buffer.getvalue().decode("cp1252") == "arrow ? and ?\n"


def test_utf8_console_is_left_alone(monkeypatch):
    out = io.TextIOWrapper(io.BytesIO(), encoding="utf-8")
    monkeypatch.setattr(sys, "stdout", out)
    cli._tolerant_console()
    assert out.errors == "strict"
