import pytest

from zt_memory.cli import main
from zt_memory.project import ProjectConfig
from zt_memory.router import GeometricRouter
from zt_memory.stacktrace import Location, PathResolver, parse_locations

PY_TRACEBACK = '''Traceback (most recent call last):
  File "/home/dev/shop/app.py", line 3, in <module>
    checkout()
  File "/home/dev/shop/shop/cart.py", line 9, in checkout
    return apply_discount(total, pct)
  File "C:\\Users\\dev\\shop\\shop\\pricing.py", line 14, in apply_discount
    raise ValueError("bad percent")
ValueError: bad percent
'''


def test_parse_python_traceback_innermost_first():
    locs = parse_locations(PY_TRACEBACK)
    assert [(l.path, l.line, l.func) for l in locs] == [
        ("C:/Users/dev/shop/shop/pricing.py", 14, "apply_discount"),
        ("/home/dev/shop/shop/cart.py", 9, "checkout"),
        ("/home/dev/shop/app.py", 3, "<module>"),
    ]


@pytest.mark.parametrize("text, expected", [
    ("tests/test_options.py:2295: AssertionError", Location("tests/test_options.py", 2295)),
    ("src/click/core.py:1234: in process_value", Location("src/click/core.py", 1234, "process_value")),
    ("pkg/mod.py:7:5: error: Incompatible types", Location("pkg/mod.py", 7)),
    ("    at parse (/app/src/util/parse.ts:41:13)", Location("/app/src/util/parse.ts", 41)),
    ("  --> src/main.rs:10:5", Location("src/main.rs", 10)),
    ("\t/go/src/cobra/command.go:1043 +0x1d", Location("/go/src/cobra/command.go", 1043)),
    ("\tat com.acme.Billing.total(Billing.java:88)", Location("Billing.java", 88)),
])
def test_parse_other_toolchains(text, expected):
    assert parse_locations(text) == [expected]


def test_parse_ignores_text_without_locations_and_dedups():
    assert parse_locations("the discount is wrong for 100% and version 2.0.1") == []
    assert len(parse_locations("a.py:3: x\na.py:3: y\na.py:4: z")) == 2


def test_resolver_suffixes_installed_copies_and_ties():
    r = PathResolver(["src/click/core.py", "src/click/types.py", "tests/test_core.py", "docs/conf.py", "a/util.py", "b/util.py"])
    assert r.resolve("src/click/core.py") == "src/click/core.py"
    assert r.resolve("/home/me/click/src/click/core.py") == "src/click/core.py"
    assert r.resolve("/usr/lib/python3.12/site-packages/click/types.py") == "src/click/types.py"
    assert r.resolve("C:/x/tests/test_core.py") == "tests/test_core.py"
    assert r.resolve("conf.py") == "docs/conf.py"  # unique bare name
    assert r.resolve("util.py") is None and r.resolve("/x/util.py") is None  # ambiguous: refused
    assert r.resolve("/usr/lib/python3.12/json/decoder.py") is None


LONG_FN = "def long_function(x):\n" + "".join(f"    x = x + {i}\n" for i in range(150)) + "    return x\n"


@pytest.fixture
def repo(tmp_path, capsys):
    root = tmp_path / "shop"
    (root / "shop").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "shop/__init__.py").write_text("")
    (root / "shop/pricing.py").write_text(
        "VAT = 0.22\n\n\ndef with_vat(price):\n    return price * (1 + VAT)\n\n\n"
        "def apply_discount(price, percent):\n    if percent > 1:\n        raise ValueError('bad percent')\n"
        "    return price - price * percent\n")
    (root / "shop/cart.py").write_text("from shop.pricing import apply_discount\n\n\ndef checkout(total, pct):\n    return apply_discount(total, pct)\n")
    (root / "shop/big.py").write_text(LONG_FN)
    (root / "tests/test_cart.py").write_text("from shop.cart import checkout\n\n\ndef test_checkout():\n    assert checkout(10, 50) == 5\n")
    assert main(["init", "--repo", str(root), "--rank", "32"]) == 0
    cfg = ProjectConfig.load(root)
    cfg.memory.update(calibration_samples=128)
    cfg.save(overwrite=True)
    assert main(["sync", "--repo", str(root), "--quiet"]) == 0
    capsys.readouterr()
    return root


def _route(root, query, **router):
    pc = ProjectConfig.load(root)
    with pc.open_manager() as mgr:
        rc = pc.router_config()
        for k, v in router.items():
            setattr(rc, k, v)
        return GeometricRouter(mgr, rc).route_to_compute_engine(query, token_budget=4000)


def test_traceback_symbols_are_pinned_first(repo):
    log = (f'Traceback (most recent call last):\n  File "{repo}/tests/test_cart.py", line 5, in test_checkout\n'
           f'  File "{repo}/shop/cart.py", line 5, in checkout\n'
           f'  File "/opt/venv/site-packages/shop/pricing.py", line 10, in apply_discount\nValueError: bad percent')
    ctx = _route(repo, log)
    assert ctx.pinned == ("shop/pricing.py:10", "shop/cart.py:5", "tests/test_cart.py:5")  # source first, then tests
    assert ctx.blocks[0].path == "shop/pricing.py"  # the innermost frame's file comes first
    assert any("[apply_discount]" in b.title for b in ctx.blocks)
    assert any("[checkout]" in b.title for b in ctx.blocks)
    off = _route(repo, log, trace_anchors=0)
    assert off.pinned == ()


def test_long_symbol_is_cut_to_a_window_around_the_line(repo):
    ctx = _route(repo, "shop/big.py:120: in long_function\nE   OverflowError")
    block = next(b for b in ctx.blocks if b.path == "shop/big.py")
    assert block.start_line <= 120 <= block.end_line and block.end_line - block.start_line < 80
    assert ctx.pinned == ("shop/big.py:120",)


def test_unknown_locations_change_nothing(repo):
    q = "apply_discount raises for percent above one"
    assert _route(repo, q + "\n  File \"/usr/lib/python3.12/json/decoder.py\", line 3").pinned == ()
