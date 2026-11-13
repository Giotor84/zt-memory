"""Semantic chunking: symbol extraction per language and line-shift-invariant identities."""
from __future__ import annotations

import textwrap

import pytest

from zt_memory.chunking import chunk_source
from zt_memory.syntax import available_languages

TS_LANGS = set(available_languages())


def needs(lang: str):
    return pytest.mark.skipif(lang not in TS_LANGS, reason=f"tree-sitter grammar for {lang} not installed")


def ident(chunks) -> dict[str, str]:
    return {c.symbol: c.content_hash for c in chunks}


def chunks(path: str, src: str, **kw):
    return chunk_source(path, textwrap.dedent(src).lstrip("\n"), **kw)


SAMPLES: dict[str, tuple[str, str, set[str]]] = {
    "python": ("m.py", '''
        import os
        LIMIT = 3


        # Parses things.
        @cache
        def parse(path: str,
                  strict: bool = False) -> dict:
            """Doc."""
            return {}


        class Store(Base):
            """A store."""
            size = 4

            def get(self, key):
                return key

            @property
            def name(self):
                return "x"

            @name.setter
            def name(self, value):
                pass

            class Meta:
                def opts(self):
                    return 1
        ''', {"<header>", "parse", "Store", "Store.get", "Store.name", "Store.Meta", "Store.Meta.opts"}),
    "javascript": ("m.js", '''
        import x from "y";

        /** Adds. */
        export function add(a, b) {
          return a + b;
        }

        export default class Cart extends Base {
          constructor() {
            super();
          }

          total() {
            return 1;
          }

          onClick = () => {
            this.total();
          };
        }

        const handler = async (req) => {
          return req;
        };
        ''', {"<header>", "add", "Cart", "Cart.constructor", "Cart.total", "Cart.onClick", "handler"}),
    "typescript": ("m.ts", '''
        import { Pool } from "pg";

        export interface Options {
          retries: number;
        }

        export type Id = string;

        export class Repo {
          private pool: Pool;

          constructor(pool: Pool) {
            this.pool = pool;
          }

          async find(id: Id): Promise<void> {
            return;
          }
        }

        export enum Color {
          Red,
        }
        ''', {"<header>", "Options", "Id", "Repo", "Repo.constructor", "Repo.find", "Color"}),
    "go": ("m.go", '''
        package server

        import "net/http"

        // Server handles requests.
        type Server struct {
            mux *http.ServeMux
        }

        func (s *Server) Start(addr string) error {
            return http.ListenAndServe(addr, s.mux)
        }

        func New() *Server {
            return &Server{}
        }
        ''', {"<header>", "Server", "Server.Start", "New"}),
    "rust": ("m.rs", '''
        use std::fmt;

        #[derive(Debug)]
        pub struct Point {
            x: i32,
        }

        impl Point {
            /// Constructor.
            pub fn new() -> Self {
                Point { x: 0 }
            }
        }

        impl fmt::Display for Point {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                write!(f, "{}", self.x)
            }
        }
        ''', {"<header>", "Point", "impl Point", "impl Point.new", "impl fmt::Display for Point", "impl fmt::Display for Point.fmt"}),
    "java": ("M.java", '''
        package app;

        import java.util.List;

        public class Service {
            private int count;

            public Service() {
                count = 0;
            }

            /** By id. */
            public void load(int id) {
                count++;
            }

            public void load(String name) {
                count--;
            }
        }
        ''', {"<header>", "Service", "Service.Service", "Service.load"}),
    "c": ("m.c", '''
        #include <stdio.h>

        struct point {
            int x;
        };

        /* Adds two ints. */
        static int add(int a, int b) {
            return a + b;
        }

        char *name(void) {
            return "x";
        }
        ''', {"<header>", "point", "add", "name"}),
    "cpp": ("m.cpp", '''
        #include <vector>

        namespace geo {

        class Shape {
         public:
          virtual double area() const {
            return 0.0;
          }
        };

        }  // namespace geo

        template <typename T>
        T clamp(T v, T lo, T hi) {
            return v < lo ? lo : (v > hi ? hi : v);
        }

        double geo::Shape::perimeter() const {
            return 1.0;
        }
        ''', {"<header>", "geo", "geo.Shape", "geo.Shape.area", "clamp", "geo::Shape::perimeter"}),
}


@pytest.mark.parametrize("lang", sorted(SAMPLES))
def test_symbols_per_language(lang):
    if lang not in TS_LANGS and lang != "python":
        pytest.skip("grammar missing")
    path, src, expected = SAMPLES[lang]
    syms = {c.symbol for c in chunks(path, src)}
    assert expected <= syms, f"missing: {expected - syms}"


def _mutations(path: str, src: str) -> list[str]:
    text = textwrap.dedent(src).lstrip("\n")
    lines = text.splitlines()
    comment = "#" if path.endswith(".py") else "//"
    # (a) blank lines + an unrelated comment inserted after the first line
    a = "\n".join(lines[:1] + ["", "", f"{comment} unrelated note", ""] + lines[1:])
    # (b) every non-empty top-level line shifted down by 5 blank lines at the end of the header
    first_blank = next(i for i, ln in enumerate(lines) if not ln.strip())
    b = "\n".join(lines[:first_blank] + [""] * 5 + lines[first_blank:])
    return [a, b]


@pytest.mark.parametrize("lang", sorted(SAMPLES))
def test_identity_invariant_to_line_shifts(lang):
    if lang not in TS_LANGS and lang != "python":
        pytest.skip("grammar missing")
    path, src, _ = SAMPLES[lang]
    base = ident(chunks(path, src))
    for mutated in _mutations(path, src):
        after = chunk_source(path, mutated)
        got = ident(after)
        same = {s: h for s, h in base.items() if s != "<header>"}
        assert {s: got.get(s) for s in same} == same
        moved = {c.symbol: c.start_line for c in after}
        assert any(moved[s] != c.start_line for c in chunks(path, src) for s in [c.symbol] if s in moved)


def test_inserting_and_reordering_symbols_keeps_identities():
    src = "import os\n\n\ndef a():\n    return 1\n\n\ndef b():\n    return 2\n\n\nclass C:\n    def m(self):\n        return 3\n"
    base = ident(chunk_source("x.py", src))
    inserted = src.replace("def a():", "def new_fn():\n    return 0\n\n\ndef a():")
    got = ident(chunk_source("x.py", inserted))
    assert {k: got[k] for k in base} == base and "new_fn" in got
    reordered = "import os\n\n\nclass C:\n    def m(self):\n        return 3\n\n\ndef b():\n    return 2\n\n\ndef a():\n    return 1\n"
    assert ident(chunk_source("x.py", reordered)) == base


def test_classes_always_split_regardless_of_size():
    small = "class K:\n    x = 1\n\n    def m(self):\n        return 1\n"
    assert [c.symbol for c in chunk_source("k.py", small)] == ["K", "K.m"]
    big = small + "".join(f"\n    def f{i}(self):\n        return {i}\n" for i in range(80))
    syms = [c.symbol for c in chunk_source("k.py", big)]
    assert syms[:2] == ["K", "K.m"] and "K.f79" in syms
    # identity of K.m does not depend on how large the class is
    assert ident(chunk_source("k.py", small))["K.m"] == ident(chunk_source("k.py", big))["K.m"]


def test_leading_comments_attach_only_when_adjacent():
    src = "# detached\n\n# attached 1\n# attached 2\n@deco\ndef f():\n    pass\n"
    [header, f] = chunk_source("c.py", src)
    assert header.symbol == "<header>" and header.text == "# detached"
    assert f.symbol == "f" and f.start_line == 3 and f.text.startswith("# attached 1")


@needs("java")
def test_overloads_are_disambiguated_and_stable():
    two = "class S {\n  void load(int id) {}\n  void load(String n) {}\n}\n"
    three = "class S {\n  void load(int id) {}\n  void load(String n) {}\n  void load(long x) {}\n}\n"
    a, b = ident(chunk_source("S.java", two)), ident(chunk_source("S.java", three))
    loads_a = sorted(s for s in a if s.startswith("S.load"))
    assert len(loads_a) == 2 and "S.load" in loads_a
    assert {s: b[s] for s in loads_a} == {s: a[s] for s in loads_a}


@needs("python")
def test_treesitter_tolerates_syntax_errors():
    good = "def a():\n    return 1\n\n\ndef c():\n    return 3\n"
    broken = "def a():\n    return 1\n\n\ndef b(:\n    return (\n\n\ndef c():\n    return 3\n"
    g, b = ident(chunk_source("e.py", good)), ident(chunk_source("e.py", broken, backend="treesitter"))
    assert b.get("a") == g["a"] and b.get("c") == g["c"]
    assert chunk_source("e.py", broken, backend="ast")[0].symbol.startswith("<block:")  # ast cannot parse it


@needs("python")
def test_ast_backend_matches_treesitter_backend():
    from conftest import SAMPLE_REPO

    sources = [textwrap.dedent(v).lstrip("\n") for k, v in SAMPLE_REPO.items() if k.endswith(".py")]
    sources.append(textwrap.dedent(SAMPLES["python"][1]).lstrip("\n"))
    for src in sources:
        ts = [(c.symbol, c.start_line, c.end_line, c.content_hash) for c in chunk_source("x.py", src, backend="treesitter")]
        st = [(c.symbol, c.start_line, c.end_line, c.content_hash) for c in chunk_source("x.py", src, backend="ast")]
        assert ts == st


def test_oversized_symbol_split_into_content_anchored_blocks():
    body = "\n\n".join(f"    step_{i} = compute({i})\n    check(step_{i})" for i in range(60))
    src = f"def huge():\n{body}\n"
    base = chunk_source("h.py", src, max_lines=40, min_lines=8)
    assert base[0].symbol == "huge" and all(c.symbol.startswith("huge#<blk:") for c in base[1:])
    edited = src.replace("check(step_59)", "check(step_59, strict=True)")
    after = chunk_source("h.py", edited, max_lines=40, min_lines=8)
    changed = {s for s, h in ident(after).items() if ident(base).get(s) != h}
    assert len(changed) == 1  # only the block containing the edit


def test_block_fallback_is_anchored_for_unknown_languages():
    sections = "\n\n".join(f"section_{i}:\n  key: {i}\n  other: value_{i}\n  more: {i * 2}\n  last: true" for i in range(8))
    base = ident(chunk_source("c.yaml", sections, min_lines=4))
    assert len(base) > 2 and all(s.startswith("<block:") for s in base)
    prepended = "new_section:\n  key: 0\n  a: 1\n  b: 2\n  c: 3\n\n" + sections
    got = ident(chunk_source("c.yaml", prepended, min_lines=4))
    assert len(set(base.items()) & set(got.items())) >= len(base) - 1
