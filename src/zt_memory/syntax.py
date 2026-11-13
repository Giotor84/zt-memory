"""Semantic symbol extraction: tree-sitter grammars (multi-language) with a stdlib `ast` fallback for Python.

Extractors return a language-neutral tree of SymbolNode (local name, line span, signature line,
members). Qualification, comment attachment, disambiguation and gap handling are done once,
backend-independently, in `chunking.py`, so both Python backends yield identical chunks.

Grammars are optional wheels (`pip install zt-memory[treesitter]`); missing ones degrade to
content-anchored blocks without failing.
"""
from __future__ import annotations

import ast
import importlib
import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any, Callable, Iterable, Iterator

try:
    from tree_sitter import Language, Node, Parser
except ImportError:  # pragma: no cover - optional dependency
    Language = Node = Parser = None  # type: ignore[assignment,misc]


@dataclass
class SymbolNode:
    name: str  # local (unqualified) name
    start: int  # 1-based first line, decorators/wrappers included
    end: int  # 1-based last line
    sig_line: int  # line of the defining keyword (excludes decorators)
    members: list["SymbolNode"] = field(default_factory=list)
    container: bool = False  # class-like: split into header + members


@dataclass(frozen=True)
class LanguageSpec:
    name: str
    extensions: tuple[str, ...]
    module: str
    loader: str
    comment_prefixes: tuple[str, ...]
    symbol_types: frozenset[str]
    container_types: frozenset[str] = frozenset()
    wrapper_types: frozenset[str] = frozenset()
    transparent_types: frozenset[str] = frozenset()
    name_of: Callable[[Any], str | None] = lambda n: _field_text(n, "name")


# ------------------------------------------------------------------------------------------
# name resolution helpers
# ------------------------------------------------------------------------------------------
def _text(node: Any) -> str:
    return " ".join(node.text.decode("utf-8", "replace").split())


def _field_text(node: Any, name: str) -> str | None:
    child = node.child_by_field_name(name)
    return _text(child) if child is not None else None


_JS_FN_VALUES = frozenset({"arrow_function", "function_expression", "function", "generator_function", "class"})


def _js_name(node: Any) -> str | None:
    if node.type in ("lexical_declaration", "variable_declaration"):
        for decl in node.named_children:
            if decl.type == "variable_declarator":
                value = decl.child_by_field_name("value")
                if value is not None and value.type in _JS_FN_VALUES:
                    return _field_text(decl, "name")
        return None
    if node.type in ("field_definition", "public_field_definition"):
        value = node.child_by_field_name("value")
        if value is None or value.type not in _JS_FN_VALUES:
            return None
        return _field_text(node, "property") or _field_text(node, "name")
    return _field_text(node, "name")


_GO_RECV_STRIP = re.compile(r"\[.*?\]|[*\s]")


def _go_name(node: Any) -> str | None:
    if node.type == "method_declaration":
        recv_type = ""
        recv = node.child_by_field_name("receiver")
        if recv is not None:
            for param in recv.named_children:
                t = param.child_by_field_name("type")
                if t is not None:
                    recv_type = _GO_RECV_STRIP.sub("", t.text.decode("utf-8", "replace"))
                    break
        name = _field_text(node, "name")
        return f"{recv_type}.{name}" if recv_type and name else name
    if node.type == "type_declaration":
        names = [_field_text(s, "name") for s in node.named_children if s.type in ("type_spec", "type_alias")]
        return "+".join(n for n in names if n) or None
    return _field_text(node, "name")


def _rust_name(node: Any) -> str | None:
    if node.type == "impl_item":
        ty, trait = _field_text(node, "type"), _field_text(node, "trait")
        if ty is None:
            return None
        return f"impl {trait} for {ty}" if trait else f"impl {ty}"
    return _field_text(node, "name")


_C_LEAF = frozenset({"identifier", "field_identifier", "qualified_identifier", "operator_name", "destructor_name", "type_identifier"})


def _c_declarator_name(node: Any) -> str | None:
    d = node.child_by_field_name("declarator")
    for _ in range(16):
        if d is None or d.type in _C_LEAF:
            break
        nxt = d.child_by_field_name("declarator")
        if nxt is None:
            nxt = next((c for c in d.named_children if c.type.endswith(("declarator", "identifier"))), None)
        d = nxt
    return _text(d) if d is not None and d.type in _C_LEAF else None


def _c_name(node: Any) -> str | None:
    if node.type in ("function_definition", "type_definition"):
        return _c_declarator_name(node)
    if node.type in ("struct_specifier", "union_specifier", "enum_specifier", "class_specifier"):
        return _field_text(node, "name") if node.child_by_field_name("body") is not None else None
    if node.type == "namespace_definition":
        return _field_text(node, "name") or "<anonymous>"
    return _field_text(node, "name")


_C_PREPROC = frozenset({"preproc_if", "preproc_ifdef", "preproc_else", "preproc_elif", "preproc_elifdef"})
_JS_SYMBOLS = frozenset({
    "function_declaration", "generator_function_declaration", "class_declaration", "method_definition",
    "lexical_declaration", "variable_declaration", "field_definition",
})
_TS_SYMBOLS = _JS_SYMBOLS | frozenset({
    "abstract_class_declaration", "interface_declaration", "type_alias_declaration", "enum_declaration",
    "public_field_definition", "internal_module", "function_signature",
})
_C_SYMBOLS = frozenset({"function_definition", "struct_specifier", "union_specifier", "enum_specifier", "type_definition"})
_BRACE_COMMENTS = ("//", "/*", "*", "*/")

LANGUAGES: dict[str, LanguageSpec] = {}


def register_language(spec: LanguageSpec) -> None:
    """Add or override a grammar (any pip-installable `tree_sitter_<lang>` module)."""
    LANGUAGES[spec.name] = spec
    _PARSERS.pop(spec.name, None)


for _spec in (
    LanguageSpec("python", (".py", ".pyi"), "tree_sitter_python", "language", ("#",),
                 frozenset({"function_definition", "class_definition"}),
                 container_types=frozenset({"class_definition"}),
                 wrapper_types=frozenset({"decorated_definition"})),
    LanguageSpec("javascript", (".js", ".jsx", ".mjs", ".cjs"), "tree_sitter_javascript", "language", _BRACE_COMMENTS,
                 _JS_SYMBOLS, container_types=frozenset({"class_declaration"}),
                 wrapper_types=frozenset({"export_statement"}), name_of=_js_name),
    LanguageSpec("typescript", (".ts", ".mts", ".cts"), "tree_sitter_typescript", "language_typescript",
                 _BRACE_COMMENTS + ("@",), _TS_SYMBOLS,
                 container_types=frozenset({"class_declaration", "abstract_class_declaration", "internal_module"}),
                 wrapper_types=frozenset({"export_statement", "expression_statement"}), name_of=_js_name),
    LanguageSpec("tsx", (".tsx",), "tree_sitter_typescript", "language_tsx", _BRACE_COMMENTS + ("@",), _TS_SYMBOLS,
                 container_types=frozenset({"class_declaration", "abstract_class_declaration", "internal_module"}),
                 wrapper_types=frozenset({"export_statement", "expression_statement"}), name_of=_js_name),
    LanguageSpec("go", (".go",), "tree_sitter_go", "language", ("//", "/*", "*"),
                 frozenset({"function_declaration", "method_declaration", "type_declaration"}), name_of=_go_name),
    LanguageSpec("rust", (".rs",), "tree_sitter_rust", "language", ("//", "/*", "*", "#["),
                 frozenset({"function_item", "function_signature_item", "struct_item", "enum_item", "union_item",
                            "trait_item", "impl_item", "mod_item", "type_item", "macro_definition"}),
                 container_types=frozenset({"impl_item", "trait_item", "mod_item"}), name_of=_rust_name),
    LanguageSpec("java", (".java",), "tree_sitter_java", "language", _BRACE_COMMENTS + ("@",),
                 frozenset({"class_declaration", "interface_declaration", "enum_declaration", "record_declaration",
                            "annotation_type_declaration", "method_declaration", "constructor_declaration",
                            "compact_constructor_declaration"}),
                 container_types=frozenset({"class_declaration", "interface_declaration", "enum_declaration",
                                            "record_declaration", "annotation_type_declaration"}),
                 transparent_types=frozenset({"enum_body_declarations"})),
    LanguageSpec("c", (".c",), "tree_sitter_c", "language", _BRACE_COMMENTS, _C_SYMBOLS,
                 transparent_types=_C_PREPROC, name_of=_c_name),
    LanguageSpec("cpp", (".cc", ".cpp", ".cxx", ".c++", ".h", ".hh", ".hpp", ".hxx", ".ipp"), "tree_sitter_cpp", "language",
                 _BRACE_COMMENTS, _C_SYMBOLS | frozenset({"class_specifier", "namespace_definition"}),
                 container_types=frozenset({"class_specifier", "struct_specifier", "namespace_definition"}),
                 wrapper_types=frozenset({"template_declaration"}),
                 transparent_types=_C_PREPROC | frozenset({"linkage_specification"}), name_of=_c_name),
):
    LANGUAGES[_spec.name] = _spec

_PARSERS: dict[str, Any] = {}
_UNAVAILABLE: set[str] = set()


def language_for_path(rel_path: str) -> LanguageSpec | None:
    suffix = PurePosixPath(rel_path).suffix.lower()
    for spec in LANGUAGES.values():
        if suffix in spec.extensions:
            return spec
    return None


def _parser(spec: LanguageSpec) -> Any | None:
    if spec.name in _PARSERS:
        return _PARSERS[spec.name]
    if Parser is None or spec.name in _UNAVAILABLE:
        return None
    try:
        module = importlib.import_module(spec.module)
        parser = Parser(Language(getattr(module, spec.loader)()))
    except (ImportError, AttributeError, ValueError, TypeError):
        _UNAVAILABLE.add(spec.name)
        return None
    _PARSERS[spec.name] = parser
    return parser


def available_languages() -> list[str]:
    return sorted(name for name, spec in LANGUAGES.items() if _parser(spec) is not None)


# ------------------------------------------------------------------------------------------
# tree-sitter extraction
# ------------------------------------------------------------------------------------------
def _end_line(node: Any) -> int:
    row, col = node.end_point
    if col == 0 and row > node.start_point[0]:
        row -= 1
    return row + 1


def _iter_members(spec: LanguageSpec, nodes: Iterable[Any]) -> Iterator[Any]:
    for n in nodes:
        if n.type in spec.transparent_types:
            body = n.child_by_field_name("body")
            yield from _iter_members(spec, (body or n).named_children)
        else:
            yield n


def _unwrap(spec: LanguageSpec, node: Any) -> Any | None:
    for _ in range(4):
        if node.type not in spec.wrapper_types:
            return node
        inner = node.child_by_field_name("definition") or node.child_by_field_name("declaration")
        if inner is None:
            inner = next((c for c in node.named_children if c.type in spec.symbol_types or c.type in spec.wrapper_types), None)
        if inner is None:
            return None
        node = inner
    return node if node.type not in spec.wrapper_types else None


def _ts_extract(spec: LanguageSpec, nodes: Iterable[Any]) -> list[SymbolNode]:
    out: list[SymbolNode] = []
    for outer in _iter_members(spec, nodes):
        node = _unwrap(spec, outer)
        if node is None or node.type not in spec.symbol_types:
            continue
        name = spec.name_of(node)
        if not name:
            continue
        container = node.type in spec.container_types
        members: list[SymbolNode] = []
        if container:
            body = node.child_by_field_name("body")
            if body is not None:
                members = _ts_extract(spec, body.named_children)
        out.append(SymbolNode(name, outer.start_point[0] + 1, _end_line(outer), node.start_point[0] + 1, members, container))
    return out


def treesitter_symbols(spec: LanguageSpec, text: str) -> list[SymbolNode] | None:
    """None if the grammar is unavailable. Tolerates syntax errors (partial trees)."""
    parser = _parser(spec)
    if parser is None:
        return None
    tree = parser.parse(text.encode("utf-8"))
    return _ts_extract(spec, tree.root_node.named_children)


# ------------------------------------------------------------------------------------------
# Python stdlib `ast` fallback (same symbol model as the tree-sitter Python grammar)
# ------------------------------------------------------------------------------------------
_PY_DEFS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _py_nodes(body: list[ast.stmt]) -> list[SymbolNode]:
    out: list[SymbolNode] = []
    for node in body:
        if not isinstance(node, _PY_DEFS):
            continue
        start = min([node.lineno] + [d.lineno for d in node.decorator_list])
        is_class = isinstance(node, ast.ClassDef)
        members = _py_nodes(node.body) if is_class else []
        out.append(SymbolNode(node.name, start, node.end_lineno or node.lineno, node.lineno, members, is_class))
    return out


def python_ast_symbols(text: str) -> list[SymbolNode] | None:
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return None
    return _py_nodes(tree.body)
