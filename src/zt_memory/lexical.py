"""Lexical tier: SQLite FTS5 (BM25) inside the metadata DB, so it shares transactions with the nodes.

Code-aware analysis is done in Python before FTS5 sees the text: every identifier is indexed both
whole (`parse_json_config`, `econnreset`, `0x80070005`) and split into sub-tokens
(`parse`, `json`, `config`; CamelCase too). Exact names therefore match their own rare token
(high IDF) while natural-language queries still match sub-tokens.

When SQLite >= 3.43 the table is contentless (`content=''`, `contentless_delete=1`): only the
inverted index is stored, code text is never duplicated on SSD.
"""
from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from typing import Iterable, Sequence

from .config import NodeKind

LEXICON_VERSION = "2"  # v2: porter stemming (bump => automatic rebuild on open)

_WORD = re.compile(r"[^\W\d]\w*|\d\w*")
_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")
_QUOTED = re.compile(r"`([^`]+)`|\"([^\"]+)\"|'([^']{3,})'")

# Grammar words only: domain words ("file", "error", "config") must stay searchable.
LEXICAL_STOPWORDS = frozenset(
    {
        "a", "an", "the", "and", "or", "of", "to", "in", "on", "at", "by", "for", "with", "from", "into", "is",
        "are", "be", "it", "its", "this", "that", "these", "those", "as", "if", "when", "where", "which", "what",
        "why", "how", "do", "does", "should", "would", "could", "can", "please", "so", "but", "not", "no",
        "il", "lo", "la", "i", "gli", "le", "un", "una", "di", "da", "che", "per", "con", "su", "del", "della",
        "we", "you", "our", "your", "my", "me", "us", "they", "there", "here", "then", "than", "also", "only",
        "def", "self", "return", "import", "class", "function", "const", "let", "var",
    }
)


def split_identifier(word: str) -> list[str]:
    parts: list[str] = []
    for piece in word.split("_"):
        if piece:
            parts.extend(p.lower() for p in _CAMEL.findall(piece) or [piece])
    return [p for p in parts if len(p) >= 2]


def lexical_tokens(text: str) -> list[str]:
    """Whole identifiers (lowercased) followed by their sub-tokens, in source order."""
    out: list[str] = []
    for m in _WORD.finditer(text):
        word = m.group(0)
        low = word.lower()
        out.append(low)
        subs = split_identifier(word)
        if len(subs) > 1 or (subs and subs[0] != low):
            out.extend(s for s in subs if s != low)
    return out


def _looks_like_code(word: str) -> bool:
    return (
        "_" in word.strip("_")
        or bool(re.search(r"[a-z][A-Z]", word))
        or (word.isupper() and len(word) >= 3)
        or (any(c.isdigit() for c in word) and any(c.isalpha() for c in word))
        or word.endswith(("Error", "Exception"))
    )


@dataclass(frozen=True)
class LexicalQuery:
    expression: str  # FTS5 MATCH expression ("" when nothing searchable)
    terms: tuple[str, ...]
    phrases: tuple[str, ...]
    code_like: bool  # identifiers, error codes or quoted literals present


def build_query(query: str, *, max_terms: int = 32) -> LexicalQuery:
    phrases: list[str] = []
    for groups in _QUOTED.findall(query):
        raw = next(g for g in groups if g)
        toks = lexical_tokens(raw)
        if toks:
            phrases.append(" ".join(toks))
    words = [m.group(0) for m in _WORD.finditer(query)]
    code_like = bool(phrases) or any(_looks_like_code(w) for w in words) or bool(re.search(r"\w\.\w|\w::\w|\w\(\)", query))
    terms = list(dict.fromkeys(t for t in lexical_tokens(query) if t not in LEXICAL_STOPWORDS and len(t) >= 2))[:max_terms]
    parts = [f'"{p}"' for p in phrases] + [f'"{t}"' for t in terms]  # tokens are \w-only: safe to quote
    return LexicalQuery(" OR ".join(parts), tuple(terms), tuple(phrases), code_like)


@dataclass(frozen=True)
class LexicalHit:
    node_id: int
    rank: int  # 0-based
    bm25: float  # lower is better (FTS5 convention)


class LexicalIndex:
    TABLE = "lexicon"

    def __init__(self, conn: sqlite3.Connection, *, weights: tuple[float, float, float] = (6.0, 2.0, 1.0)) -> None:
        self._conn = conn
        self._weights = tuple(float(w) for w in weights)
        self.contentless = sqlite3.sqlite_version_info >= (3, 43, 0)

    # -- schema -------------------------------------------------------------------------------
    def exists(self) -> bool:
        row = self._conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (self.TABLE,)).fetchone()
        return row is not None

    def create(self) -> None:
        """(Re)create an empty table. Caller owns the transaction and the backfill."""
        self._conn.execute(f"DROP TABLE IF EXISTS {self.TABLE}")
        opts = ", content='', contentless_delete=1" if self.contentless else ""
        self._conn.execute(
            f"CREATE VIRTUAL TABLE {self.TABLE} USING fts5(symbol, path, body, "
            # porter wraps unicode61: "syncing"/"sync", "parsing"/"parse" meet; index and query are stemmed
            # identically, so exact identifier tokens still match only themselves.
            f"tokenize=\"porter unicode61 remove_diacritics 0 tokenchars '_'\"{opts})"
        )

    # -- maintenance --------------------------------------------------------------------------
    def upsert(self, rows: Iterable[tuple[int, str | None, str | None, str | None]]) -> None:
        rows = list(rows)
        if not rows:
            return
        self.delete([r[0] for r in rows])
        self._conn.executemany(
            f"INSERT INTO {self.TABLE}(rowid, symbol, path, body) VALUES(?, ?, ?, ?)",
            [(int(i), " ".join(lexical_tokens(s or "")), " ".join(lexical_tokens(p or "")), " ".join(lexical_tokens(b or "")))
             for i, s, p, b in rows],
        )

    def delete(self, ids: Sequence[int]) -> None:
        ids = [int(i) for i in ids]
        for i in range(0, len(ids), 900):
            part = ids[i : i + 900]
            self._conn.execute(f"DELETE FROM {self.TABLE} WHERE rowid IN ({','.join('?' * len(part))})", part)

    def count(self) -> int:
        return int(self._conn.execute(f"SELECT COUNT(*) FROM {self.TABLE}").fetchone()[0])

    # -- query --------------------------------------------------------------------------------
    def search(self, query: str | LexicalQuery, k: int, kinds: frozenset[NodeKind] | None = None) -> list[LexicalHit]:
        q = build_query(query) if isinstance(query, str) else query
        if not q.expression or k <= 0:
            return []
        w = ", ".join(repr(x) for x in self._weights)
        sql = f"SELECT {self.TABLE}.rowid, bm25({self.TABLE}, {w}) AS s FROM {self.TABLE}"
        params: list[object] = []
        if kinds is not None:
            ks = [int(x) for x in kinds]
            sql += f" JOIN nodes ON nodes.id = {self.TABLE}.rowid WHERE {self.TABLE} MATCH ? AND nodes.kind IN ({','.join('?' * len(ks))})"
            params = [q.expression, *ks]
        else:
            sql += f" WHERE {self.TABLE} MATCH ?"
            params = [q.expression]
        sql += " ORDER BY s LIMIT ?"
        params.append(int(k))
        try:
            rows = self._conn.execute(sql, params).fetchall()
        except sqlite3.OperationalError:  # malformed expression: degrade to no lexical evidence
            return []
        return [LexicalHit(int(r[0]), i, float(r[1])) for i, r in enumerate(rows)]
