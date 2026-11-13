"""SSD tier: memory-mapped node arrays + SQLite (WAL) metadata.

Node id == slot index in every array. Slots of deleted nodes are recycled.
Dense numeric state (positions, velocities, energies) lives in fixed-stride memmaps so
the OS page cache, not the Python heap, decides what stays resident.
"""
from __future__ import annotations

import contextlib
import math
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np

from .config import NodeKind

_SQL_CHUNK = 900


class StoreLockedError(RuntimeError):
    """Another process (watcher, git hook, CLI) holds the store."""


class StoreLock:
    """Exclusive inter-process lock on a store directory (flock on POSIX, msvcrt on Windows).

    Memmaps and the RAM working set are per-process caches: two writers would corrupt each other,
    so a manager holds this lock for its whole lifetime."""

    def __init__(self, directory: Path, timeout_s: float) -> None:
        self.path = Path(directory) / ".lock"
        self.timeout_s = timeout_s
        self._fh = None

    def _try_lock(self) -> bool:
        assert self._fh is not None
        try:
            if os.name == "nt":  # pragma: no cover - platform specific
                import msvcrt

                self._fh.seek(0)
                msvcrt.locking(self._fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError:
            return False

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a+")
        deadline = time.monotonic() + max(self.timeout_s, 0.0)
        while not self._try_lock():
            if time.monotonic() >= deadline:
                try:
                    self._fh.seek(0)
                    holder = self._fh.read().strip() or "unknown"
                except OSError:  # Windows: the holder's locked range is unreadable from another handle
                    holder = "unknown"
                self._fh.close()
                self._fh = None
                raise StoreLockedError(f"store {self.path.parent} is locked by pid {holder}")
            time.sleep(0.05)
        self._fh.seek(0)
        self._fh.truncate()
        self._fh.write(str(os.getpid()))
        self._fh.flush()

    def release(self) -> None:
        if self._fh is None:
            return
        try:
            if os.name == "nt":  # pragma: no cover
                import msvcrt

                self._fh.seek(0)
                msvcrt.locking(self._fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
        finally:
            self._fh.close()
            self._fh = None


@dataclass(frozen=True, slots=True)
class FileState:
    size: int
    mtime_ns: int
    digest: str  # blake2b of raw bytes; "-" for files present but not indexable (binary, too large)


class GrowableMemmap:
    def __init__(self, path: Path, dtype: np.dtype | type, row_shape: tuple[int, ...] = (), initial_rows: int = 1024) -> None:
        self.path = Path(path)
        self.dtype = np.dtype(dtype)
        self.row_shape = tuple(row_shape)
        self._row_bytes = self.dtype.itemsize * int(math.prod(self.row_shape))
        if not self.path.exists() or self.path.stat().st_size < self._row_bytes:
            with open(self.path, "wb") as fh:
                fh.truncate(max(initial_rows, 1) * self._row_bytes)
        self._mm: np.memmap | None = None
        self._open()

    def _open(self) -> None:
        rows = self.path.stat().st_size // self._row_bytes
        self._mm = np.memmap(self.path, dtype=self.dtype, mode="r+", shape=(rows, *self.row_shape))

    @property
    def array(self) -> np.memmap:
        if self._mm is None:
            raise RuntimeError(f"{self.path} is closed")
        return self._mm

    @property
    def capacity(self) -> int:
        return int(self.array.shape[0])

    def ensure(self, rows: int) -> None:
        if rows <= self.capacity:
            return
        new_rows = max(rows, 2 * self.capacity)
        self.array.flush()
        self._mm = None
        with open(self.path, "r+b") as fh:
            fh.truncate(new_rows * self._row_bytes)
        self._open()

    def flush(self) -> None:
        if self._mm is not None:
            self._mm.flush()

    def close(self) -> None:
        self.flush()
        self._mm = None


class NodeArrays:
    def __init__(self, directory: Path, rank: int, initial_rows: int = 1024) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.rank = rank
        self.position = GrowableMemmap(directory / "position.f32", np.float32, (rank,), initial_rows)
        self.velocity = GrowableMemmap(directory / "velocity.f32", np.float32, (rank,), initial_rows)
        self.energy = GrowableMemmap(directory / "energy.f32", np.float32, (), initial_rows)
        self.last_t = GrowableMemmap(directory / "last_t.i64", np.int64, (), initial_rows)
        self.kind = GrowableMemmap(directory / "kind.u8", np.uint8, (), initial_rows)
        self.alive = GrowableMemmap(directory / "alive.u8", np.uint8, (), initial_rows)
        self._all = (self.position, self.velocity, self.energy, self.last_t, self.kind, self.alive)

    def ensure(self, rows: int) -> None:
        for mm in self._all:
            mm.ensure(rows)

    @property
    def capacity(self) -> int:
        return min(mm.capacity for mm in self._all)

    def init_slots(self, ids: np.ndarray, kinds: np.ndarray, positions: np.ndarray, now: int) -> None:
        self.ensure(int(ids.max()) + 1)
        self.position.array[ids] = positions
        self.velocity.array[ids] = 0.0
        self.energy.array[ids] = 0.0
        self.last_t.array[ids] = now - 1
        self.kind.array[ids] = kinds
        self.alive.array[ids] = 1

    def flush(self) -> None:
        for mm in self._all:
            mm.flush()

    def close(self) -> None:
        for mm in self._all:
            mm.close()

    def disk_bytes(self) -> int:
        return sum(mm.path.stat().st_size for mm in self._all)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS nodes(
    id INTEGER PRIMARY KEY,
    key TEXT NOT NULL UNIQUE,
    kind INTEGER NOT NULL,
    parent_id INTEGER,
    path TEXT,
    symbol TEXT,
    start_line INTEGER,
    end_line INTEGER,
    content_hash TEXT,
    payload TEXT,
    created_t INTEGER NOT NULL,
    updated_t INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_nodes_parent ON nodes(parent_id);
CREATE INDEX IF NOT EXISTS idx_nodes_path_kind ON nodes(path, kind);
CREATE INDEX IF NOT EXISTS idx_nodes_kind ON nodes(kind);
CREATE TABLE IF NOT EXISTS edges(
    src INTEGER NOT NULL,
    dst INTEGER NOT NULL,
    kind TEXT NOT NULL,
    weight REAL NOT NULL,
    PRIMARY KEY(src, dst, kind)
);
CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst);
CREATE TABLE IF NOT EXISTS free_slots(slot INTEGER PRIMARY KEY);
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS file_state(
    path TEXT PRIMARY KEY,
    size INTEGER NOT NULL,
    mtime_ns INTEGER NOT NULL,
    digest TEXT NOT NULL
);
"""

_COLUMNS = (
    "id", "key", "kind", "parent_id", "path", "symbol", "start_line", "end_line",
    "content_hash", "payload", "created_t", "updated_t",
)
_SELECT = f"SELECT {', '.join(_COLUMNS)} FROM nodes"
_UPDATABLE = frozenset(_COLUMNS) - {"id", "key", "kind", "created_t"}


@dataclass(frozen=True, slots=True)
class NodeRecord:
    id: int
    key: str
    kind: NodeKind
    parent_id: int | None
    path: str | None
    symbol: str | None
    start_line: int | None
    end_line: int | None
    content_hash: str | None
    payload: str | None
    created_t: int
    updated_t: int

    @classmethod
    def from_row(cls, row: tuple) -> "NodeRecord":
        return cls(row[0], row[1], NodeKind(row[2]), *row[3:])

    def as_row(self) -> tuple:
        return (self.id, self.key, int(self.kind), self.parent_id, self.path, self.symbol, self.start_line,
                self.end_line, self.content_hash, self.payload, self.created_t, self.updated_t)


@dataclass(frozen=True, slots=True)
class Edge:
    src: int
    dst: int
    kind: str
    weight: float


def _chunks(seq: Sequence, n: int = _SQL_CHUNK) -> Iterator[Sequence]:
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


class MetadataStore:
    def __init__(self, path: Path, cache_kib: int = 65_536) -> None:
        self._conn = sqlite3.connect(str(path), isolation_level=None, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute(f"PRAGMA cache_size=-{int(cache_kib)}")
        self._conn.execute("PRAGMA temp_store=MEMORY")
        self._conn.executescript(_SCHEMA)
        self._depth = 0

    @property
    def connection(self) -> sqlite3.Connection:
        return self._conn

    # -- transactions -------------------------------------------------------------------------
    @contextlib.contextmanager
    def transaction(self) -> Iterator[None]:
        if self._depth == 0:
            self._conn.execute("BEGIN IMMEDIATE")
        self._depth += 1
        try:
            yield
        except BaseException:
            self._depth -= 1
            if self._depth == 0:
                self._conn.execute("ROLLBACK")
            raise
        self._depth -= 1
        if self._depth == 0:
            self._conn.execute("COMMIT")

    # -- meta ---------------------------------------------------------------------------------
    def get_meta(self, key: str, default: str | None = None) -> str | None:
        row = self._conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return default if row is None else row[0]

    def set_meta(self, key: str, value: str) -> None:
        self._conn.execute("INSERT INTO meta(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))

    def high_water(self) -> int:
        return int(self.get_meta("next_slot", "0") or 0)

    def allocate_slots(self, n: int) -> list[int]:
        if n <= 0:
            return []
        slots = [r[0] for r in self._conn.execute("SELECT slot FROM free_slots ORDER BY slot LIMIT ?", (n,))]
        for part in _chunks(slots):
            self._conn.execute(f"DELETE FROM free_slots WHERE slot IN ({','.join('?' * len(part))})", tuple(part))
        remaining = n - len(slots)
        if remaining:
            hw = self.high_water()
            slots.extend(range(hw, hw + remaining))
            self.set_meta("next_slot", str(hw + remaining))
        return slots

    # -- nodes --------------------------------------------------------------------------------
    def insert(self, records: Sequence[NodeRecord]) -> None:
        self._conn.executemany(f"INSERT INTO nodes({', '.join(_COLUMNS)}) VALUES({','.join('?' * len(_COLUMNS))})", [r.as_row() for r in records])

    def update(self, node_id: int, **fields: object) -> None:
        bad = set(fields) - _UPDATABLE
        if bad:
            raise ValueError(f"non-updatable fields: {sorted(bad)}")
        if not fields:
            return
        assignments = ", ".join(f"{k}=?" for k in fields)
        self._conn.execute(f"UPDATE nodes SET {assignments} WHERE id=?", (*fields.values(), node_id))

    def delete(self, ids: Sequence[int]) -> None:
        ids = [int(i) for i in ids]
        for part in _chunks(ids):
            marks = ",".join("?" * len(part))
            self._conn.execute(f"DELETE FROM nodes WHERE id IN ({marks})", tuple(part))
            self._conn.execute(f"DELETE FROM edges WHERE src IN ({marks}) OR dst IN ({marks})", (*part, *part))
        self._conn.executemany("INSERT OR IGNORE INTO free_slots(slot) VALUES(?)", [(i,) for i in ids])

    def get(self, node_id: int) -> NodeRecord | None:
        row = self._conn.execute(f"{_SELECT} WHERE id=?", (int(node_id),)).fetchone()
        return None if row is None else NodeRecord.from_row(row)

    def get_by_key(self, key: str) -> NodeRecord | None:
        row = self._conn.execute(f"{_SELECT} WHERE key=?", (key,)).fetchone()
        return None if row is None else NodeRecord.from_row(row)

    def get_many(self, ids: Iterable[int]) -> dict[int, NodeRecord]:
        ids = list(dict.fromkeys(int(i) for i in ids))
        out: dict[int, NodeRecord] = {}
        for part in _chunks(ids):
            for row in self._conn.execute(f"{_SELECT} WHERE id IN ({','.join('?' * len(part))})", tuple(part)):
                rec = NodeRecord.from_row(row)
                out[rec.id] = rec
        return out

    def chunks_of_file(self, path: str) -> list[NodeRecord]:
        rows = self._conn.execute(f"{_SELECT} WHERE path=? AND kind=?", (path, int(NodeKind.CHUNK)))
        return [NodeRecord.from_row(r) for r in rows]

    def children(self, parent_id: int) -> list[NodeRecord]:
        return [NodeRecord.from_row(r) for r in self._conn.execute(f"{_SELECT} WHERE parent_id=?", (int(parent_id),))]

    def by_kind(self, kind: NodeKind) -> list[NodeRecord]:
        return [NodeRecord.from_row(r) for r in self._conn.execute(f"{_SELECT} WHERE kind=? ORDER BY path, id", (int(kind),))]

    def ids_by_kinds(self, kinds: Iterable[NodeKind]) -> list[int]:
        ks = [int(k) for k in kinds]
        rows = self._conn.execute(f"SELECT id FROM nodes WHERE kind IN ({','.join('?' * len(ks))}) ORDER BY id", tuple(ks))
        return [r[0] for r in rows]

    def count_by_kind(self) -> dict[NodeKind, int]:
        out = {k: 0 for k in NodeKind}
        for kind, n in self._conn.execute("SELECT kind, COUNT(*) FROM nodes GROUP BY kind"):
            out[NodeKind(kind)] = n
        return out

    def file_paths(self) -> list[str]:
        return [r[0] for r in self._conn.execute("SELECT path FROM nodes WHERE kind=? ORDER BY path", (int(NodeKind.FILE),))]

    # -- file state (stat cache for incremental sync) ------------------------------------------
    def file_states(self) -> dict[str, FileState]:
        return {p: FileState(s, m, d) for p, s, m, d in self._conn.execute("SELECT path, size, mtime_ns, digest FROM file_state")}

    def set_file_states(self, rows: Iterable[tuple[str, FileState]]) -> None:
        self._conn.executemany(
            "INSERT INTO file_state(path, size, mtime_ns, digest) VALUES(?, ?, ?, ?) "
            "ON CONFLICT(path) DO UPDATE SET size=excluded.size, mtime_ns=excluded.mtime_ns, digest=excluded.digest",
            [(p, st.size, st.mtime_ns, st.digest) for p, st in rows],
        )

    def delete_file_states(self, paths: Iterable[str]) -> None:
        self._conn.executemany("DELETE FROM file_state WHERE path=?", [(p,) for p in paths])

    # -- edges --------------------------------------------------------------------------------
    def add_edges(self, edges: Iterable[Edge]) -> None:
        self._conn.executemany(
            "INSERT INTO edges(src, dst, kind, weight) VALUES(?, ?, ?, ?) "
            "ON CONFLICT(src, dst, kind) DO UPDATE SET weight=excluded.weight",
            [(e.src, e.dst, e.kind, e.weight) for e in edges],
        )

    def neighbors(self, ids: Iterable[int]) -> list[Edge]:
        ids = list(dict.fromkeys(int(i) for i in ids))
        out: list[Edge] = []
        for part in _chunks(ids, _SQL_CHUNK // 2):
            marks = ",".join("?" * len(part))
            rows = self._conn.execute(f"SELECT src, dst, kind, weight FROM edges WHERE src IN ({marks}) OR dst IN ({marks})", (*part, *part))
            out.extend(Edge(*r) for r in rows)
        return out

    # -- lifecycle ----------------------------------------------------------------------------
    def checkpoint(self) -> None:
        self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    def close(self) -> None:
        self._conn.close()
