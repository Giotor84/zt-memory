"""NewtonianMemoryManager: SSD-resident hyperbolic memory with Newtonian dynamics.

Positions are content-anchored (a chunk's point is the projection of its embedding),
so retrieval never drifts away from the code it addresses. Dynamics (velocity, energy)
is the state that evolves: surprise exerts force, inertia accumulates recurring
patterns, friction cools nodes down until the working set pages them out to SSD.

The store is a *derived* index of the repository plus the textual specs/decisions:
code text is never duplicated, it is resolved from disk (path + span + hash) at routing time.
"""
from __future__ import annotations

import hashlib
import os
import threading
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Iterable, Sequence

import numpy as np
import torch
from torch import Tensor

from . import hyperbolic as H
from .calibration import stdlib_calibration_corpus
from .chunking import CodeChunk, chunk_source, chunker_fingerprint, content_hash, embedding_text, slice_lines
from .discovery import discover, is_indexable
from .config import MEMORY_KINDS, MemoryConfig, NodeKind
from .dynamics import current_energy, newtonian_step, surprise_force
from .encoders import HiddenStateEncoder
from .index import BlockScanIndex, HnswIndex, VectorIndex, kind_mask
from .lexical import LEXICON_VERSION, LexicalHit, LexicalIndex, LexicalQuery
from .projector import SubspaceProjector
from .storage import Edge, FileState, MetadataStore, NodeArrays, NodeRecord, StoreLock, StoreLockedError
from .working_set import WorkingSet, WorkingSetStats

ROOT_KEY = "root"


class IncompatibleStoreError(RuntimeError):
    """Store was built with a different encoder/rank/geometry; rebuild it."""


@dataclass
class IngestReport:
    tick: int = 0
    files_seen: int = 0
    files_skipped: int = 0
    files_removed: int = 0
    chunks_added: int = 0
    chunks_updated: int = 0
    chunks_removed: int = 0
    chunks_unchanged: int = 0
    surprise_sum: float = 0.0
    surprise_max: float = 0.0
    surprise_events: int = 0
    changed_symbols: list[str] = field(default_factory=list)

    def record_surprise(self, values: Tensor) -> None:
        if values.numel():
            self.surprise_sum += float(values.sum())
            self.surprise_max = max(self.surprise_max, float(values.max()))
            self.surprise_events += int(values.numel())

    @property
    def mean_surprise(self) -> float:
        return self.surprise_sum / self.surprise_events if self.surprise_events else 0.0


@dataclass
class ChangeSet:
    """Result of a stat scan: what a sync would ingest, without touching the store."""

    scanned: int
    added: list[str]
    changed: list[str]
    removed: list[str]
    touched: list[tuple[str, FileState]]  # mtime changed, bytes identical: stat refresh only
    full: bool

    @property
    def pending(self) -> list[str]:
        return [*self.added, *self.changed, *self.removed]


@dataclass
class SyncReport:
    ingest: IngestReport
    scanned: int
    added: int
    changed: int
    removed: int
    touched: int
    seconds: float
    full: bool

    def to_dict(self) -> dict[str, object]:
        r = self.ingest
        return {
            "scanned": self.scanned, "files_added": self.added, "files_changed": self.changed,
            "files_removed": self.removed, "files_touched": self.touched, "full": self.full,
            "chunks_added": r.chunks_added, "chunks_updated": r.chunks_updated, "chunks_removed": r.chunks_removed,
            "chunks_unchanged": r.chunks_unchanged, "mean_surprise": round(r.mean_surprise, 4),
            "max_surprise": round(r.surprise_max, 4), "tick": r.tick, "seconds": round(self.seconds, 3),
        }

    def summary(self) -> str:
        r = self.ingest
        return (f"scanned {self.scanned} files: +{self.added} ~{self.changed} -{self.removed} (touched {self.touched}) | "
                f"chunks +{r.chunks_added} ~{r.chunks_updated} -{r.chunks_removed} | "
                f"surprise mean {r.mean_surprise:.3f} max {r.surprise_max:.3f} | {self.seconds:.2f}s")


@dataclass(frozen=True)
class Requirement:
    text: str
    key: str | None = None


@dataclass
class MemoryStats:
    clock: int
    nodes_by_kind: dict[str, int]
    working_set: WorkingSetStats
    disk_bytes: int
    index_backend: str
    projector_fingerprint: str


@dataclass
class _ChunkPlan:
    rel_path: str
    chunk: CodeChunk
    record: NodeRecord | None
    file_id: int

    @property
    def embed_text(self) -> str:
        return embedding_text(self.rel_path, self.chunk)


@dataclass
class _SyncContext:
    now: int
    ancestor_forces: dict[int, Tensor] = field(default_factory=dict)
    touched_structural: set[int] = field(default_factory=set)
    created: list[int] = field(default_factory=list)
    killed: list[int] = field(default_factory=list)
    killed_set: set[int] = field(default_factory=set)
    parent_cache: dict[int, int | None] = field(default_factory=dict)
    key_cache: dict[str, int] = field(default_factory=dict)
    file_states: dict[str, FileState] = field(default_factory=dict)
    gone_paths: set[str] = field(default_factory=set)


class NewtonianMemoryManager:
    def __init__(self, config: MemoryConfig, encoder: HiddenStateEncoder, *, projector: SubspaceProjector | None = None) -> None:
        self.config = config
        self.encoder = encoder
        self._c = config.curvature
        self._lock = threading.RLock()
        self._closed = False
        config.storage_dir.mkdir(parents=True, exist_ok=True)
        self._store_lock = StoreLock(config.storage_dir, config.lock_timeout_s)
        self._store_lock.acquire()
        try:
            self._open(projector)
        except BaseException:
            if getattr(self, "meta", None) is not None:
                self.meta.close()
            self._store_lock.release()
            raise

    def _open(self, projector: SubspaceProjector | None) -> None:
        config = self.config
        self.meta = MetadataStore(config.storage_dir / "meta.sqlite3", config.sqlite_cache_kib)
        self._check_compatibility()
        self.arrays = NodeArrays(config.storage_dir / "arrays", config.rank)
        self.arrays.ensure(self.meta.high_water() + 1)
        phys = config.physics
        self._beta = np.array([phys[k].beta for k in NodeKind], dtype=np.float64)
        self._gamma = np.array([phys[k].gamma for k in NodeKind], dtype=np.float64)
        self._radius = np.array([phys[k].radius for k in NodeKind], dtype=np.float32)
        self._projector_path = config.storage_dir / "projector.pt"
        self.projector = self._load_projector(projector)
        self.ws = WorkingSet(
            self.arrays, config.rank, config.ram_budget_bytes, self._beta, self._gamma,
            eviction_fraction=config.eviction_fraction, ram_floor_bytes=config.ram_floor_bytes,
        )
        self.index, self._index_backend = self._make_index()
        self._clock = int(self.meta.get_meta("clock", "0") or 0)
        self.lexicon = LexicalIndex(self.meta.connection)
        self._root_id = self._ensure_root()
        self._check_chunker()
        if self.meta.get_meta("lexicon") != LEXICON_VERSION or not self.lexicon.exists():
            self.rebuild_lexicon()

    # ======================================================================================
    # setup
    # ======================================================================================
    def _check_compatibility(self) -> None:
        cfg = self.config
        expected = {
            "rank": str(cfg.rank),
            "curvature": repr(cfg.curvature),
            "encoder": self.encoder.fingerprint,
            "encoder_dim": str(self.encoder.dim),
            "radii": ",".join(f"{cfg.physics[k].radius:.6f}" for k in NodeKind),
        }
        for key, value in expected.items():
            stored = self.meta.get_meta(key)
            if stored is None:
                self.meta.set_meta(key, value)
            elif stored != value:
                raise IncompatibleStoreError(f"store {cfg.storage_dir} has {key}={stored!r}, config wants {value!r}")

    def _check_chunker(self) -> None:
        current = chunker_fingerprint() + f":{self.config.chunk_backend}:{self.config.chunk_max_lines}:{self.config.chunk_min_lines}"
        stored = self.meta.get_meta("chunker")
        if stored is not None and stored != current and self.meta.count_by_kind()[NodeKind.CHUNK] > 0:
            warnings.warn(
                f"chunker changed ({stored} -> {current}): the next sync re-keys chunks of affected languages "
                "(remove+add, their energy restarts)", RuntimeWarning, stacklevel=3,
            )
        self.meta.set_meta("chunker", current)

    def _load_projector(self, projector: SubspaceProjector | None) -> SubspaceProjector:
        if projector is not None:
            if projector.rank != self.config.rank:
                raise ValueError("projector rank != config rank")
            if projector.frozen:
                projector.save(self._projector_path)
            return projector
        if self._projector_path.exists():
            proj = SubspaceProjector.load(self._projector_path)
            if proj.rank != self.config.rank or proj.dim != self.encoder.dim:
                raise IncompatibleStoreError("stored projector does not match rank/encoder dim")
            return proj
        if self.meta.high_water() > 1:
            raise IncompatibleStoreError("store has nodes but no projector basis; rebuild it")
        return SubspaceProjector(self.config.rank, drop_top=self.config.svd_drop_top, center=self.config.svd_center)

    def _make_index(self) -> tuple[VectorIndex, str]:
        cfg = self.config
        backend = cfg.index_backend
        if backend == "auto":
            backend = "hnsw" if self.meta.high_water() > cfg.hnsw_threshold else "scan"
        if backend == "hnsw":
            return HnswIndex(cfg.storage_dir / "index.hnsw", self.arrays, self.meta.high_water, self._c, cfg.rank), "hnsw"
        return BlockScanIndex(self.arrays, self.meta.high_water, self._c, cfg.scan_block_rows), "scan"

    def _ensure_root(self) -> int:
        rec = self.meta.get_by_key(ROOT_KEY)
        if rec is not None:
            return rec.id
        with self.meta.transaction():
            ctx = _SyncContext(now=self._clock)
            [root_id] = self._create_nodes(
                [(ROOT_KEY, NodeKind.ROOT, None, "", None, None, None, None, None)],
                np.zeros((1, self.config.rank), dtype=np.float32), ctx,
            )
        return root_id

    # ======================================================================================
    # basic accessors
    # ======================================================================================
    @property
    def lock(self) -> threading.RLock:
        return self._lock

    @property
    def clock(self) -> int:
        return self._clock

    @property
    def root_id(self) -> int:
        return self._root_id

    def _tick(self) -> int:
        self._clock += 1
        return self._clock

    def radius_of(self, kind: NodeKind) -> float:
        return float(self._radius[int(kind)])

    def record(self, node_id: int) -> NodeRecord | None:
        return self.meta.get(node_id)

    def record_by_key(self, key: str) -> NodeRecord | None:
        return self.meta.get_by_key(key)

    def records(self, ids: Iterable[int]) -> dict[int, NodeRecord]:
        return self.meta.get_many(ids)

    def positions(self, ids: Sequence[int] | np.ndarray) -> np.ndarray:
        return np.array(self.arrays.position.array[np.asarray(ids, dtype=np.int64)], dtype=np.float32)

    def neighbors(self, ids: Iterable[int]) -> list[Edge]:
        return self.meta.neighbors(ids)

    def file_paths(self) -> list[str]:
        return self.meta.file_paths()

    def project_mode(self) -> str | None:
        return self.meta.get_meta("project_mode")

    def count_by_kind(self) -> dict[NodeKind, int]:
        return self.meta.count_by_kind()

    # ======================================================================================
    # embedding pipeline: encoder (hidden states) -> truncated SVD -> Poincaré ball
    # ======================================================================================
    def calibrate(self, texts: Sequence[str] | None = None) -> None:
        """Fit and freeze the SVD basis. Project text first, stdlib chunks as filler."""
        with self._lock:
            if self.projector.frozen:
                return
            need = max(self.config.calibration_samples, 4 * self.projector.n_target)
            corpus = [t for t in (texts or []) if t.strip()][:need]
            if len(corpus) < need:
                corpus.extend(stdlib_calibration_corpus(need - len(corpus)))
            if len(corpus) < self.projector.n_target:
                raise RuntimeError(f"calibration corpus too small: {len(corpus)} < {self.projector.n_target}")
            group = max(256, self.projector.n_target)
            for start in range(0, len(corpus), group):
                self.projector.partial_fit(self.encoder.encode(corpus[start : start + group]))
            self.projector.freeze()
            self.projector.save(self._projector_path)
            self.meta.set_meta("projector", self.projector.fingerprint)

    def embed_directions(self, texts: Sequence[str], *, is_query: bool = False) -> Tensor:
        """Unit directions in R^rank (float32)."""
        if not self.projector.frozen:
            raise RuntimeError("projector not calibrated: call calibrate() or ingest/seed first")
        if not texts:
            return torch.zeros((0, self.config.rank), dtype=torch.float32)
        z = self.projector.transform(self.encoder.encode(list(texts), is_query=is_query))
        norms = z.norm(dim=-1, keepdim=True)
        fallback = torch.zeros_like(z)
        fallback[:, 0] = 1.0
        return torch.where(norms > 1e-8, z / norms.clamp_min(1e-12), fallback)

    def embed_query(self, query: str, kind: NodeKind = NodeKind.CHUNK) -> np.ndarray:
        # The frozen projector makes a query's direction immutable: cache it (repeated routing/evaluation of the
        # same query with a neural encoder would otherwise pay a full forward pass each time).
        cache: dict[str, np.ndarray] = self.__dict__.setdefault("_query_cache", {})
        direction = cache.get(query)
        if direction is None:
            direction = self.embed_directions([query], is_query=True)[0].numpy()
            if len(cache) >= 256:
                cache.pop(next(iter(cache)))
            cache[query] = direction
        return (direction * self.radius_of(kind)).astype(np.float32)

    def _place(self, directions: Tensor, kinds: np.ndarray) -> np.ndarray:
        return (directions * torch.from_numpy(self._radius[kinds])[:, None]).numpy().astype(np.float32)

    # ======================================================================================
    # node lifecycle
    # ======================================================================================
    def _create_nodes(
        self,
        specs: Sequence[tuple[str, NodeKind, int | None, str | None, str | None, int | None, int | None, str | None, str | None]],
        positions: np.ndarray,
        ctx: _SyncContext,
    ) -> list[int]:
        """specs: (key, kind, parent_id, path, symbol, start_line, end_line, content_hash, payload)."""
        if not specs:
            return []
        slots = self.meta.allocate_slots(len(specs))
        now = ctx.now
        records = [
            NodeRecord(slot, key, kind, parent, path, sym, s, e, h, payload, now, now)
            for slot, (key, kind, parent, path, sym, s, e, h, payload) in zip(slots, specs)
        ]
        self.meta.insert(records)
        ids = np.asarray(slots, dtype=np.int64)
        kinds = np.array([int(s[1]) for s in specs], dtype=np.uint8)
        self.ws.drop(ids)  # recycled slots must not carry stale cached state
        self.arrays.init_slots(ids, kinds, positions, now)
        self.index.upsert(ids, positions)
        ctx.created.extend(slots)
        for rec in records:
            ctx.key_cache[rec.key] = rec.id
            ctx.parent_cache[rec.id] = rec.parent_id
        return slots

    def _kill(self, ids: Sequence[int], ctx: _SyncContext) -> None:
        if not ids:
            return
        arr = np.asarray(ids, dtype=np.int64)
        self.meta.delete(ids)
        self.lexicon.delete(ids)
        self.arrays.alive.array[arr] = 0
        self.ws.drop(arr)
        self.index.remove(arr)
        ctx.killed.extend(int(i) for i in ids)
        ctx.killed_set.update(int(i) for i in ids)
        for i in ids:
            ctx.parent_cache.pop(int(i), None)
        ctx.key_cache = {k: v for k, v in ctx.key_cache.items() if v not in ctx.killed_set}

    def _compensate(self, ctx: _SyncContext) -> None:
        """Undo memmap/index side effects after a SQLite rollback."""
        if ctx.created:
            created = np.asarray(ctx.created, dtype=np.int64)
            self.arrays.alive.array[created] = 0
            self.ws.drop(created)
            self.index.remove(created)
        created_set = set(ctx.created)
        revived = [i for i in ctx.killed if i not in created_set]
        if revived:
            arr = np.asarray(revived, dtype=np.int64)
            self.arrays.alive.array[arr] = 1
            self.index.upsert(arr, self.positions(arr))

    def _parent_of(self, node_id: int, ctx: _SyncContext) -> int | None:
        if node_id in ctx.parent_cache:
            return ctx.parent_cache[node_id]
        rec = self.meta.get(node_id)
        parent = None if rec is None else rec.parent_id
        ctx.parent_cache[node_id] = parent
        return parent

    # ======================================================================================
    # dynamics
    # ======================================================================================
    def _apply_forces(self, ids: np.ndarray, forces: Tensor, now: int) -> None:
        """Aggregate forces per node and advance each node by one Newtonian step at `now`."""
        ids = np.asarray(ids, dtype=np.int64)
        if ids.size == 0:
            return
        uniq, inv = np.unique(ids, return_inverse=True)
        agg = torch.zeros((uniq.size, forces.shape[1]), dtype=torch.float64)
        agg.index_add_(0, torch.from_numpy(inv.astype(np.int64)), forces.to(torch.float64))
        step = self.ws.max_batch
        for s in range(0, uniq.size, step):
            batch = uniq[s : s + step]
            slots = self.ws.fetch(batch, now)
            v, e, t, kinds = self.ws.read(slots)
            beta = torch.from_numpy(self._beta[kinds])
            gamma = torch.from_numpy(self._gamma[kinds])
            v2, e2 = newtonian_step(v, e, t, now, agg[s : s + step], beta, gamma)
            self.ws.write(slots, v2, e2, now)

    def _propagate(self, parent_id: int | None, force: Tensor, ctx: _SyncContext) -> None:
        """Attenuated force on the ancestor chain: rho^1 at the parent, rho^2 at the grandparent..."""
        rho = self.config.ancestor_decay
        f = force.to(torch.float64) * rho
        node = parent_id
        while node is not None and rho > 0:
            acc = ctx.ancestor_forces.get(node)
            ctx.ancestor_forces[node] = f.clone() if acc is None else acc + f
            if node != self._root_id:
                ctx.touched_structural.add(node)
            f = f * rho
            node = self._parent_of(node, ctx)

    def _flush_ancestor_forces(self, ctx: _SyncContext) -> None:
        items = [(i, f) for i, f in ctx.ancestor_forces.items() if i not in ctx.killed_set]
        ctx.ancestor_forces.clear()
        if items:
            ids = np.array([i for i, _ in items], dtype=np.int64)
            self._apply_forces(ids, torch.stack([f for _, f in items]), ctx.now)

    def _edit_impulse(self, positions: np.ndarray) -> Tensor:
        """kappa * x_hat in T_0 coordinates (origin Riemannian norm = kappa)."""
        x = torch.from_numpy(np.asarray(positions, dtype=np.float32)).to(torch.float64)
        return (self.config.edit_impulse / 2.0) * x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    def _novelty_forces(self, z: np.ndarray) -> tuple[Tensor, Tensor]:
        """Surprise of brand-new inputs w.r.t. the current memory state (nearest memory node)."""
        zt = torch.from_numpy(z)
        nn_ids, _ = self.index.search(zt, 1, MEMORY_KINDS)
        ref = np.zeros_like(z)
        has = (nn_ids[:, 0] >= 0).numpy()
        if has.any():
            ref[has] = self.positions(nn_ids[:, 0].numpy()[has])
        return surprise_force(torch.from_numpy(ref), zt, self._c)

    def energies(self, ids: Sequence[int] | np.ndarray) -> np.ndarray:
        """Current (lazily decayed) energies, without promoting nodes into the working set."""
        ids = np.asarray(ids, dtype=np.int64)
        if ids.size == 0:
            return np.zeros(0, dtype=np.float64)
        v, e, t, kinds = self.ws.peek(ids)
        return current_energy(
            torch.from_numpy(v.astype(np.float64)), torch.from_numpy(e.astype(np.float64)), torch.from_numpy(t),
            self._clock, torch.from_numpy(self._beta[kinds]), torch.from_numpy(self._gamma[kinds]),
        ).numpy()

    def hottest(self, k: int, kinds: frozenset[NodeKind] | None = None) -> list[tuple[int, float]]:
        """Top-k nodes by current energy across the whole SSD store (block scan)."""
        with self._lock:
            self.ws.flush()
            ok = kind_mask(kinds)
            hw = min(self.meta.high_water(), self.arrays.capacity)
            best_ids = np.empty(0, dtype=np.int64)
            best_e = np.empty(0, dtype=np.float64)
            block = self.config.scan_block_rows
            for start in range(0, hw, block):
                end = min(start + block, hw)
                mask = (self.arrays.alive.array[start:end] == 1) & ok[self.arrays.kind.array[start:end]]
                local = np.flatnonzero(mask)
                if local.size == 0:
                    continue
                ids = local.astype(np.int64) + start
                e = self.energies(ids)
                cat_ids = np.concatenate([best_ids, ids])
                cat_e = np.concatenate([best_e, e])
                keep = np.argsort(-cat_e, kind="stable")[:k]
                best_ids, best_e = cat_ids[keep], cat_e[keep]
            return [(int(i), float(e)) for i, e in zip(best_ids, best_e)]

    def reinforce(self, ids: Sequence[int], query_position: np.ndarray, weights: Sequence[float], *, scale: float | None = None) -> None:
        """Read-side force: retrieved nodes are pulled (angularly) toward the query; they heat up."""
        with self._lock:
            recs = self.meta.get_many(ids)
            pairs = [(i, w) for i, w in zip(ids, weights) if i in recs and w > 0]
            if not pairs:
                return
            now = self._tick()
            node_ids = np.array([i for i, _ in pairs], dtype=np.int64)
            w = torch.tensor([w for _, w in pairs], dtype=torch.float64)
            kinds = np.array([int(recs[i].kind) for i in node_ids], dtype=np.int64)
            q = torch.from_numpy(np.asarray(query_position, dtype=np.float32))
            q_dir = (q / q.norm().clamp_min(1e-12)).expand(node_ids.size, -1)
            target = q_dir * torch.from_numpy(self._radius[kinds])[:, None]
            f0, _ = surprise_force(torch.from_numpy(self.positions(node_ids)), target, self._c)
            s = self.config.query_force_scale if scale is None else scale
            self._apply_forces(node_ids, f0 * (s * w)[:, None], now)

    # ======================================================================================
    # BROWNFIELD: repository analysis (incremental, hash-diffed per symbol)
    # ======================================================================================
    def _normalize_rel(self, path: str | os.PathLike[str]) -> str:
        p = Path(path)
        if p.is_absolute():
            try:
                p = p.resolve().relative_to(self.config.repo_root)
            except ValueError as exc:
                raise ValueError(f"{path} is outside repo_root {self.config.repo_root}") from exc
        rel = PurePosixPath(p.as_posix())
        if rel.is_absolute() or ".." in rel.parts or not rel.parts:
            raise ValueError(f"invalid relative path: {path}")
        return str(rel)

    def _is_indexable(self, rel: str) -> bool:
        return is_indexable(rel, self.config)

    def discover_files(self) -> list[str]:
        """Indexable files under repo_root (git-aware: honours .gitignore inside a work tree)."""
        return discover(self.config)

    def _read_source(self, rel: str) -> str | None:
        path = self.config.repo_root / rel
        try:
            if not path.is_file() or path.stat().st_size > self.config.max_file_bytes:
                return None
            data = path.read_bytes()
        except OSError:
            return None
        if b"\x00" in data[:8192]:
            return None
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            return None

    def _read_source_stat(self, rel: str) -> tuple[str | None, FileState | None]:
        """Text (None if not indexable) and the file's state (None if it does not exist).
        stat() precedes the read: a write racing with us leaves a newer mtime than the one recorded,
        so the next scan re-detects it instead of missing it."""
        path = self.config.repo_root / rel
        try:
            st = path.stat()
            if not path.is_file():
                return None, None
            if st.st_size > self.config.max_file_bytes:
                return None, FileState(st.st_size, st.st_mtime_ns, "-")
            data = path.read_bytes()
        except OSError:
            return None, None
        state = FileState(st.st_size, st.st_mtime_ns, hashlib.blake2b(data, digest_size=16).hexdigest())
        if b"\x00" in data[:8192]:
            return None, state
        try:
            return data.decode("utf-8"), state
        except UnicodeDecodeError:
            return None, state

    def _chunk(self, rel: str, text: str) -> list[CodeChunk]:
        cfg = self.config
        return chunk_source(rel, text, max_lines=cfg.chunk_max_lines, min_lines=cfg.chunk_min_lines, backend=cfg.chunk_backend)

    def ingest_repository(self) -> IngestReport:
        """Full sync: every file is re-chunked and hash-diffed per symbol (no stat shortcut)."""
        return self.sync(full=True).ingest

    def scan_changes(self, *, full: bool = False) -> ChangeSet:
        """Stat-level diff of repo_root against the store. Reads a file only when its mtime moved but
        its size did not (to tell a real edit from a touch/checkout). Never writes."""
        with self._lock:
            on_disk = self.discover_files()
            states = self.meta.file_states()
            if self.meta.get_meta("file_state_chunker") != self.meta.get_meta("chunker"):
                full = True  # chunk identities may differ: every file must be re-chunked
            known_files = set(self.meta.file_paths())
            added: list[str] = []
            changed: list[str] = []
            touched: list[tuple[str, FileState]] = []
            for rel in on_disk:
                prev = states.get(rel)
                if prev is None:
                    (changed if rel in known_files else added).append(rel)
                    continue
                try:
                    st = (self.config.repo_root / rel).stat()
                except OSError:
                    continue
                if not full and (prev.size, prev.mtime_ns) == (st.st_size, st.st_mtime_ns):
                    continue
                if not full and prev.digest != "-" and prev.size == st.st_size:
                    try:
                        digest = hashlib.blake2b((self.config.repo_root / rel).read_bytes(), digest_size=16).hexdigest()
                    except OSError:
                        continue
                    if digest == prev.digest:
                        touched.append((rel, FileState(st.st_size, st.st_mtime_ns, digest)))
                        continue
                changed.append(rel)
            removed = sorted((known_files | set(states)) - set(on_disk))
            return ChangeSet(len(on_disk), added, changed, removed, touched, full)

    def sync(self, *, full: bool = False) -> SyncReport:
        """Bring the store in line with repo_root: the single entry point for automation (CLI, hooks,
        watcher). Unchanged files cost one stat(); only added/changed/removed files are ingested,
        and within them only symbols whose content hash changed are re-embedded."""
        with self._lock:
            t0 = time.perf_counter()
            cs = self.scan_changes(full=full)
            if self.meta.get_meta("project_mode") is None and (cs.added or cs.changed):
                self.meta.set_meta("project_mode", "brownfield")
            report = self.ingest_files(cs.pending) if cs.pending else IngestReport(tick=self._clock)
            with self.meta.transaction():
                if cs.touched:
                    self.meta.set_file_states(cs.touched)
                self.meta.set_meta("file_state_chunker", self.meta.get_meta("chunker") or "")
            return SyncReport(report, cs.scanned, len(cs.added), len(cs.changed), len(cs.removed), len(cs.touched),
                              time.perf_counter() - t0, cs.full)

    def ingest_files(self, paths: Sequence[str | os.PathLike[str]]) -> IngestReport:
        """Incremental sync of specific files (diff-driven). One clock tick per call."""
        rels = list(dict.fromkeys(self._normalize_rel(p) for p in paths))
        with self._lock:
            if not self.projector.frozen:
                self.calibrate(self._calibration_texts(rels))
            now = self._tick()
            report = IngestReport(tick=now)
            ctx = _SyncContext(now=now)
            try:
                with self.meta.transaction():
                    pending: list[_ChunkPlan] = []
                    for rel in rels:
                        self._sync_file(rel, ctx, pending, report)
                        if len(pending) >= self.config.ingest_group_size:
                            self._apply_chunk_group(pending, ctx, report)
                            pending = []
                    if pending:
                        self._apply_chunk_group(pending, ctx, report)
                    self._flush_ancestor_forces(ctx)
                    self._recompute_structural(ctx)
                    self.meta.set_file_states(ctx.file_states.items())
                    self.meta.delete_file_states(ctx.gone_paths)
                    self.meta.set_meta("clock", str(self._clock))
            except BaseException:
                self._compensate(ctx)
                raise
            return report

    def remove_files(self, paths: Sequence[str | os.PathLike[str]]) -> IngestReport:
        rels = list(dict.fromkeys(self._normalize_rel(p) for p in paths))
        with self._lock:
            now = self._tick()
            report = IngestReport(tick=now)
            ctx = _SyncContext(now=now)
            try:
                with self.meta.transaction():
                    for rel in rels:
                        self._remove_file_nodes(rel, ctx, report)
                    self.meta.delete_file_states(rels)
                    self._flush_ancestor_forces(ctx)
                    self._recompute_structural(ctx)
                    self.meta.set_meta("clock", str(self._clock))
            except BaseException:
                self._compensate(ctx)
                raise
            return report

    def _calibration_texts(self, rels: Sequence[str]) -> list[str]:
        out: list[str] = []
        for rel in rels:
            text = self._read_source(rel)
            if text is None:
                continue
            out.extend(embedding_text(rel, c) for c in self._chunk(rel, text))
            if len(out) >= self.config.calibration_samples:
                break
        return out

    def _ensure_path_nodes(self, rel: str, ctx: _SyncContext) -> int:
        parts = PurePosixPath(rel).parts
        parent = self._root_id
        chain: list[tuple[str, NodeKind, str]] = [(f"dir:{'/'.join(parts[:i])}", NodeKind.DIR, "/".join(parts[:i])) for i in range(1, len(parts))]
        chain.append((f"file:{rel}", NodeKind.FILE, rel))
        for key, kind, path in chain:
            node_id = ctx.key_cache.get(key)
            if node_id is None:
                rec = self.meta.get_by_key(key)
                if rec is None:
                    [node_id] = self._create_nodes(
                        [(key, kind, parent, path, None, None, None, None, None)],
                        np.zeros((1, self.config.rank), dtype=np.float32), ctx,
                    )
                    ctx.touched_structural.add(node_id)
                    if parent != self._root_id:
                        ctx.touched_structural.add(parent)
                    if kind == NodeKind.FILE:
                        self.lexicon.upsert([(node_id, None, path, None)])
                else:
                    node_id = rec.id
                    ctx.parent_cache[node_id] = rec.parent_id
                ctx.key_cache[key] = node_id
            parent = node_id
        return parent

    def _sync_file(self, rel: str, ctx: _SyncContext, pending: list[_ChunkPlan], report: IngestReport) -> None:
        text, state = self._read_source_stat(rel) if self._is_indexable(rel) else (None, None)
        if state is not None:
            ctx.file_states[rel] = state
        else:
            ctx.gone_paths.add(rel)
        if text is None:
            if (self.config.repo_root / rel).exists():
                report.files_skipped += 1
            self._remove_file_nodes(rel, ctx, report)
            return
        report.files_seen += 1
        chunks = self._chunk(rel, text)
        existing = {r.symbol: r for r in self.meta.chunks_of_file(rel)}
        file_id = self._ensure_path_nodes(rel, ctx)
        seen: set[str] = set()
        for ch in chunks:
            seen.add(ch.symbol)
            rec = existing.get(ch.symbol)
            if rec is not None and rec.content_hash == ch.content_hash:
                report.chunks_unchanged += 1
                if (rec.start_line, rec.end_line) != (ch.start_line, ch.end_line):
                    self.meta.update(rec.id, start_line=ch.start_line, end_line=ch.end_line)  # pure shift: no force
                continue
            pending.append(_ChunkPlan(rel, ch, rec, file_id))
        stale = [r for sym, r in existing.items() if sym not in seen]
        if stale:
            self._remove_chunks(stale, file_id, ctx, report)

    def _remove_chunks(
        self, records: Sequence[NodeRecord], parent_id: int, ctx: _SyncContext, report: IngestReport, *, propagate: bool = True
    ) -> None:
        ids = np.array([r.id for r in records], dtype=np.int64)
        x = torch.from_numpy(self.positions(ids))
        anchor = torch.from_numpy(self.positions([parent_id])).expand_as(x)
        f0, s = surprise_force(anchor, x, self._c)
        report.record_surprise(s)
        if propagate:
            self._propagate(parent_id, f0.sum(0), ctx)
        self._kill(ids.tolist(), ctx)
        report.chunks_removed += ids.size
        report.changed_symbols.extend(f"-{r.path}::{r.symbol}" for r in records)

    def _remove_file_nodes(self, rel: str, ctx: _SyncContext, report: IngestReport) -> None:
        file_rec = self.meta.get_by_key(f"file:{rel}")
        if file_rec is None:
            return
        chunks = self.meta.chunks_of_file(rel)
        if chunks:
            self._remove_chunks(chunks, file_rec.id, ctx, report, propagate=False)
        parent = file_rec.parent_id
        if parent is not None:
            x = torch.from_numpy(self.positions([file_rec.id]))
            f0, _ = surprise_force(torch.from_numpy(self.positions([parent])), x, self._c)
            self._propagate(parent, f0[0], ctx)
        ctx.ancestor_forces.pop(file_rec.id, None)
        ctx.touched_structural.discard(file_rec.id)
        self._kill([file_rec.id], ctx)
        report.files_removed += 1

    def _apply_chunk_group(self, plans: Sequence[_ChunkPlan], ctx: _SyncContext, report: IngestReport) -> None:
        n = len(plans)
        now = ctx.now
        kinds = np.full(n, int(NodeKind.CHUNK), dtype=np.int64)
        pos = self._place(self.embed_directions([p.embed_text for p in plans]), kinds)
        forces = torch.zeros((n, self.config.rank), dtype=torch.float64)
        ids = np.empty(n, dtype=np.int64)
        upd = [i for i, p in enumerate(plans) if p.record is not None]
        new = [i for i, p in enumerate(plans) if p.record is None]

        if upd:
            old_ids = np.array([plans[i].record.id for i in upd], dtype=np.int64)  # type: ignore[union-attr]
            f0, s = surprise_force(torch.from_numpy(self.positions(old_ids)), torch.from_numpy(pos[upd]), self._c)
            forces[upd] = f0 + self._edit_impulse(pos[upd])
            report.record_surprise(s)
            self.arrays.position.array[old_ids] = pos[upd]  # content-anchored: write-through
            self.index.upsert(old_ids, pos[upd])
            for i, nid in zip(upd, old_ids.tolist()):
                ch = plans[i].chunk
                self.meta.update(nid, content_hash=ch.content_hash, start_line=ch.start_line, end_line=ch.end_line, updated_t=now)
            ids[upd] = old_ids
            report.chunks_updated += len(upd)
            report.changed_symbols.extend(f"~{plans[i].rel_path}::{plans[i].chunk.symbol}" for i in upd)

        if new:
            f0, s = self._novelty_forces(pos[new])  # measured before insertion: vs prior memory only
            forces[new] = f0 + self._edit_impulse(pos[new])
            report.record_surprise(s)
            specs = [
                (f"chunk:{p.rel_path}::{p.chunk.symbol}", NodeKind.CHUNK, p.file_id, p.rel_path, p.chunk.symbol,
                 p.chunk.start_line, p.chunk.end_line, p.chunk.content_hash, None)
                for p in (plans[i] for i in new)
            ]
            ids[new] = self._create_nodes(specs, pos[new], ctx)
            report.chunks_added += len(new)
            report.changed_symbols.extend(f"+{plans[i].rel_path}::{plans[i].chunk.symbol}" for i in new)

        self.lexicon.upsert((int(ids[i]), p.chunk.symbol, p.rel_path, p.chunk.text) for i, p in enumerate(plans))
        self._apply_forces(ids, forces, now)
        by_file: dict[int, Tensor] = {}
        for i, p in enumerate(plans):
            by_file[p.file_id] = forces[i] if p.file_id not in by_file else by_file[p.file_id] + forces[i]
        for file_id, f in by_file.items():
            self._propagate(file_id, f, ctx)

    def _recompute_structural(self, ctx: _SyncContext) -> None:
        """FILE/DIR anchors = normalized mean direction of their children, at their level radius."""
        todo = [i for i in ctx.touched_structural if i not in ctx.killed_set and i != self._root_id]
        recs = self.meta.get_many(todo)
        order = sorted(recs.values(), key=lambda r: (-(len(PurePosixPath(r.path or "").parts)), -int(r.kind)))
        for rec in order:
            if rec.id in ctx.killed_set:
                continue
            children = self.meta.children(rec.id)
            if not children:
                if rec.kind == NodeKind.DIR:
                    self._kill([rec.id], ctx)
                    continue
                direction = self.embed_directions([rec.path or rec.key])[0]
            else:
                x = torch.from_numpy(self.positions([c.id for c in children]))
                norms = x.norm(dim=-1, keepdim=True)
                valid = norms[:, 0] > 1e-8
                mean = (x[valid] / norms[valid]).mean(0) if valid.any() else torch.zeros(self.config.rank)
                direction = mean / mean.norm() if mean.norm() > 1e-8 else self.embed_directions([rec.path or rec.key])[0]
            p = (direction * self.radius_of(rec.kind)).numpy().astype(np.float32)[None, :]
            self.arrays.position.array[rec.id] = p[0]
            self.index.upsert(np.array([rec.id], dtype=np.int64), p)

    # ======================================================================================
    # GREENFIELD: requirements, decisions, generated artifacts
    # ======================================================================================
    def _upsert_text_nodes(self, kind: NodeKind, items: Sequence[tuple[str, str]], parent_id: int, ctx: _SyncContext) -> list[int]:
        now = ctx.now
        out: list[int] = [0] * len(items)
        to_embed: list[int] = []
        existing: dict[int, NodeRecord] = {}
        for i, (key, text) in enumerate(items):
            rec = self.meta.get_by_key(key)
            if rec is not None and rec.kind != kind:
                raise ValueError(f"key {key!r} already used by a {rec.kind.name} node")
            if rec is not None and rec.content_hash == content_hash(text):
                out[i] = rec.id
                continue
            if rec is not None:
                existing[i] = rec
            to_embed.append(i)
        if not to_embed:
            return out
        kinds = np.full(len(to_embed), int(kind), dtype=np.int64)
        pos = self._place(self.embed_directions([items[i][1] for i in to_embed]), kinds)
        forces = torch.zeros((len(to_embed), self.config.rank), dtype=torch.float64)
        ids = np.empty(len(to_embed), dtype=np.int64)
        upd = [j for j, i in enumerate(to_embed) if i in existing]
        new = [j for j, i in enumerate(to_embed) if i not in existing]
        if upd:
            old_ids = np.array([existing[to_embed[j]].id for j in upd], dtype=np.int64)
            f0, _ = surprise_force(torch.from_numpy(self.positions(old_ids)), torch.from_numpy(pos[upd]), self._c)
            forces[upd] = f0 + self._edit_impulse(pos[upd])
            self.arrays.position.array[old_ids] = pos[upd]
            self.index.upsert(old_ids, pos[upd])
            for j, nid in zip(upd, old_ids.tolist()):
                text = items[to_embed[j]][1]
                self.meta.update(nid, payload=text, content_hash=content_hash(text), updated_t=now)
            ids[upd] = old_ids
        if new:
            f0, _ = self._novelty_forces(pos[new])
            forces[new] = f0 + self._edit_impulse(pos[new])
            specs = [
                (items[to_embed[j]][0], kind, parent_id, None, None, None, None,
                 content_hash(items[to_embed[j]][1]), items[to_embed[j]][1])
                for j in new
            ]
            ids[new] = self._create_nodes(specs, pos[new], ctx)
        self.lexicon.upsert((int(ids[j]), None, None, items[i][1]) for j, i in enumerate(to_embed))
        self._apply_forces(ids, forces, now)
        self._propagate(parent_id, forces.sum(0), ctx)
        for j, i in enumerate(to_embed):
            out[i] = int(ids[j])
        return out

    def _text_transaction(self, kind: NodeKind, items: Sequence[tuple[str, str]], edges: Sequence[tuple[int, str, float]] = ()) -> list[int]:
        with self._lock:
            if not self.projector.frozen:
                self.calibrate([t for _, t in items])
            now = self._tick()
            ctx = _SyncContext(now=now)
            try:
                with self.meta.transaction():
                    ids = self._upsert_text_nodes(kind, items, self._root_id, ctx)
                    if edges:
                        self.meta.add_edges(Edge(src, dst, ek, w) for src in ids for dst, ek, w in edges)
                    self._flush_ancestor_forces(ctx)
                    self.meta.set_meta("clock", str(self._clock))
            except BaseException:
                self._compensate(ctx)
                raise
            return ids

    def seed_requirements(self, requirements: Sequence[str | Requirement]) -> list[int]:
        """Project the user's architectural requirements into the ball (empty-repo bootstrap)."""
        items = []
        for r in requirements:
            req = r if isinstance(r, Requirement) else Requirement(text=r)
            if not req.text.strip():
                raise ValueError("empty requirement")
            items.append((f"spec:{req.key or content_hash(req.text)}", req.text))
        with self._lock:
            if self.meta.get_meta("project_mode") is None:
                has_code = self.meta.count_by_kind()[NodeKind.CHUNK] > 0
                self.meta.set_meta("project_mode", "brownfield" if has_code else "greenfield")
            return self._text_transaction(NodeKind.SPEC, items)

    def record_decision(self, text: str, *, key: str | None = None, related_ids: Sequence[int] = ()) -> int:
        edges = [(int(r), "relates", 1.0) for r in related_ids]
        return self._text_transaction(NodeKind.DECISION, [(f"decision:{key or content_hash(text)}", text)], edges)[0]

    def add_ephemeral(self, text: str, *, key: str | None = None) -> int:
        return self._text_transaction(NodeKind.EPHEMERAL, [(f"ephemeral:{key or content_hash(text)}", text)])[0]

    def register_generated_artifact(self, path: str | os.PathLike[str], spec_ids: Sequence[int]) -> IngestReport:
        """Ingest a file written by the compute engine and bind it to the specs it implements."""
        rel = self._normalize_rel(path)
        with self._lock:
            report = self.ingest_files([rel])
            file_rec = self.meta.get_by_key(f"file:{rel}")
            specs = self.meta.get_many(spec_ids)
            spec_list = [s for s in spec_ids if s in specs and specs[s].kind == NodeKind.SPEC]
            if file_rec is None or not spec_list:
                return report
            targets = [file_rec.id, *(c.id for c in self.meta.chunks_of_file(rel))]
            with self.meta.transaction():
                self.meta.add_edges(Edge(s, t, "implements", 1.0) for s in spec_list for t in targets)
            # Co-activation: implemented specs are pulled toward their artifact.
            self.reinforce(spec_list, self.positions([file_rec.id])[0], [1.0] * len(spec_list), scale=self.config.artifact_force_scale)
            return report

    def link(self, src: int, dst: int, kind: str, weight: float = 1.0) -> None:
        with self._lock, self.meta.transaction():
            self.meta.add_edges([Edge(int(src), int(dst), kind, float(weight))])

    def prune(self, kinds: frozenset[NodeKind] = frozenset({NodeKind.EPHEMERAL}), energy_floor: float = 0.05) -> int:
        """Delete cold nodes of the given (text) kinds. Code nodes are owned by the repository sync."""
        if kinds & {NodeKind.ROOT, NodeKind.DIR, NodeKind.FILE, NodeKind.CHUNK}:
            raise ValueError("prune only text kinds; code nodes follow the repository")
        with self._lock:
            ids = np.array(self.meta.ids_by_kinds(kinds), dtype=np.int64)
            if ids.size == 0:
                return 0
            cold = ids[self.energies(ids) < energy_floor]
            ctx = _SyncContext(now=self._clock)
            with self.meta.transaction():
                self._kill(cold.tolist(), ctx)
            return int(cold.size)

    # ======================================================================================
    # retrieval primitives (used by the router)
    # ======================================================================================
    def search(self, positions: np.ndarray | Tensor, k: int, kinds: frozenset[NodeKind] | None = None) -> tuple[np.ndarray, np.ndarray]:
        q = positions if isinstance(positions, Tensor) else torch.from_numpy(np.asarray(positions, dtype=np.float32))
        ids, d = self.index.search(q.reshape(-1, self.config.rank), k, kinds)
        return ids.numpy(), d.numpy()

    def read_lines(self, rel: str, start: int, end: int) -> str | None:
        text = self._read_source(rel)
        return None if text is None else slice_lines(text.splitlines(), start, end)

    def read_chunk(self, rec: NodeRecord) -> str | None:
        """Exact source of a chunk, or None if the file changed since ingestion (hash mismatch)."""
        if rec.kind != NodeKind.CHUNK or rec.path is None or rec.start_line is None or rec.end_line is None:
            return None
        text = self.read_lines(rec.path, rec.start_line, rec.end_line)
        return text if text is not None and content_hash(text) == rec.content_hash else None

    def read_file_header(self, rel: str, max_lines: int) -> tuple[str, int] | None:
        """The file's `<header>` chunk (imports/constants before the first symbol), capped at max_lines."""
        text = self._read_source(rel)
        if text is None:
            return None
        header = next((c for c in self._chunk(rel, text) if c.symbol == "<header>"), None)
        if header is None or header.start_line > max_lines:
            return None
        end = min(header.end_line, max_lines)
        return slice_lines(text.splitlines(), 1, end), end

    def lexical_search(self, query: str | LexicalQuery, k: int, kinds: frozenset[NodeKind] | None = None) -> list[LexicalHit]:
        """BM25 over code-aware tokens (exact identifiers, sub-tokens, error codes, quoted phrases)."""
        with self._lock:
            return self.lexicon.search(query, k, kinds)

    def rebuild_lexicon(self) -> int:
        """(Re)build the FTS5 index from the repository (chunks, file paths) and stored text payloads."""
        with self._lock, self.meta.transaction():
            self.lexicon.create()
            rows: list[tuple[int, str | None, str | None, str | None]] = []
            stale = 0
            cached_path: str | None = None
            cached_lines: list[str] | None = None
            for kind in (NodeKind.FILE, NodeKind.CHUNK, NodeKind.SPEC, NodeKind.DECISION, NodeKind.EPHEMERAL):
                for rec in self.meta.by_kind(kind):
                    if kind == NodeKind.FILE:
                        rows.append((rec.id, None, rec.path, None))
                    elif kind == NodeKind.CHUNK:
                        if rec.path != cached_path:
                            cached_path, source = rec.path, self._read_source(rec.path or "")
                            cached_lines = source.splitlines() if source is not None else None
                        text = None
                        if cached_lines is not None and rec.start_line and rec.end_line:
                            text = slice_lines(cached_lines, rec.start_line, rec.end_line)
                            text = text if content_hash(text) == rec.content_hash else None
                        stale += text is None  # drifted file: indexed again at its next sync
                        rows.append((rec.id, rec.symbol, rec.path, text))
                    else:
                        rows.append((rec.id, None, None, rec.payload))
                    if len(rows) >= 2048:
                        self.lexicon.upsert(rows)
                        rows = []
            self.lexicon.upsert(rows)
            self.meta.set_meta("lexicon", LEXICON_VERSION)
            return stale

    # ======================================================================================
    # lifecycle
    # ======================================================================================
    def flush(self) -> None:
        with self._lock:
            self.ws.flush()
            self.arrays.flush()
            self.index.save()
            self.meta.set_meta("clock", str(self._clock))
            self.meta.checkpoint()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            try:
                self.flush()
                self.arrays.close()
                self.meta.close()
            finally:
                self._closed = True
                self._store_lock.release()

    def __enter__(self) -> "NewtonianMemoryManager":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def stats(self) -> MemoryStats:
        with self._lock:
            counts = self.meta.count_by_kind()
            disk = self.arrays.disk_bytes() + sum(
                p.stat().st_size for p in self.config.storage_dir.glob("meta.sqlite3*") if p.is_file()
            )
            return MemoryStats(
                clock=self._clock,
                nodes_by_kind={k.name: v for k, v in counts.items()},
                working_set=self.ws.stats(),
                disk_bytes=disk,
                index_backend=self._index_backend,
                projector_fingerprint=self.projector.fingerprint,
            )
