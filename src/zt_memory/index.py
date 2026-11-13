"""Vector indices over the SSD node arrays.

BlockScanIndex: exact Poincaré k-NN by streaming the position memmap in fixed blocks.
    RAM = O(block_rows * rank); ~1M nodes x rank 64 is ~256 MB of reads per query,
    served by the page cache. Default for repository-scale stores.
HnswIndex: approximate candidates with hnswlib on tangent-space (log_0) vectors, then
    exact hyperbolic re-ranking. The HNSW graph is RAM-resident (~(4r + 8M + 64) B/node).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Protocol

import numpy as np
import torch
from torch import Tensor

from . import hyperbolic as H
from .config import NodeKind
from .storage import NodeArrays


class VectorIndex(Protocol):
    def search(self, queries: Tensor, k: int, kinds: frozenset[NodeKind] | None = None) -> tuple[Tensor, Tensor]:
        """Return (ids [m, k] int64 padded with -1, distances [m, k] float32 padded with +inf)."""
        ...

    def upsert(self, ids: np.ndarray, positions: np.ndarray) -> None: ...

    def remove(self, ids: np.ndarray) -> None: ...

    def save(self) -> None: ...


def kind_mask(kinds: frozenset[NodeKind] | None) -> np.ndarray:
    mask = np.zeros(256, dtype=bool)
    if kinds is None:
        mask[[int(k) for k in NodeKind]] = True
    else:
        mask[[int(k) for k in kinds]] = True
    return mask


def _pad(ids: Tensor, dists: Tensor, k: int) -> tuple[Tensor, Tensor]:
    m, have = ids.shape
    if have >= k:
        return ids[:, :k], dists[:, :k]
    pid = torch.full((m, k - have), -1, dtype=torch.int64)
    pd = torch.full((m, k - have), float("inf"), dtype=torch.float32)
    return torch.cat([ids, pid], 1), torch.cat([dists, pd], 1)


class BlockScanIndex:
    def __init__(self, arrays: NodeArrays, high_water: Callable[[], int], curvature: float, block_rows: int = 65_536) -> None:
        self._arrays = arrays
        self._high_water = high_water
        self._c = curvature
        self._block = block_rows

    def search(self, queries: Tensor, k: int, kinds: frozenset[NodeKind] | None = None) -> tuple[Tensor, Tensor]:
        q = queries.to(torch.float32).reshape(-1, queries.shape[-1])
        m = q.shape[0]
        ok = kind_mask(kinds)
        best_d = torch.empty((m, 0), dtype=torch.float32)
        best_i = torch.empty((m, 0), dtype=torch.int64)
        hw = min(self._high_water(), self._arrays.capacity)
        for start in range(0, hw, self._block):
            end = min(start + self._block, hw)
            mask = (self._arrays.alive.array[start:end] == 1) & ok[self._arrays.kind.array[start:end]]
            local = np.flatnonzero(mask)
            if local.size == 0:
                continue
            x = torch.from_numpy(np.array(self._arrays.position.array[start:end][local], dtype=np.float32))
            d = H.pairwise_distance(q, x, self._c)
            ids = torch.from_numpy(local.astype(np.int64) + start).expand(m, -1)
            cat_d = torch.cat([best_d, d], 1)
            cat_i = torch.cat([best_i, ids], 1)
            kk = min(k, cat_d.shape[1])
            best_d, pos = torch.topk(cat_d, kk, dim=1, largest=False)
            best_i = torch.gather(cat_i, 1, pos)
        return _pad(best_i, best_d, k)

    def upsert(self, ids: np.ndarray, positions: np.ndarray) -> None:  # reads the memmap directly
        return None

    def remove(self, ids: np.ndarray) -> None:
        return None

    def save(self) -> None:
        return None


class HnswIndex:
    def __init__(
        self,
        path: Path,
        arrays: NodeArrays,
        high_water: Callable[[], int],
        curvature: float,
        rank: int,
        *,
        m: int = 16,
        ef_construction: int = 200,
        ef_search: int = 128,
        oversample: int = 4,
    ) -> None:
        try:
            import hnswlib
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("HnswIndex requires `pip install hnswlib`") from exc
        self._hnswlib = hnswlib
        self._path = Path(path)
        self._sidecar = self._path.with_suffix(".json")
        self._arrays = arrays
        self._high_water = high_water
        self._c = curvature
        self._rank = rank
        self._m = m
        self._efc = ef_construction
        self._ef = ef_search
        self._oversample = oversample
        self._index = hnswlib.Index(space="l2", dim=rank)
        if self._path.exists() and self._sidecar_matches():
            self._index.load_index(str(self._path), max_elements=max(1024, self._arrays.capacity))
        else:
            self._rebuild()
        self._index.set_ef(self._ef)

    def _alive_ids(self) -> np.ndarray:
        hw = min(self._high_water(), self._arrays.capacity)
        return np.flatnonzero(self._arrays.alive.array[:hw] == 1).astype(np.int64)

    def _sidecar_matches(self) -> bool:
        try:
            meta = json.loads(self._sidecar.read_text())
        except (OSError, ValueError):
            return False
        return meta.get("alive") == int(self._alive_ids().size) and meta.get("high_water") == self._high_water()

    def _rebuild(self) -> None:
        ids = self._alive_ids()
        self._index = self._hnswlib.Index(space="l2", dim=self._rank)
        self._index.init_index(max_elements=max(1024, 2 * self._arrays.capacity), M=self._m, ef_construction=self._efc)
        if ids.size:
            self._add(ids, np.array(self._arrays.position.array[ids], dtype=np.float32))

    def _tangent(self, positions: np.ndarray | Tensor) -> np.ndarray:
        t = positions if isinstance(positions, Tensor) else torch.from_numpy(np.asarray(positions, dtype=np.float32))
        return H.logmap0(t.to(torch.float32), self._c).numpy()

    def _add(self, ids: np.ndarray, positions: np.ndarray) -> None:
        need = int(ids.max()) + 1
        if need > self._index.get_max_elements() or self._index.get_current_count() + ids.size > self._index.get_max_elements():
            self._index.resize_index(max(2 * self._index.get_max_elements(), need, self._index.get_current_count() + ids.size))
        self._index.add_items(self._tangent(positions), ids)

    def upsert(self, ids: np.ndarray, positions: np.ndarray) -> None:
        ids = np.asarray(ids, dtype=np.int64)
        if ids.size:
            self._add(ids, positions)

    def remove(self, ids: np.ndarray) -> None:
        for i in np.asarray(ids, dtype=np.int64):
            try:
                self._index.mark_deleted(int(i))
            except RuntimeError:  # unknown or already deleted label
                pass

    def search(self, queries: Tensor, k: int, kinds: frozenset[NodeKind] | None = None) -> tuple[Tensor, Tensor]:
        q = queries.to(torch.float32).reshape(-1, queries.shape[-1])
        ok = kind_mask(kinds)
        kind_arr = self._arrays.kind.array
        alive = self._arrays.alive.array

        def admit(label: int) -> bool:
            return bool(alive[label] == 1 and ok[kind_arr[label]])

        vecs = self._tangent(q)
        out_i, out_d = [], []
        count = self._index.get_current_count()
        for row in range(q.shape[0]):
            kq = min(k * self._oversample, count)
            labels = np.empty((0,), dtype=np.int64)
            while kq > 0:
                try:
                    self._index.set_ef(max(self._ef, kq))
                    lab, _ = self._index.knn_query(vecs[row : row + 1], k=kq, filter=admit)
                    labels = lab[0].astype(np.int64)
                    break
                except RuntimeError:  # fewer admissible elements than kq
                    kq //= 2
            if labels.size == 0:
                out_i.append(torch.empty((1, 0), dtype=torch.int64))
                out_d.append(torch.empty((1, 0), dtype=torch.float32))
                continue
            x = torch.from_numpy(np.array(self._arrays.position.array[labels], dtype=np.float32))
            d = H.pairwise_distance(q[row : row + 1], x, self._c)
            kk = min(k, d.shape[1])
            dd, pos = torch.topk(d, kk, dim=1, largest=False)
            out_i.append(torch.from_numpy(labels)[pos[0]][None, :])
            out_d.append(dd)
        padded = [_pad(i, d, k) for i, d in zip(out_i, out_d)]
        if not padded:
            return _pad(torch.empty((0, 0), dtype=torch.int64), torch.empty((0, 0)), k)
        return torch.cat([p[0] for p in padded], 0), torch.cat([p[1] for p in padded], 0)

    def save(self) -> None:
        self._index.save_index(str(self._path))
        self._sidecar.write_text(json.dumps({"alive": int(self._alive_ids().size), "high_water": self._high_water()}))
