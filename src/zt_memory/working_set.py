"""RAM working set: hot dynamic state (velocity, energy, last_t) with energy-based paging.

Fixed-capacity numpy slabs sized from a byte budget; no per-node Python objects beyond
one dict entry. Eviction removes the residents with the lowest *current* (lazily
decayed) energy and writes dirty rows back to the SSD memmaps.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch

from .dynamics import current_energy
from .storage import NodeArrays

try:  # optional: system memory pressure monitoring
    import psutil
except ImportError:  # pragma: no cover
    psutil = None  # type: ignore[assignment]

_DICT_ENTRY_OVERHEAD = 104  # bytes: dict slot + two boxed ints (CPython, approx.)


@dataclass(frozen=True)
class WorkingSetStats:
    capacity: int
    resident: int
    dirty: int
    hits: int
    misses: int
    evictions: int
    writebacks: int
    pressure_evictions: int
    budget_bytes: int


class WorkingSet:
    def __init__(
        self,
        arrays: NodeArrays,
        rank: int,
        budget_bytes: int,
        beta_by_kind: np.ndarray,
        gamma_by_kind: np.ndarray,
        *,
        eviction_fraction: float = 0.1,
        ram_floor_bytes: int = 0,
        pressure_check_interval_s: float = 1.0,
    ) -> None:
        per_entry = rank * 4 + 4 + 8 + 8 + 1 + 1 + _DICT_ENTRY_OVERHEAD
        capacity = int(budget_bytes // per_entry)
        if capacity < 16:
            raise ValueError(f"RAM budget {budget_bytes} B too small (need >= {16 * per_entry} B)")
        self.capacity = capacity
        self.budget_bytes = budget_bytes
        self._arrays = arrays
        self._beta = beta_by_kind
        self._gamma = gamma_by_kind
        self._eviction_fraction = eviction_fraction
        self._ram_floor = ram_floor_bytes
        self._pressure_interval = pressure_check_interval_s
        self._last_pressure_check = 0.0
        self._ids = np.full(capacity, -1, dtype=np.int64)
        self._vel = np.zeros((capacity, rank), dtype=np.float32)
        self._energy = np.zeros(capacity, dtype=np.float32)
        self._t = np.zeros(capacity, dtype=np.int64)
        self._kind = np.zeros(capacity, dtype=np.uint8)
        self._dirty = np.zeros(capacity, dtype=bool)
        self._slot_of: dict[int, int] = {}
        self._free: list[int] = list(range(capacity - 1, -1, -1))
        self._hits = self._misses = self._evictions = self._writebacks = self._pressure_evictions = 0

    @property
    def max_batch(self) -> int:
        return max(1, self.capacity // 2)

    def __contains__(self, node_id: int) -> bool:
        return int(node_id) in self._slot_of

    def __len__(self) -> int:
        return len(self._slot_of)

    # -- paging -------------------------------------------------------------------------------
    def fetch(self, ids: np.ndarray, now: int) -> np.ndarray:
        """Make `ids` (unique, len <= max_batch) resident; return their slots."""
        ids = np.asarray(ids, dtype=np.int64)
        if ids.size > self.max_batch:
            raise ValueError(f"batch of {ids.size} exceeds max_batch={self.max_batch}")
        self._relieve_pressure(now)
        slots = np.empty(ids.size, dtype=np.int64)
        missing_pos: list[int] = []
        for j, nid in enumerate(ids.tolist()):
            s = self._slot_of.get(nid)
            if s is None:
                missing_pos.append(j)
            else:
                slots[j] = s
        self._hits += ids.size - len(missing_pos)
        self._misses += len(missing_pos)
        if missing_pos:
            if len(missing_pos) > len(self._free):
                missing_set = set(missing_pos)
                protect = slots[[j for j in range(ids.size) if j not in missing_set]]
                n_evict = max(len(missing_pos) - len(self._free), int(self._eviction_fraction * self.capacity))
                self.evict(n_evict, now, protect=protect)
            miss_ids = ids[missing_pos]
            new_slots = np.array([self._free.pop() for _ in missing_pos], dtype=np.int64)
            self._ids[new_slots] = miss_ids
            self._vel[new_slots] = self._arrays.velocity.array[miss_ids]
            self._energy[new_slots] = self._arrays.energy.array[miss_ids]
            self._t[new_slots] = self._arrays.last_t.array[miss_ids]
            self._kind[new_slots] = self._arrays.kind.array[miss_ids]
            self._dirty[new_slots] = False
            for nid, s in zip(miss_ids.tolist(), new_slots.tolist()):
                self._slot_of[nid] = s
            slots[missing_pos] = new_slots
        return slots

    def read(self, slots: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
        return (
            torch.from_numpy(self._vel[slots].astype(np.float64)),
            torch.from_numpy(self._energy[slots].astype(np.float64)),
            torch.from_numpy(self._t[slots].copy()),
            self._kind[slots].copy(),
        )

    def write(self, slots: np.ndarray, velocity: torch.Tensor, energy: torch.Tensor, now: int) -> None:
        self._vel[slots] = velocity.detach().cpu().numpy().astype(np.float32)
        self._energy[slots] = energy.detach().cpu().numpy().astype(np.float32)
        self._t[slots] = now
        self._dirty[slots] = True

    def peek(self, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """State of `ids` without promoting them (resident rows override SSD rows)."""
        ids = np.asarray(ids, dtype=np.int64)
        vel = np.array(self._arrays.velocity.array[ids], dtype=np.float32)
        energy = np.array(self._arrays.energy.array[ids], dtype=np.float32)
        t = np.array(self._arrays.last_t.array[ids], dtype=np.int64)
        kind = np.array(self._arrays.kind.array[ids], dtype=np.uint8)
        for j, nid in enumerate(ids.tolist()):
            s = self._slot_of.get(nid)
            if s is not None:
                vel[j], energy[j], t[j], kind[j] = self._vel[s], self._energy[s], self._t[s], self._kind[s]
        return vel, energy, t, kind

    def evict(self, n: int, now: int, *, protect: np.ndarray | None = None) -> int:
        resident = np.flatnonzero(self._ids >= 0)
        if protect is not None and protect.size:
            resident = np.setdiff1d(resident, protect, assume_unique=False)
        n = min(n, resident.size)
        if n <= 0:
            return 0
        kinds = self._kind[resident]
        e = current_energy(
            torch.from_numpy(self._vel[resident].astype(np.float64)),
            torch.from_numpy(self._energy[resident].astype(np.float64)),
            torch.from_numpy(self._t[resident]),
            now,
            torch.from_numpy(self._beta[kinds]),
            torch.from_numpy(self._gamma[kinds]),
        ).numpy()
        victims = resident[np.argpartition(e, n - 1)[:n]] if n < resident.size else resident
        self._writeback(victims[self._dirty[victims]])
        self._release(victims)
        self._evictions += victims.size
        return int(victims.size)

    def drop(self, ids: np.ndarray) -> None:
        """Forget rows of deleted nodes without write-back."""
        slots = [self._slot_of[i] for i in np.asarray(ids, dtype=np.int64).tolist() if i in self._slot_of]
        if slots:
            self._release(np.array(slots, dtype=np.int64))

    def flush(self) -> None:
        dirty = np.flatnonzero(self._dirty & (self._ids >= 0))
        self._writeback(dirty)

    def clear(self) -> None:
        self.flush()
        self._release(np.flatnonzero(self._ids >= 0))

    def _writeback(self, slots: np.ndarray) -> None:
        if slots.size == 0:
            return
        ids = self._ids[slots]
        self._arrays.velocity.array[ids] = self._vel[slots]
        self._arrays.energy.array[ids] = self._energy[slots]
        self._arrays.last_t.array[ids] = self._t[slots]
        self._dirty[slots] = False
        self._writebacks += slots.size

    def _release(self, slots: np.ndarray) -> None:
        for nid in self._ids[slots].tolist():
            self._slot_of.pop(nid, None)
        self._ids[slots] = -1
        self._dirty[slots] = False
        self._free.extend(slots.tolist())

    def _relieve_pressure(self, now: int) -> None:
        if psutil is None or self._ram_floor <= 0:
            return
        t = time.monotonic()
        if t - self._last_pressure_check < self._pressure_interval:
            return
        self._last_pressure_check = t
        if psutil.virtual_memory().available < self._ram_floor and self._slot_of:
            n = max(1, int(self._eviction_fraction * len(self._slot_of)))
            self._pressure_evictions += self.evict(n, now)

    def stats(self) -> WorkingSetStats:
        return WorkingSetStats(
            capacity=self.capacity,
            resident=len(self._slot_of),
            dirty=int(self._dirty.sum()),
            hits=self._hits,
            misses=self._misses,
            evictions=self._evictions,
            writebacks=self._writebacks,
            pressure_evictions=self._pressure_evictions,
            budget_bytes=self.budget_bytes,
        )
