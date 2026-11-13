"""Polling watcher with debounce: syncs after the tree has been quiet for `debounce_s`.

Polling (stat of indexable files) is portable and cheap thanks to the store's stat cache; it needs no
inotify/FSEvents dependency. The store is opened only for the duration of each sync, so git hooks and
CLI calls can interleave with a running watcher (they serialize on the store lock).
"""
from __future__ import annotations

import hashlib
import os
import time
from typing import Callable

from .config import MemoryConfig
from .discovery import discover
from .manager import NewtonianMemoryManager, SyncReport


def tree_signature(config: MemoryConfig) -> str:
    h = hashlib.blake2b(digest_size=16)
    for rel in discover(config):
        try:
            st = os.stat(config.repo_root / rel)
        except OSError:
            continue
        h.update(f"{rel}\0{st.st_size}\0{st.st_mtime_ns}\n".encode("utf-8", "surrogateescape"))
    return h.hexdigest()


class RepoWatcher:
    def __init__(
        self,
        config: MemoryConfig,
        open_manager: Callable[[], NewtonianMemoryManager],
        *,
        interval_s: float = 2.0,
        debounce_s: float = 1.5,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        on_sync: Callable[[SyncReport], None] | None = None,
    ) -> None:
        self._config = config
        self._open = open_manager
        self.interval_s = interval_s
        self.debounce_s = debounce_s
        self._clock = clock
        self._sleep = sleep
        self._on_sync = on_sync
        self._last_sig: str | None = None
        self._changed_at = 0.0
        self._dirty = True  # first poll syncs: the tree may have changed while nobody watched

    def sync_now(self) -> SyncReport:
        with self._open() as mgr:
            report = mgr.sync()
        if self._on_sync:
            self._on_sync(report)
        return report

    def poll_once(self) -> SyncReport | None:
        sig = tree_signature(self._config)
        now = self._clock()
        if sig != self._last_sig:
            first = self._last_sig is None
            self._last_sig = sig
            if not first:
                self._changed_at = now
                self._dirty = True
                return None  # still changing: wait for quiet
        if self._dirty and now - self._changed_at >= self.debounce_s:
            self._dirty = False
            return self.sync_now()
        return None

    def run(self, *, max_polls: int | None = None) -> None:
        polls = 0
        while max_polls is None or polls < max_polls:
            self.poll_once()
            polls += 1
            if max_polls is None or polls < max_polls:
                self._sleep(self.interval_s)
