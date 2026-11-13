"""Portable process memory introspection (the `resource` module does not exist on Windows)."""
from __future__ import annotations

import sys


def current_rss_mib() -> float | None:
    try:
        import psutil

        return psutil.Process().memory_info().rss / 2**20
    except ImportError:
        return None


def peak_rss_mib() -> float | None:
    """Peak resident set size of this process in MiB, or None if it cannot be measured."""
    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak / 2**20 if sys.platform == "darwin" else peak / 1024  # macOS: bytes, Linux: KiB
    except ImportError:
        pass
    try:
        import psutil

        info = psutil.Process().memory_info()
        return getattr(info, "peak_wset", info.rss) / 2**20  # Windows: peak working set
    except ImportError:
        return None
