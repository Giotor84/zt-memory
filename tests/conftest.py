from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Callable

import pytest

from zt_memory import HashingEncoder, MemoryConfig, NewtonianMemoryManager

SAMPLE_REPO: dict[str, str] = {
    "src/app/config.py": '''
        import json
        from pathlib import Path

        DEFAULT_TIMEOUT = 30


        def parse_json_config(path: Path) -> dict:
            """Parse a JSON configuration file from disk."""
            with open(path) as fh:
                return json.load(fh)


        def validate_config(cfg: dict) -> bool:
            """Validate that the configuration declares a database url."""
            return "database_url" in cfg and cfg["database_url"].startswith("postgres")
    ''',
    "src/app/db.py": '''
        class DatabasePool:
            """Connection pool for the relational database."""

            def __init__(self, url: str, size: int = 4) -> None:
                self.url = url
                self.size = size
                self._free = []

            def acquire_connection(self):
                """Acquire a pooled database connection."""
                return self._free.pop() if self._free else object()

            def release_connection(self, conn) -> None:
                """Release a database connection back to the pool."""
                self._free.append(conn)
    ''',
    "src/app/http_server.py": '''
        from .config import parse_json_config


        def handle_request(method: str, route: str) -> str:
            """Dispatch an HTTP request to a route handler."""
            if method == "GET" and route == "/health":
                return "ok"
            return render_template("not_found.html")


        def render_template(name: str) -> str:
            """Render an HTML template by name."""
            return f"<html>{name}</html>"
    ''',
    "src/util/strings.py": '''
        import re


        def slugify_title(title: str) -> str:
            """Turn a title into a url slug."""
            return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


        def camel_to_snake(name: str) -> str:
            """Convert CamelCase identifiers to snake_case."""
            return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()
    ''',
    "README.md": "# Sample service\n\nHTTP service with a database pool and JSON configuration.\n",
}


def write_repo(root: Path, files: dict[str, str]) -> None:
    for rel, body in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(textwrap.dedent(body).lstrip("\n"), encoding="utf-8")


@pytest.fixture(autouse=True)
def _isolate_zt_discovery(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZT_DISCOVERY_CEILING", str(tmp_path))


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    write_repo(root, SAMPLE_REPO)
    return root


@pytest.fixture
def make_manager(tmp_path: Path) -> Callable[..., NewtonianMemoryManager]:
    created: list[NewtonianMemoryManager] = []

    def factory(repo_root: Path | None = None, **overrides) -> NewtonianMemoryManager:
        root = repo_root or (tmp_path / "repo")
        root.mkdir(parents=True, exist_ok=True)
        params = dict(
            storage_dir=tmp_path / "store",
            repo_root=root,
            rank=32,
            calibration_samples=128,
            ram_floor_bytes=0,
            ram_budget_bytes=8 * 2**20,
            lock_timeout_s=0.2,
        )
        params.update(overrides)
        encoder = params.pop("encoder", None) or HashingEncoder(dim=512)
        mgr = NewtonianMemoryManager(MemoryConfig(**params), encoder)
        created.append(mgr)
        return mgr

    yield factory
    for mgr in created:
        try:
            mgr.close()
        except Exception:
            pass
