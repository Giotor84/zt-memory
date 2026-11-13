"""Per-repository configuration: `<repo>/.zt/config.toml`.

    [store]    path = ".zt/store"                       # relative to the repo root, or absolute
    [encoder]  spec = "hf:Qwen/Qwen3-Embedding-0.6B"   # or "hashing[:dim]"; + pooling, dtype, max_length, ...
    [memory]   rank = 128, ram_budget_mb = 1536, ...   # any scalar MemoryConfig field
    [router]   token_budget = 12000, ...               # any scalar RouterConfig field
    [engine]   url = "http://127.0.0.1:8080/v1", max_new_tokens = 3072, temperature = 0.1, ...
"""
from __future__ import annotations

import dataclasses
import json
import os
import shutil
import subprocess
import tomllib
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import REMOVED_ROUTER_KEYS, RENAMED_ROUTER_KEYS, MemoryConfig, RouterConfig
from .encoders import HiddenStateEncoder, build_encoder
from .manager import NewtonianMemoryManager

CONFIG_DIR = ".zt"
CONFIG_FILE = "config.toml"
_SCALARS = (bool, int, float, str)
_MEMORY_FIELDS = {f.name for f in dataclasses.fields(MemoryConfig)} - {"storage_dir", "repo_root", "physics",
                                                                       "code_extensions", "code_filenames", "excluded_dirs"}
_ROUTER_FIELDS = {f.name for f in dataclasses.fields(RouterConfig)} - {"brownfield_kind_weights", "greenfield_kind_weights"}
_ENGINE_FIELDS = {"backend", "url", "model", "max_new_tokens", "temperature", "timeout_s", "ctx", "retries",
                  "region", "proxy", "max_calls", "aws_retries", "need_rounds"}
# `retries` = how many times zt code re-asks the model with feedback when an edit does not apply (each one is a full
# call); `aws_retries` = botocore attempts on throttling/transient errors (Bedrock only).
ENGINE_DEFAULTS: dict[str, Any] = {"backend": "openai", "url": "http://127.0.0.1:8080/v1", "model": "local",
                                   "max_new_tokens": 3072, "temperature": 0.1, "timeout_s": 900.0, "ctx": 16384,
                                   "retries": 1, "region": "eu-central-1", "proxy": "", "max_calls": 20, "aws_retries": 3,
                                   # NEED protocol (zt code): -1 = 1 extra round with [router] preset = "api", else 0
                                   "need_rounds": -1}
# Bedrock models declare their window in the model profile (no /props endpoint) and reasoning models need room to
# think before the answer: unless set in [engine], ctx comes from the profile and max_new_tokens is 8192.
BEDROCK_DEFAULTS: dict[str, Any] = {"ctx": None, "max_new_tokens": 8192}
# Router presets for `[router] preset`; explicit [router] keys override them.
#   api    - models that can ask for more code (>=120B via API): anti-clone, bodies for the first 5 production chunks
#            (and those >= 0.6 x best), up to 20 more signatures in the map; with a traceback, 10 bodies and no map.
#            Pairs with the NEED protocol of `zt code` (one extra round when the model asks for a body).
#   single - one call, no second round: bodies until 85% of the score mass (at least 3); with a traceback, 15 bodies.
ROUTER_PRESETS: dict[str, dict[str, Any]] = {
    "api": {"dedup": True, "diversity_lambda": 0.3, "focus_blocks": 5, "focus_ratio": 0.6, "signature_extra": 20,
            "traceback_focus_blocks": 10},
    "single": {"dedup": True, "diversity_lambda": 0.3, "focus_blocks": 3, "focus_ratio": 0.6, "focus_mass": 0.85,
               "traceback_focus_blocks": 15},
}
_ENCODER_FIELDS = {"spec", "layer", "pooling", "device", "dtype", "max_length", "batch_size", "query_prefix", "padding_side", "dim"}


class ProjectConfigError(ValueError):
    pass


def _check(section: str, values: dict[str, Any], allowed: set[str]) -> None:
    unknown = set(values) - allowed
    if unknown:
        raise ProjectConfigError(f"[{section}] unknown keys {sorted(unknown)}; allowed: {sorted(allowed)}")
    bad = [k for k, v in values.items() if not isinstance(v, _SCALARS)]
    if bad:
        raise ProjectConfigError(f"[{section}] keys {bad} must be scalars")


def _toml_value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    return json.dumps(str(v))  # JSON string escapes are valid TOML basic-string escapes


@dataclass
class ProjectConfig:
    repo_root: Path
    store: Path
    encoder: dict[str, Any] = field(default_factory=lambda: {"spec": "hashing"})
    memory: dict[str, Any] = field(default_factory=dict)
    router: dict[str, Any] = field(default_factory=dict)
    engine: dict[str, Any] = field(default_factory=dict)

    @property
    def path(self) -> Path:
        return self.repo_root / CONFIG_DIR / CONFIG_FILE

    # -- io -----------------------------------------------------------------------------------
    @classmethod
    def load(cls, repo_root: Path) -> "ProjectConfig":
        root = Path(repo_root).resolve()
        path = root / CONFIG_DIR / CONFIG_FILE
        if not path.is_file():
            raise ProjectConfigError(f"{path} not found: run `zt init` in the repository first")
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        unknown = set(data) - {"store", "encoder", "memory", "router", "engine"}
        if unknown:
            raise ProjectConfigError(f"unknown sections {sorted(unknown)}")
        enc = dict(data.get("encoder", {}))
        mem = dict(data.get("memory", {}))
        rtr = dict(data.get("router", {}))
        eng = dict(data.get("engine", {}))
        _check("engine", eng, _ENGINE_FIELDS)
        backend = eng.get("backend", ENGINE_DEFAULTS["backend"])
        if backend not in ("openai", "bedrock"):
            raise ProjectConfigError(f"[engine] backend {backend!r} is not supported; choose \"openai\" or \"bedrock\"")
        if "need_rounds" in eng and (not isinstance(eng["need_rounds"], int) or isinstance(eng["need_rounds"], bool)
                                     or eng["need_rounds"] < -1):
            raise ProjectConfigError("[engine] need_rounds must be -1 (auto) or a non-negative integer")
        for key in ("max_calls", "aws_retries"):
            if key in eng and (not isinstance(eng[key], int) or isinstance(eng[key], bool) or eng[key] < 0):
                raise ProjectConfigError(f"[engine] {key} must be a non-negative integer")
        _check("encoder", enc, _ENCODER_FIELDS)
        _check("memory", mem, _MEMORY_FIELDS | {"ram_budget_mb"})
        _check("router", rtr, _ROUTER_FIELDS | REMOVED_ROUTER_KEYS | set(RENAMED_ROUTER_KEYS) | {"preset"})
        if rtr.get("preset", "") not in ("", *ROUTER_PRESETS):
            raise ProjectConfigError(f"[router] preset {rtr['preset']!r} is unknown; choose from {sorted(ROUTER_PRESETS)}")
        if "spec" not in enc:
            raise ProjectConfigError("[encoder] spec is required")
        store = Path(data.get("store", {}).get("path", f"{CONFIG_DIR}/store")).expanduser()
        return cls(root, store if store.is_absolute() else root / store, enc, mem, rtr, eng)

    @classmethod
    def find(cls, start: Path | None = None) -> "ProjectConfig":
        """Nearest ancestor of `start` (default: cwd) holding .zt/config.toml."""
        here = Path(start or Path.cwd()).resolve()
        ceiling_env = os.environ.get("ZT_DISCOVERY_CEILING")
        ceiling = Path(ceiling_env).resolve() if ceiling_env else None
        for candidate in (here, *here.parents):
            # Never auto-discover a filesystem root (.zt at / or C:\) from a subdirectory
            if candidate == candidate.parent and candidate != here:
                break
            if (candidate / CONFIG_DIR / CONFIG_FILE).is_file():
                return cls.load(candidate)
            if ceiling is not None and candidate == ceiling:
                break
        raise ProjectConfigError(f"no {CONFIG_DIR}/{CONFIG_FILE} in {here} or its parents: run `zt init`")

    def save(self, *, overwrite: bool = False) -> Path:
        if self.path.exists() and not overwrite:
            raise ProjectConfigError(f"{self.path} exists (use --force to overwrite)")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            store_value = str(self.store.relative_to(self.repo_root))
        except ValueError:
            store_value = str(self.store)
        lines = ["# zt-memory project configuration", "", "[store]", f"path = {_toml_value(store_value)}"]
        for name, table in (("encoder", self.encoder), ("memory", self.memory), ("router", self.router), ("engine", self.engine)):
            lines += ["", f"[{name}]"] + [f"{k} = {_toml_value(v)}" for k, v in table.items()]
        self.path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        tomllib.loads(self.path.read_text(encoding="utf-8"))  # round-trip guard
        return self.path

    # -- builders -----------------------------------------------------------------------------
    def memory_config(self) -> MemoryConfig:
        mem = dict(self.memory)
        if "ram_budget_mb" in mem:
            mem["ram_budget_bytes"] = int(mem.pop("ram_budget_mb")) * 2**20
        return MemoryConfig(storage_dir=self.store, repo_root=self.repo_root, **mem)

    @property
    def router_preset(self) -> str:
        return str(self.router.get("preset", ""))

    def router_config(self) -> RouterConfig:
        opts: dict[str, Any] = dict(ROUTER_PRESETS.get(self.router_preset, {}))
        for key, value in self.router.items():
            if key == "preset":
                continue
            if key in REMOVED_ROUTER_KEYS:
                warnings.warn(f"[router] {key} in {self.path} is no longer used (geodesic expansion was removed)", stacklevel=2)
                continue
            opts[RENAMED_ROUTER_KEYS.get(key, key)] = value
        return RouterConfig(**opts)

    def engine_settings(self) -> dict[str, Any]:
        """[engine] merged with the defaults of its backend (with "bedrock", `url` is ignored)."""
        backend = self.engine.get("backend", ENGINE_DEFAULTS["backend"])
        extra = BEDROCK_DEFAULTS if backend == "bedrock" else {}
        return {**ENGINE_DEFAULTS, **extra, **self.engine}

    @property
    def engine_usage_log(self) -> Path:
        """One JSON line per paid engine call (Bedrock): time, model, tokens, stop reason, seconds."""
        return self.repo_root / CONFIG_DIR / "engine_usage.jsonl"

    def build_encoder(self) -> HiddenStateEncoder:
        opts = dict(self.encoder)
        return build_encoder(str(opts.pop("spec")), **opts)

    def open_manager(self, encoder: HiddenStateEncoder | None = None) -> NewtonianMemoryManager:
        return NewtonianMemoryManager(self.memory_config(), encoder or self.build_encoder())


def exclude_from_git(repo_root: Path) -> bool:
    """Add `.zt/` to .git/info/exclude (local, never committed). False if not a git work tree."""
    if shutil.which("git") is None:
        return False
    try:
        git_dir = subprocess.run(["git", "-C", str(repo_root), "rev-parse", "--git-path", "info/exclude"],
                                 capture_output=True, text=True, check=True, timeout=10).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return False
    exclude = Path(git_dir) if Path(git_dir).is_absolute() else Path(repo_root) / git_dir
    exclude.parent.mkdir(parents=True, exist_ok=True)
    current = exclude.read_text(encoding="utf-8") if exclude.exists() else ""
    if f"/{CONFIG_DIR}/" not in current.split():
        exclude.write_text(current + ("" if current.endswith("\n") or not current else "\n") + f"/{CONFIG_DIR}/\n", encoding="utf-8")
    return True
