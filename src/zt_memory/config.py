"""Configuration objects and node taxonomy."""
from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


class NodeKind(enum.IntEnum):
    """Hierarchy levels. Lower value = closer to the origin of the Poincaré ball."""

    ROOT = 0
    DIR = 1
    FILE = 2
    CHUNK = 3
    SPEC = 4  # greenfield requirement (text payload stored in metadata)
    DECISION = 5  # architectural decision / scaffold note
    EPHEMERAL = 6  # transient note, high friction


TEXT_KINDS: frozenset[NodeKind] = frozenset({NodeKind.SPEC, NodeKind.DECISION, NodeKind.EPHEMERAL})
STRUCTURAL_KINDS: frozenset[NodeKind] = frozenset({NodeKind.ROOT, NodeKind.DIR, NodeKind.FILE})
# Nodes that constitute the "memory state" against which novelty (surprise) of new inputs is measured.
MEMORY_KINDS: frozenset[NodeKind] = frozenset({NodeKind.CHUNK, NodeKind.SPEC, NodeKind.DECISION})


@dataclass(frozen=True)
class KindPhysics:
    """Per-kind physical constants.

    beta:   inertia,  v_t = beta * v_{t-1} + (1 - beta) * F_t
    gamma:  friction, E_t = gamma * E_{t-1} + ||v_t||   (gamma < 1)
    radius: Euclidean radius in the Poincaré ball encoding the hierarchy level.
    """

    beta: float
    gamma: float
    radius: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.beta < 1.0:
            raise ValueError(f"beta must be in [0, 1), got {self.beta}")
        if not 0.0 < self.gamma < 1.0:
            raise ValueError(f"gamma must be in (0, 1), got {self.gamma}")
        if not 0.0 <= self.radius < 1.0:
            raise ValueError(f"radius must be in [0, 1), got {self.radius}")


def default_physics() -> dict[NodeKind, KindPhysics]:
    return {
        NodeKind.ROOT: KindPhysics(beta=0.90, gamma=0.990, radius=0.00),
        NodeKind.DIR: KindPhysics(beta=0.90, gamma=0.970, radius=0.55),
        NodeKind.FILE: KindPhysics(beta=0.90, gamma=0.950, radius=0.80),
        NodeKind.CHUNK: KindPhysics(beta=0.85, gamma=0.920, radius=0.90),
        NodeKind.SPEC: KindPhysics(beta=0.95, gamma=0.995, radius=0.85),
        NodeKind.DECISION: KindPhysics(beta=0.95, gamma=0.990, radius=0.85),
        NodeKind.EPHEMERAL: KindPhysics(beta=0.50, gamma=0.600, radius=0.90),
    }


DEFAULT_CODE_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".py", ".pyi", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".java", ".kt", ".kts", ".scala",
        ".go", ".rs", ".c", ".h", ".cc", ".cpp", ".cxx", ".hpp", ".hh", ".cs", ".rb", ".php", ".swift",
        ".m", ".mm", ".lua", ".sh", ".bash", ".zsh", ".sql", ".proto", ".graphql", ".toml", ".yaml",
        ".yml", ".json", ".md", ".rst", ".cfg", ".ini", ".gradle", ".cmake", ".vue", ".svelte", ".dart",
        ".ex", ".exs", ".erl", ".hs", ".ml", ".r", ".jl", ".zig", ".nim", ".tf", ".html", ".htm", ".css",
    }
)
DEFAULT_CODE_FILENAMES: frozenset[str] = frozenset({"Dockerfile", "Makefile", "CMakeLists.txt", "Justfile"})
DEFAULT_EXCLUDED_DIRS: frozenset[str] = frozenset(
    {
        ".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv", "env", ".mypy_cache",
        ".pytest_cache", ".ruff_cache", ".tox", "build", "dist", "target", ".idea", ".vscode", ".next",
        ".gradle", "vendor", ".cache", "site-packages",
    }
)


@dataclass
class MemoryConfig:
    """Configuration of the SSD/RAM memory tier. Nothing here ever touches the GPU."""

    storage_dir: Path
    repo_root: Path
    # Truncated-SVD rank. Validate with projector.explained_variance_ratio() and a recall probe:
    # rank 64 on lexical features keeps ~34% variance and loses rare-term recall (see README).
    rank: int = 128
    curvature: float = 1.0
    # RAM working-set budget for hot dynamic state. Must leave room for OS, encoder and LLM host process.
    ram_budget_bytes: int = 1536 * 2**20
    # Evict aggressively when system available RAM drops below this floor (requires psutil; 0 disables).
    ram_floor_bytes: int = 2 * 2**30
    eviction_fraction: float = 0.10
    ancestor_decay: float = 0.5  # rho: force attenuation per hierarchy level
    query_force_scale: float = 0.25  # read-side reinforcement (attention heats nodes)
    # Write events add kappa * x_hat (Riemannian magnitude) on top of the geometric surprise, so an edit
    # that barely moves the embedding still registers, and recurring edits accumulate coherent momentum.
    edit_impulse: float = 0.5
    artifact_force_scale: float = 0.5  # greenfield: generated artifacts reinforce their specs
    index_backend: Literal["auto", "scan", "hnsw"] = "auto"
    hnsw_threshold: int = 1_000_000  # "auto" switches to HNSW above this many slots
    scan_block_rows: int = 65_536
    ingest_group_size: int = 256  # chunks embedded per encoder batch group
    calibration_samples: int = 1024
    svd_drop_top: int = 0  # "all-but-the-top": drop dominant anisotropic directions of raw hidden states
    # False = uncentered truncated SVD. Qwen3-Embedding-0.6B on this codebase: nbr@10 0.938 vs 0.876 centered (rank 128).
    svd_center: bool = False
    max_file_bytes: int = 1_000_000
    respect_gitignore: bool = True  # inside a git work tree, index exactly what git would track
    # Comma-separated patterns never indexed even if git tracks them (files an agent must not read), matched
    # right-anchored on the repo-relative path: "debtors.py" anywhere, "pkg/debtors.py", "fixtures/*.json".
    # Case-insensitive (Windows file systems are): an exclusion list errs on the side of excluding.
    exclude_globs: str = ""
    lock_timeout_s: float = 30.0  # wait for another process holding the store (watcher, git hook)
    chunk_max_lines: int = 120  # larger symbols/gaps are split into content-anchored blocks
    chunk_min_lines: int = 12  # minimum block size before a natural (blank-line) cut
    chunk_backend: Literal["auto", "treesitter", "ast", "blocks"] = "auto"
    sqlite_cache_kib: int = 65_536
    code_extensions: frozenset[str] = DEFAULT_CODE_EXTENSIONS
    code_filenames: frozenset[str] = DEFAULT_CODE_FILENAMES
    excluded_dirs: frozenset[str] = DEFAULT_EXCLUDED_DIRS
    physics: dict[NodeKind, KindPhysics] = field(default_factory=default_physics)

    def __post_init__(self) -> None:
        self.storage_dir = Path(self.storage_dir).expanduser().resolve()
        self.repo_root = Path(self.repo_root).expanduser().resolve()
        if self.rank <= 0:
            raise ValueError("rank must be positive")
        if self.curvature <= 0:
            raise ValueError("curvature must be positive")
        if self.ram_budget_bytes <= 0:
            raise ValueError("ram_budget_bytes must be positive")
        if not 0.0 < self.eviction_fraction <= 1.0:
            raise ValueError("eviction_fraction must be in (0, 1]")
        if self.edit_impulse < 0:
            raise ValueError("edit_impulse must be >= 0")
        if not 0.0 <= self.ancestor_decay < 1.0:
            raise ValueError("ancestor_decay must be in [0, 1)")
        if not 1 <= self.chunk_min_lines < self.chunk_max_lines:
            raise ValueError("need 1 <= chunk_min_lines < chunk_max_lines")
        missing = set(NodeKind) - set(self.physics)
        if missing:
            raise ValueError(f"physics missing kinds: {sorted(k.name for k in missing)}")
        max_radius = 1.0 / math.sqrt(self.curvature)
        for kind, phys in self.physics.items():
            if phys.radius >= max_radius:
                raise ValueError(f"radius of {kind.name} must be < 1/sqrt(c) = {max_radius:.4f}")
        if self.physics[NodeKind.ROOT].radius != 0.0:
            raise ValueError("ROOT must sit at the origin (radius 0)")


Mode = Literal["brownfield", "greenfield"]


def _default_kind_weights(mode: Mode) -> dict[NodeKind, float]:
    if mode == "greenfield":
        return {
            NodeKind.CHUNK: 0.8, NodeKind.SPEC: 1.3, NodeKind.DECISION: 1.0,
            NodeKind.EPHEMERAL: 0.5, NodeKind.FILE: 0.35,
        }
    return {
        NodeKind.CHUNK: 1.0, NodeKind.SPEC: 0.8, NodeKind.DECISION: 0.8,
        NodeKind.EPHEMERAL: 0.5, NodeKind.FILE: 0.35,
    }


@dataclass
class RouterConfig:
    seed_k: int = 48
    expansion_seeds: int = 8  # top geometric seeds whose graph edges (spec/decision links) and file headers join
    token_budget: int = 12_000  # ~2.3 GB fp16 KV for a 14B GQA model (48L x 8KV x 128d)
    template_overhead_tokens: int = 256
    min_block_tokens: int = 32
    # Physics prior: w_E=1.5 raised recall@10 in 4/4 repos of the git-history ablation (+0.018 mean); 0.5 did not.
    energy_weight: float = 1.5
    header_lines: int = 40
    include_file_headers: bool = True
    reinforce: bool = True
    chars_per_token: float = 3.2
    file_map_limit: int = 200
    # Hybrid retrieval: Reciprocal Rank Fusion of the geometric and the lexical (FTS5/BM25) rankings.
    lexical_k: int = 48
    lexical_seeds: int = 4  # top lexical hits that also seed the graph/header expansion
    geometric_weight: float = 1.0
    lexical_weight: float = 1.0
    code_query_lexical_boost: float = 2.0  # multiplies lexical_weight for identifier/error-code queries
    # Physics as a third ranked list: candidates ranked by current energy join the RRF fusion, and the hottest
    # chunks are added as candidates (the recency prior also proposes, it does not only re-weight).
    recency_weight: float = 0.0
    recency_seeds: int = 16
    rrf_k: float = 10.0  # small: candidate lists are short (~100), the standard 60 over-flattens them
    # Non-symbol chunks (<header>, <gap:...>, <block:...>) are context, rarely the edit target.
    nonsymbol_chunk_weight: float = 0.6
    # Tests are context, rarely the edit target; 1.0 disables (e.g. when the task is about a failing test).
    test_path_weight: float = 1.0
    # Dynamic cut: after min_blocks, stop when score < cutoff_alpha * best score (0 = fill the budget).
    cutoff_alpha: float = 0.25  # ablation: -21% tokens for -5 points of true-file coverage (mean of 4 repos)
    min_blocks: int = 3
    structural_headers: int = 3  # headers of the top files with selected code, added even below the cut
    # Locations named in the query's error output (tracebacks, compiler messages): the symbols at those lines are
    # pinned first, source frames before test frames, innermost first. 0 disables.
    trace_anchors: int = 4
    trace_window_lines: int = 80  # a pinned symbol longer than this is cut to a window around the named line
    # Focused context (all off by default = the behaviour measured so far).
    # dedup: drop a stub (.pyi/.d.ts) when its implementation is a candidate (a lone stub is shown as a signature) and
    # drop chunks whose cosine with an already chosen chunk exceeds dup_threshold. diversity_lambda: MMR order,
    # score/best - lambda * max cosine with the chunks already chosen (chunk vs chunk only: files and notes live at
    # other radii of the ball and are never compared). focus_blocks / focus_ratio: only traceback-pinned chunks, the
    # first focus_blocks chosen and those scoring >= focus_ratio * best keep their body (stubs and tests never take a
    # focus slot); every other chosen chunk becomes its signature line. signature_extra: spend the tokens saved on up
    # to this many more candidates, as signatures.
    dedup: bool = False
    dup_threshold: float = 0.95
    diversity_lambda: float = 0.0
    focus_blocks: int = 0
    focus_ratio: float = 0.6
    signature_extra: int = 0
    # Adaptive focus (on top of focus_blocks): focus_mass > 0 also keeps whole the chosen production chunks (not
    # pinned, not tests or stubs) until they cover this share of their total score; focus_flat_blocks widens the focus to
    # that many chunks when the score drop from the 1st to the 5th production chunk is below focus_flat_drop.
    focus_mass: float = 0.0
    focus_flat_drop: float = 0.0
    focus_flat_blocks: int = 0
    # focus_skip_glue: module code between symbols (<header>, <gap:...> chunks) takes no focus slot and is sent whole
    # only when its file has a symbol in focus (like the file headers); files without a grammar (<block:...>) still count.
    # Measured worse (fase 1b: -3..-6 cases): fixes are often in that module code, so it must keep competing for slots.
    focus_skip_glue: bool = False
    # With a traceback in the query (pinned symbols) and traceback_focus_blocks > 0: plain focus of that width, no
    # score-mass rule, no extra signatures (the answer is already pinned: no second round is offered).
    traceback_focus_blocks: int = 0
    brownfield_kind_weights: dict[NodeKind, float] = field(default_factory=lambda: _default_kind_weights("brownfield"))
    greenfield_kind_weights: dict[NodeKind, float] = field(default_factory=lambda: _default_kind_weights("greenfield"))

    def kind_weights(self, mode: Mode) -> dict[NodeKind, float]:
        return self.greenfield_kind_weights if mode == "greenfield" else self.brownfield_kind_weights


# Router keys accepted in old .zt/config.toml files but no longer used (ignored with a warning).
REMOVED_ROUTER_KEYS = frozenset({"geodesic_samples", "geodesic_k"})
RENAMED_ROUTER_KEYS = {"geodesic_seeds": "expansion_seeds"}
