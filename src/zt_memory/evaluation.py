"""Encoder / SVD-rank calibration on your own code.

One encoding pass of a corpus sample, then for every candidate rank r:

  evr(r)              cumulative explained variance ratio of the truncated SVD basis.
  neighbor_recall(r)  mean |kNN_full(i) ∩ kNN_r(i)| / k over anchor chunks, cosine in both spaces.
                      Reference = raw encoder embeddings, i.e. what the model was trained for; this
                      measures what *retrieval* loses to compression (EVR only measures variance).
  probe_recall(r)     optional, labelled: share of probe queries whose expected path is in the top-k.

Methods: "svd" (PCA, centered), "svd-raw" (uncentered truncated SVD: preserves raw cosine), "prefix" (first r raw
dims, valid for Matryoshka-trained models such as Qwen3-Embedding).

Centering alone moves cosine neighbourhoods when embeddings are anisotropic (they share a mean direction), so
"svd" cannot exceed the *centering ceiling* = neighbor_recall(raw, raw - mean) at any rank. Fidelity is reported
as neighbor_recall / ceiling of the method, which separates compression loss from the centering choice.
"""
from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Literal, Sequence

import torch
from torch import Tensor

from .chunking import chunk_source, embedding_text
from .config import MemoryConfig
from .discovery import discover
from .encoders import HiddenStateEncoder
from .projector import SubspaceProjector

Method = Literal["svd", "svd-raw", "prefix"]


@dataclass(frozen=True)
class Probe:
    query: str
    expected: str  # path or path prefix that must appear in the top-k


@dataclass
class CorpusSample:
    texts: list[str]
    paths: list[str]
    symbols: list[str]
    total_chunks: int
    total_files: int


@dataclass(frozen=True)
class RankReport:
    method: Method
    rank: int
    evr: float | None
    neighbor_recall: float
    probe_recall: float | None
    bytes_per_node: int  # SSD: position + velocity (float32)
    ceiling: float = 1.0  # best neighbor recall this method can reach at full rank

    @property
    def fidelity(self) -> float:
        return self.neighbor_recall / self.ceiling if self.ceiling > 0 else 0.0


@dataclass
class CalibrationReport:
    encoder: str
    dim: int
    samples: int
    total_chunks: int
    encode_seconds: float
    chunks_per_second: float
    k: int
    rows: list[RankReport] = field(default_factory=list)
    centering_ceiling: float = 1.0
    probe_details: list[dict[str, object]] = field(default_factory=list)

    def best(self, target: float, method: Method = "svd") -> RankReport | None:
        return recommend_rank(self.rows, target, method)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def load_probes(path: Path) -> list[Probe]:
    """JSON: [{"query": ..., "expected": ...}, ...] or {"query": "expected", ...}."""
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))  # tolerate the BOM PowerShell writes
    if isinstance(data, dict):
        return [Probe(q, e) for q, e in data.items()]
    return [Probe(d["query"], d["expected"]) for d in data]


def sample_corpus(config: MemoryConfig, n: int, *, seed: int = 0, include_prefixes: Sequence[str] = ()) -> CorpusSample:
    """Chunk the whole repository the way ingest does, then sample n chunks deterministically.
    Chunks under `include_prefixes` (probe targets) are always kept so probe recall is measurable."""
    texts: list[str] = []
    paths: list[str] = []
    symbols: list[str] = []
    files = discover(config)
    for rel in files:
        try:
            raw = (config.repo_root / rel).read_bytes()
            if len(raw) > config.max_file_bytes or b"\x00" in raw[:8192]:
                continue
            text = raw.decode("utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for ch in chunk_source(rel, text, max_lines=config.chunk_max_lines, min_lines=config.chunk_min_lines, backend=config.chunk_backend):
            texts.append(embedding_text(rel, ch))
            paths.append(rel)
            symbols.append(ch.symbol)
    total = len(texts)
    forced = [i for i, p in enumerate(paths) if any(p.startswith(x) for x in include_prefixes)]
    forced_set = set(forced)
    rest = [i for i in range(total) if i not in forced_set]
    rng = random.Random(seed)
    pick = sorted(set(forced) | set(rng.sample(rest, min(len(rest), max(0, n - len(forced))))))
    return CorpusSample([texts[i] for i in pick], [paths[i] for i in pick], [symbols[i] for i in pick], total, len(files))


def encode_corpus(encoder: HiddenStateEncoder, texts: Sequence[str], *, batch: int = 64,
                  progress: Callable[[int, int], None] | None = None) -> tuple[Tensor, float]:
    t0 = time.perf_counter()
    parts: list[Tensor] = []
    for i in range(0, len(texts), batch):
        parts.append(encoder.encode(list(texts[i : i + batch])).to(torch.float32))
        if progress:
            progress(min(i + batch, len(texts)), len(texts))
    dim = encoder.dim
    return (torch.cat(parts) if parts else torch.zeros((0, dim))), time.perf_counter() - t0


def _unit(x: Tensor) -> Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def _topk_neighbors(x: Tensor, anchors: Tensor, k: int) -> Tensor:
    sims = _unit(x[anchors]) @ _unit(x).T
    sims[torch.arange(anchors.numel()), anchors] = -float("inf")  # exclude self
    return sims.topk(k, dim=1).indices


def neighbor_recall(full: Tensor, reduced: Tensor, *, k: int = 10, anchors: Tensor | None = None) -> float:
    if full.shape[0] <= k:
        raise ValueError("need more samples than k")
    a = anchors if anchors is not None else torch.arange(full.shape[0])
    ref = _topk_neighbors(full.to(torch.float64), a, k)
    got = _topk_neighbors(reduced.to(torch.float64), a, k)
    hits = sum(len(set(r.tolist()) & set(g.tolist())) for r, g in zip(ref, got))
    return hits / (k * a.numel())


def probe_hits(query_vecs: Tensor, doc_vecs: Tensor, doc_paths: Sequence[str], probes: Sequence[Probe], *, k: int = 10) -> list[dict[str, object]]:
    """Per probe: 0-based rank of the first chunk under the expected path (None if not in top-k), and the top-1 path."""
    sims = _unit(query_vecs.to(torch.float64)) @ _unit(doc_vecs.to(torch.float64)).T
    top = sims.topk(min(k, doc_vecs.shape[0]), dim=1).indices
    out: list[dict[str, object]] = []
    for row, p in zip(top, probes):
        idx = row.tolist()
        hit = next((r for r, j in enumerate(idx) if doc_paths[j].startswith(p.expected)), None)
        out.append({"query": p.query, "expected": p.expected, "hit_rank": hit, "top1": doc_paths[idx[0]]})
    return out


def probe_recall(query_vecs: Tensor, doc_vecs: Tensor, doc_paths: Sequence[str], probes: Sequence[Probe], *, k: int = 10) -> float:
    if not probes:
        raise ValueError("no probes")
    hits = probe_hits(query_vecs, doc_vecs, doc_paths, probes, k=k)
    return sum(h["hit_rank"] is not None for h in hits) / len(hits)


def centering_ceiling(hidden: Tensor, *, k: int = 10, anchors: Tensor | None = None) -> float:
    return neighbor_recall(hidden, hidden - hidden.mean(dim=0), k=k, anchors=anchors)


def reduce(method: Method, projectors: dict[str, SubspaceProjector], x: Tensor, rank: int) -> Tensor:
    if method == "prefix":
        return x[:, :rank]
    return projectors[method].truncated(rank).transform(x)


def rank_sweep(
    hidden: Tensor,
    ranks: Sequence[int],
    *,
    k: int = 10,
    n_anchors: int = 512,
    drop_top: int = 0,
    seed: int = 0,
    methods: Sequence[Method] = ("svd", "svd-raw", "prefix"),
    query_hidden: Tensor | None = None,
    probes: Sequence[Probe] = (),
    doc_paths: Sequence[str] = (),
) -> tuple[list[RankReport], dict[str, SubspaceProjector]]:
    """Evaluate all ranks from a single fit per SVD flavour at max(ranks). Returns rows and the fitted projectors."""
    n, d = hidden.shape
    ranks = sorted({r for r in ranks if 0 < r <= min(d, n - 1 - drop_top)})
    if not ranks:
        raise ValueError(f"no valid rank for n={n}, d={d}")
    g = torch.Generator().manual_seed(seed)
    anchors = torch.randperm(n, generator=g)[: min(n_anchors, n)]
    projectors: dict[str, SubspaceProjector] = {}
    for method in methods:
        if method in ("svd", "svd-raw"):
            proj = SubspaceProjector(max(ranks), drop_top=drop_top, center=method == "svd")
            proj.fit(hidden)
            proj.freeze()
            projectors[method] = proj
    ceilings = {"svd": centering_ceiling(hidden, k=k, anchors=anchors) if "svd" in projectors else 1.0, "svd-raw": 1.0, "prefix": 1.0}
    with_probes = query_hidden is not None and len(probes) > 0
    rows: list[RankReport] = []
    for method in methods:
        cum = projectors[method].cumulative_evr() if method in projectors else None
        for r in ranks:
            docs = reduce(method, projectors, hidden, r)
            queries = reduce(method, projectors, query_hidden, r) if with_probes else None
            rows.append(RankReport(
                method, r, float(cum[r - 1]) if cum is not None else None,
                neighbor_recall(hidden, docs, k=k, anchors=anchors),
                probe_recall(queries, docs, doc_paths, probes, k=k) if with_probes else None, 8 * r, ceilings[method],
            ))
    return rows, projectors


def recommend_rank(rows: Sequence[RankReport], target: float, method: Method | None = None) -> RankReport | None:
    """Smallest rank whose *absolute* neighbor recall reaches `target` and whose probe recall (when measured) is
    within one probe of the best observed; ties prefer higher recall. method=None: across both SVD flavours.
    (Fidelity vs. ceiling explains a plateau; it must not decide, or a centered basis stuck at its ceiling wins.)"""
    pool = [r for r in rows if (r.method == method if method else r.method in ("svd", "svd-raw"))]
    if not pool:
        return None
    best_probe = max((r.probe_recall or 0.0) for r in pool)
    n_probe_step = min((abs(a.probe_recall - b.probe_recall) for a in pool for b in pool
                        if a.probe_recall is not None and b.probe_recall is not None and a.probe_recall != b.probe_recall), default=0.0)
    ok = [r for r in pool if r.neighbor_recall >= target and (r.probe_recall is None or r.probe_recall >= best_probe - n_probe_step - 1e-9)]
    if ok:
        return min(ok, key=lambda r: (r.rank, -r.neighbor_recall))
    return max(pool, key=lambda r: (r.neighbor_recall, -r.rank))
