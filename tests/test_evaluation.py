import torch

from zt_memory import HashingEncoder
from zt_memory.config import MemoryConfig
from zt_memory.evaluation import Probe, encode_corpus, neighbor_recall, rank_sweep, recommend_rank, sample_corpus


def _low_rank(n=600, d=96, r=12, noise=0.02, seed=0):
    g = torch.Generator().manual_seed(seed)
    basis = torch.linalg.qr(torch.randn(d, r, generator=g))[0]
    return (torch.randn(n, r, generator=g) * torch.linspace(4, 1, r)) @ basis.T + noise * torch.randn(n, d, generator=g)


def test_neighbor_recall_bounds():
    x = _low_rank()
    assert neighbor_recall(x, x, k=10) == 1.0
    noise = torch.randn_like(x)
    assert neighbor_recall(x, noise, k=10) < 0.2


def test_rank_sweep_finds_intrinsic_rank():
    x = _low_rank(r=12)
    rows, projs = rank_sweep(x, [4, 8, 12, 24], k=10, n_anchors=200, methods=("svd",))
    proj = projs["svd"]
    by_rank = {r.rank: r for r in rows}
    assert [by_rank[r].evr for r in (4, 8, 12, 24)] == sorted(by_rank[r].evr for r in (4, 8, 12, 24))
    assert by_rank[12].evr > 0.99 and by_rank[12].neighbor_recall > 0.8
    assert by_rank[4].neighbor_recall < by_rank[12].neighbor_recall
    assert recommend_rank(rows, 0.8, "svd").rank == 12
    assert proj.truncated(12).transform(x).shape == (x.shape[0], 12)


def test_prefix_baseline_and_probe_recall_on_repo(repo, tmp_path):
    cfg = MemoryConfig(storage_dir=tmp_path / "s", repo_root=repo)
    sample = sample_corpus(cfg, 1000, include_prefixes=["src/app/db.py"])
    assert sample.total_files == 5 and len(sample.texts) == sample.total_chunks
    enc = HashingEncoder(dim=256)
    hidden, secs = encode_corpus(enc, sample.texts)
    probes = [Probe("database pool acquire connection", "src/app/db.py"), Probe("slugify title", "src/util/strings.py")]
    rows, _ = rank_sweep(hidden, [4, 8], k=3, query_hidden=enc.encode([p.query for p in probes], is_query=True),
                         probes=probes, doc_paths=sample.paths)
    assert {r.method for r in rows} == {"svd", "svd-raw", "prefix"}
    assert all(r.probe_recall is not None for r in rows)
    assert max(r.probe_recall for r in rows if r.method == "svd") == 1.0
    assert all(r.evr is None for r in rows if r.method == "prefix")


def test_centering_ceiling_bounds_centered_svd_and_raw_svd_escapes_it():
    from zt_memory.evaluation import centering_ceiling

    g = torch.Generator().manual_seed(1)
    z = _low_rank(n=500, d=128, r=40, noise=0.01, seed=1)
    x = z + 1.5 * z.norm(dim=1).mean() * torch.randn(128, generator=g) / 128**0.5  # shared offset = anisotropy
    x = x / x.norm(dim=1, keepdim=True)
    ceiling = centering_ceiling(x, k=10)
    rows, projs = rank_sweep(x, [40, 100], k=10, n_anchors=300, methods=("svd", "svd-raw"))
    by = {(r.method, r.rank): r for r in rows}
    assert ceiling < 0.9
    assert by[("svd", 100)].neighbor_recall <= ceiling + 0.02  # centered PCA cannot beat its ceiling
    assert by[("svd-raw", 100)].neighbor_recall > by[("svd", 100)].neighbor_recall  # uncentered preserves raw cosine
    assert by[("svd", 100)].fidelity > 0.95  # what is lost is the centering choice, not compression
    assert not projs["svd-raw"].center and projs["svd"].center


def test_uncentered_projector_roundtrip(tmp_path):
    from zt_memory.projector import SubspaceProjector

    x = _low_rank(n=300, d=64, r=8)
    p = SubspaceProjector(8, center=False)
    p.fit(x)
    p.freeze()
    assert float(p._mean.abs().max()) == 0.0
    p.save(tmp_path / "p.pt")
    q = SubspaceProjector.load(tmp_path / "p.pt")
    assert not q.center and torch.allclose(q.transform(x[:5]), p.transform(x[:5]))
    assert not p.truncated(4).center


def test_probe_files_with_bom_load(tmp_path):
    from zt_memory.evaluation import load_probes

    path = tmp_path / "probes.json"
    path.write_bytes(b"\xef\xbb\xbf" + b'[{"query": "q", "expected": "src/a.py"}]')
    assert load_probes(path)[0].expected == "src/a.py"


def test_recommendation_uses_absolute_recall_not_fidelity():
    from zt_memory.evaluation import RankReport

    rows = [
        RankReport("svd", 64, 0.77, 0.836, 0.917, 512, ceiling=0.886),      # fidelity 0.94, absolute 0.84
        RankReport("svd-raw", 64, 0.85, 0.868, 0.833, 512, ceiling=1.0),
        RankReport("svd-raw", 128, 0.93, 0.938, 0.917, 1024, ceiling=1.0),
    ]
    best = recommend_rank(rows, 0.9)
    assert (best.method, best.rank) == ("svd-raw", 128)
