from __future__ import annotations

from zt_memory.config import MemoryConfig
from zt_memory.encoders import HashingEncoder
from zt_memory.interferometer import compress_pytest_output, inject_failure_shockwave
from zt_memory.manager import NewtonianMemoryManager


def test_destructive_interference_on_passing_suite():
    raw_pass = """============================= test session starts =============================
platform win32 -- Python 3.12.10, pytest-9.1.1, pluggy-1.6.0
rootdir: C:\\repo
collected 48 items

tests/test_alpha.py ........................                             [ 50%]
tests/test_beta.py ........................                              [100%]

============================== 48 passed in 1.23s =============================
"""
    rep = compress_pytest_output(raw_pass, exit_code=0)
    assert rep.passed is True
    assert "PASS [48 passed in 1.23s]" in rep.compact_output
    assert rep.compact_tokens <= 20
    assert rep.compression_ratio > 0.80


def test_constructive_fault_wave_and_newtonian_shockwave(tmp_path):
    repo = tmp_path / "repo"
    (repo / "src" / "calc").mkdir(parents=True)
    (repo / "tests").mkdir(parents=True)
    (repo / "src" / "calc" / "vat.py").write_text(
        """def compute_gross(net_amount: float, vat_rate: float = 0.22) -> float:
    if net_amount < 0:
        raise ValueError("negative net")
    return round(net_amount * (1.0 - vat_rate), 2)
""",
        encoding="utf-8",
    )
    (repo / "tests" / "test_vat.py").write_text(
        """from src.calc.vat import compute_gross

def test_compute_gross_standard():
    assert compute_gross(100.0, 0.22) == 122.0
""",
        encoding="utf-8",
    )
    cfg = MemoryConfig(repo_root=repo, storage_dir=tmp_path / "store", rank=16, calibration_samples=32)
    with NewtonianMemoryManager(cfg, HashingEncoder(dim=64)) as mgr:
        mgr.sync()
        rec_before = mgr.record_by_key("chunk:src/calc/vat.py::compute_gross")
        assert rec_before is not None
        e_before = float(mgr.energies([rec_before.id])[0])

        raw_fail = """============================= test session starts =============================
platform win32 -- Python 3.12.10, pytest-9.1.1, pluggy-1.6.0
collected 1 item

tests/test_vat.py F                                                      [100%]

================================== FAILURES ===================================
_________________________ test_compute_gross_standard _________________________

    def test_compute_gross_standard():
>       assert compute_gross(100.0, 0.22) == 122.0
E       assert 78.0 == 122.0
E        +  where 78.0 = compute_gross(100.0, 0.22)

tests/test_vat.py:4: AssertionError
C:\\Python312\\Lib\\site-packages\\_pytest\\python.py:195: in pytest_pyfunc_call
    result = testfunction(**testargs)
src/calc/vat.py:4: in compute_gross
>   return round(net_amount * (1.0 - vat_rate), 2)
=========================== short test summary info ===========================
FAILED tests/test_vat.py::test_compute_gross_standard - assert 78.0 == 122.0
============================== 1 failed in 0.18s ==============================
"""
        rep = compress_pytest_output(raw_fail, exit_code=1, repo_files=mgr.file_paths())
        assert rep.passed is False
        assert "_pytest" not in rep.compact_output
        assert "src/calc/vat.py:4" in rep.compact_output
        rep_heated = inject_failure_shockwave(mgr, rep)
        e_after = float(mgr.energies([rec_before.id])[0])
        assert e_after > e_before
        assert any(sym == "src/calc/vat.py::compute_gross" for _, sym, _ in rep_heated.heated_nodes)
