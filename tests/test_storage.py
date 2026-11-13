import numpy as np
import pytest

from zt_memory.config import NodeKind
from zt_memory.storage import GrowableMemmap, MetadataStore, NodeRecord


def test_memmap_grows_and_persists(tmp_path):
    mm = GrowableMemmap(tmp_path / "a.f32", np.float32, (4,), initial_rows=2)
    mm.array[:2] = 1.5
    mm.ensure(10)
    assert mm.capacity >= 10 and np.all(mm.array[:2] == 1.5) and np.all(mm.array[2:10] == 0)
    mm.array[9] = 7.0
    mm.close()
    again = GrowableMemmap(tmp_path / "a.f32", np.float32, (4,))
    assert again.capacity >= 10 and np.all(again.array[9] == 7.0)


def _rec(i: int, key: str) -> NodeRecord:
    return NodeRecord(i, key, NodeKind.CHUNK, None, "a.py", "f", 1, 2, "h", None, 0, 0)


def test_slots_recycled_and_transactions_roll_back(tmp_path):
    store = MetadataStore(tmp_path / "m.sqlite3")
    with store.transaction():
        slots = store.allocate_slots(3)
        store.insert([_rec(s, f"k{s}") for s in slots])
    assert slots == [0, 1, 2]
    with store.transaction():
        store.delete([1])
    assert store.allocate_slots(2) == [1, 3]
    with pytest.raises(RuntimeError):
        with store.transaction():
            store.insert([_rec(10, "k10")])
            raise RuntimeError("boom")
    assert store.get(10) is None
    assert store.get_by_key("k0").id == 0
