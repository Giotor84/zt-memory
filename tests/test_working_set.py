import numpy as np
import torch

from zt_memory.config import NodeKind, default_physics
from zt_memory.storage import NodeArrays
from zt_memory.working_set import WorkingSet

R = 8


def _ws(tmp_path, budget):
    arrays = NodeArrays(tmp_path, R)
    phys = default_physics()
    beta = np.array([phys[k].beta for k in NodeKind])
    gamma = np.array([phys[k].gamma for k in NodeKind])
    ids = np.arange(200, dtype=np.int64)
    arrays.init_slots(ids, np.full(200, int(NodeKind.CHUNK), np.uint8), np.zeros((200, R), np.float32), now=1)
    return arrays, WorkingSet(arrays, R, budget, beta, gamma)


def test_budget_bounds_residency_and_writes_back(tmp_path):
    arrays, ws = _ws(tmp_path, budget=40 * (R * 4 + 126))
    assert ws.capacity == 40
    for start in range(0, 200, 20):
        ids = np.arange(start, start + 20, dtype=np.int64)
        slots = ws.fetch(ids, now=2)
        v = torch.full((20, R), float(start + 1), dtype=torch.float64)
        ws.write(slots, v, torch.full((20,), float(start + 1), dtype=torch.float64), now=2)
        assert len(ws) <= ws.capacity
    st = ws.stats()
    assert st.evictions > 0 and st.writebacks > 0
    ws.flush()
    assert np.allclose(arrays.energy.array[0:20], 1.0)
    assert np.allclose(arrays.velocity.array[180:200], 181.0)


def test_evicts_lowest_energy_first(tmp_path):
    arrays, ws = _ws(tmp_path, budget=40 * (R * 4 + 126))
    ids = np.arange(20, dtype=np.int64)
    slots = ws.fetch(ids, now=2)
    energies = torch.arange(20, dtype=torch.float64)
    ws.write(slots, torch.zeros((20, R), dtype=torch.float64), energies, now=2)
    ws.evict(5, now=2)
    resident = {i for i in range(20) if i in ws}
    assert resident == set(range(5, 20))
    assert np.allclose(arrays.energy.array[:5], np.arange(5))  # written back before release
