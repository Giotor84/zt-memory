"""Truncated-SVD subspace projector ("spaghettizzazione").

A single hidden-state vector has no singular spectrum of its own: truncated SVD is
meaningful only over a sample matrix. The projector therefore fits a basis U_r on a
calibration corpus (incremental PCA, Ross et al. 2008), is then frozen, and projects
every subsequent hidden state onto it. Refitting after vectors are stored would
invalidate them, so the basis is versioned and immutable once frozen.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import torch
from torch import Tensor


class ProjectorStateError(RuntimeError):
    pass


class SubspaceProjector:
    def __init__(self, rank: int, *, drop_top: int = 0, center: bool = True) -> None:
        """center=True: PCA (subtract the mean). center=False: truncated SVD of the raw matrix, which best
        preserves raw inner products / cosine, i.e. the geometry contrastive embedders are trained for."""
        if rank <= 0:
            raise ValueError("rank must be positive")
        if drop_top < 0:
            raise ValueError("drop_top must be >= 0")
        self.rank = rank
        self.drop_top = drop_top
        self.center = center
        self._n = 0
        self._mean: Tensor | None = None
        self._m2: Tensor | None = None
        self._components: Tensor | None = None
        self._singular: Tensor | None = None
        self._frozen = False

    @property
    def n_target(self) -> int:
        return self.rank + self.drop_top

    @property
    def dim(self) -> int | None:
        return None if self._mean is None else int(self._mean.shape[0])

    @property
    def n_samples(self) -> int:
        return self._n

    @property
    def fitted(self) -> bool:
        return self._components is not None and self._components.shape[0] >= self.n_target

    @property
    def frozen(self) -> bool:
        return self._frozen

    def reset(self) -> None:
        if self._frozen:
            raise ProjectorStateError("projector is frozen")
        self.__init__(self.rank, drop_top=self.drop_top, center=self.center)  # type: ignore[misc]

    def partial_fit(self, hidden: Tensor) -> None:
        if self._frozen:
            raise ProjectorStateError("projector is frozen; stored vectors depend on this basis")
        x = hidden.detach().to(device="cpu", dtype=torch.float64)
        if x.ndim != 2:
            raise ValueError("expected a [n, d] matrix")
        m = x.shape[0]
        if m == 0:
            return
        if self.dim is not None and x.shape[1] != self.dim:
            raise ValueError(f"dimension mismatch: {x.shape[1]} != {self.dim}")
        if x.shape[1] < self.n_target:
            raise ValueError(f"hidden dim {x.shape[1]} < rank + drop_top = {self.n_target}")
        b_mean = x.mean(dim=0) if self.center else torch.zeros(x.shape[1], dtype=torch.float64)
        b_centered = x - b_mean
        b_m2 = (b_centered**2).sum(dim=0)
        if self._n == 0:
            mean, m2, n, stack = b_mean, b_m2, m, b_centered
        else:
            assert self._mean is not None and self._m2 is not None
            assert self._components is not None and self._singular is not None
            n_old = self._n
            n = n_old + m
            delta = b_mean - self._mean
            mean = self._mean + delta * (m / n)
            m2 = self._m2 + b_m2 + delta**2 * (n_old * m / n)
            correction = math.sqrt(n_old * m / n) * (self._mean - b_mean)
            stack = torch.cat([self._singular[:, None] * self._components, b_centered, correction[None, :]], dim=0)
        _, s, vh = torch.linalg.svd(stack, full_matrices=False)
        k = min(self.n_target, s.shape[0])
        self._components = vh[:k].contiguous()
        self._singular = s[:k].contiguous()
        self._mean, self._m2, self._n = mean, m2, n

    def fit(self, hidden: Tensor) -> None:
        self.reset()
        self.partial_fit(hidden)

    def freeze(self) -> None:
        if not self.fitted:
            have = 0 if self._components is None else self._components.shape[0]
            raise ProjectorStateError(f"need {self.n_target} components, have {have} (more calibration samples needed)")
        self._frozen = True

    def explained_variance_ratio(self) -> Tensor:
        """Variance fraction captured by each kept component (after drop_top)."""
        if not self.fitted or self._m2 is None or self._singular is None:
            raise ProjectorStateError("projector not fitted")
        denom = max(self._n - 1, 1)
        total = float(self._m2.sum()) / denom
        var = self._singular[self.drop_top : self.n_target] ** 2 / denom
        return (var / max(total, 1e-30)).to(torch.float32)

    def transform(self, hidden: Tensor) -> Tensor:
        if not self.fitted or self._mean is None or self._components is None:
            raise ProjectorStateError("projector not fitted")
        x = hidden.detach().to(device="cpu", dtype=torch.float64) - self._mean
        basis = self._components[self.drop_top : self.n_target]
        return (x @ basis.T).to(torch.float32)

    def truncated(self, rank: int) -> "SubspaceProjector":
        """Frozen projector keeping the top `rank` components (PCA bases are nested): lets a rank sweep
        reuse one fit, and the chosen rank be saved without re-encoding the calibration corpus."""
        if not self.fitted or self._components is None or self._singular is None:
            raise ProjectorStateError("projector not fitted")
        if not 0 < rank <= self.rank:
            raise ValueError(f"rank must be in [1, {self.rank}]")
        out = SubspaceProjector(rank, drop_top=self.drop_top, center=self.center)
        out._n, out._mean, out._m2 = self._n, self._mean, self._m2
        out._components = self._components[: self.drop_top + rank].clone()
        out._singular = self._singular[: self.drop_top + rank].clone()
        out._frozen = True
        return out

    def cumulative_evr(self) -> Tensor:
        """Cumulative explained variance of the first 1..rank kept components."""
        return torch.cumsum(self.explained_variance_ratio(), dim=0)

    @property
    def fingerprint(self) -> str:
        if self._components is None:
            return "unfitted"
        h = hashlib.blake2b(digest_size=12)
        h.update(self._components.to(torch.float32).numpy().tobytes())
        h.update(f"{self.rank}:{self.drop_top}:{int(self.center)}".encode())
        return h.hexdigest()

    def save(self, path: Path) -> None:
        if not self.fitted:
            raise ProjectorStateError("refusing to save an unfitted projector")
        state = {
            "rank": torch.tensor(self.rank),
            "drop_top": torch.tensor(self.drop_top),
            "n": torch.tensor(self._n),
            "mean": self._mean,
            "m2": self._m2,
            "components": self._components,
            "singular": self._singular,
            "frozen": torch.tensor(int(self._frozen)),
            "center": torch.tensor(int(self.center)),
        }
        tmp = Path(path).with_suffix(".tmp")
        torch.save(state, tmp)
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> "SubspaceProjector":
        state = torch.load(path, map_location="cpu", weights_only=True)
        center = bool(int(state["center"])) if "center" in state else True  # files written before the flag existed
        proj = cls(int(state["rank"]), drop_top=int(state["drop_top"]), center=center)
        proj._n = int(state["n"])
        proj._mean = state["mean"].to(torch.float64)
        proj._m2 = state["m2"].to(torch.float64)
        proj._components = state["components"].to(torch.float64)
        proj._singular = state["singular"].to(torch.float64)
        proj._frozen = bool(int(state["frozen"]))
        return proj
