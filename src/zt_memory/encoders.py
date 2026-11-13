"""Non-generative encoders producing raw hidden states (navigator input).

The navigator never decodes tokens: it runs a forward pass, pools a hidden layer and
hands the vector to the SVD projector. Run it on CPU so VRAM stays reserved to the
compute engine.
"""
from __future__ import annotations

import gc
import hashlib
import math
import re
import warnings
from collections import Counter
from functools import lru_cache
from typing import Literal, Protocol, Sequence, runtime_checkable

import numpy as np
import torch
from torch import Tensor


@runtime_checkable
class HiddenStateEncoder(Protocol):
    @property
    def dim(self) -> int: ...

    @property
    def fingerprint(self) -> str: ...

    def encode(self, texts: Sequence[str], *, is_query: bool = False) -> Tensor:
        """Return [n, dim] float32 CPU tensor."""
        ...


_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")
_SUBTOKEN = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")
_STOPWORDS = frozenset(
    {
        "self", "cls", "def", "return", "the", "import", "from", "class", "if", "else", "elif", "for", "in",
        "is", "not", "and", "or", "none", "true", "false", "to", "of", "a", "an", "with", "as", "this",
        "that", "be", "it", "on", "by", "pass", "var", "let", "const", "function", "new", "public", "private",
    }
)


@lru_cache(maxsize=1 << 18)
def _hash_token(token: str, dim: int) -> tuple[int, float]:
    h = int.from_bytes(hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest(), "little")
    return h % dim, (1.0 if (h >> 63) & 1 else -1.0)


class HashingEncoder:
    """Deterministic lexical baseline (signed feature hashing of identifier sub-tokens).

    Not a hidden-state model: used for tests, CI, and machines without a local encoder.
    """

    def __init__(self, dim: int = 512, *, min_token_len: int = 2) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self._dim = dim
        self._min_len = min_token_len

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def fingerprint(self) -> str:
        return f"hashing:v1:dim={self._dim}:min={self._min_len}"

    def _tokens(self, text: str) -> Counter[str]:
        counts: Counter[str] = Counter()
        for ident in _IDENT.findall(text):
            subs = [s.lower() for s in _SUBTOKEN.findall(ident)]
            candidates = [ident.lower(), *subs] if len(subs) > 1 else [ident.lower()]
            for tok in candidates:
                if len(tok) >= self._min_len and tok not in _STOPWORDS:
                    counts[tok] += 1
        return counts

    def encode(self, texts: Sequence[str], *, is_query: bool = False) -> Tensor:
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok, tf in self._tokens(text).items():
                idx, sign = _hash_token(tok, self._dim)
                out[i, idx] += sign * (1.0 + math.log(tf))
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        np.divide(out, norms, out=out, where=norms > 0)
        return torch.from_numpy(out)


def _dtype_kwarg() -> str:
    import transformers

    major, minor = (int(p) for p in transformers.__version__.split(".")[:2])
    return "dtype" if (major, minor) >= (4, 56) else "torch_dtype"


class TransformersHiddenStateEncoder:
    """Pooled hidden states of a Hugging Face model (encoder or decoder), CPU by default.

    Retrieval quality is much higher with a contrastively trained embedding model
    (e.g. Qwen/Qwen3-Embedding-0.6B with pooling="last") than with raw hidden states
    of a generative coder model.
    """

    def __init__(
        self,
        model_name_or_path: str,
        *,
        layer: int | None = None,
        pooling: Literal["mean", "last"] = "mean",
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        max_length: int = 512,
        batch_size: int = 8,
        query_prefix: str = "",
        revision: str | None = None,
        padding_side: Literal["right", "left"] = "right",
    ) -> None:
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("TransformersHiddenStateEncoder requires `pip install transformers`") from exc
        if device.startswith("cuda"):
            warnings.warn("navigator encoder on CUDA consumes VRAM reserved to the compute engine", stacklevel=2)
        if pooling not in ("mean", "last"):
            raise ValueError("pooling must be 'mean' or 'last'")
        self._name = model_name_or_path
        self._revision = revision
        self._layer = layer
        self._pooling = pooling
        self._device = torch.device(device)
        self._max_length = max_length
        self._batch_size = batch_size
        self._query_prefix = query_prefix
        # Right padding is exact for any causal model: real tokens keep their true positions and never see
        # pads, so a chunk's vector is independent of the batch it was encoded in (no spurious surprise on
        # resync). Left padding without position_ids shifts positions: harmless up to float error for RoPE
        # models (scores depend on relative offsets), wrong for absolute-position encoders.
        self._tok = AutoTokenizer.from_pretrained(model_name_or_path, revision=revision, padding_side=padding_side)
        self._tok.padding_side = padding_side
        if self._tok.pad_token is None:
            self._tok.pad_token = self._tok.eos_token
        self._model = AutoModel.from_pretrained(
            model_name_or_path, revision=revision, low_cpu_mem_usage=True, **{_dtype_kwarg(): dtype}
        ).to(self._device)
        self._model.eval()
        self._dim = int(self._model.config.hidden_size)

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def device(self) -> str:
        return str(self._device)

    def release(self) -> None:
        """Free the model (RAM, or VRAM when it ran on CUDA for a bulk sync). The encoder is unusable afterwards."""
        self._model = None  # type: ignore[assignment]
        gc.collect()
        if self._device.type == "cuda":
            torch.cuda.empty_cache()

    @property
    def fingerprint(self) -> str:
        # device/dtype are deliberately excluded: a bulk sync on GPU (fp16) and incremental syncs on CPU (fp32)
        # produce vectors equal up to rounding, so they share one store.
        return (
            f"hf:{self._name}@{self._revision or 'main'}:layer={self._layer}:pool={self._pooling}"
            f":max={self._max_length}:qp={hashlib.blake2b(self._query_prefix.encode(), digest_size=4).hexdigest()}"
        )

    @torch.inference_mode()
    def encode(self, texts: Sequence[str], *, is_query: bool = False) -> Tensor:
        if self._model is None:
            raise RuntimeError("encoder was released")
        # Length-bucketed batches: padding to the longest member wastes compute quadratically in attention and
        # linearly elsewhere; sorting is free because right padding makes every vector batch-independent.
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        ordered = [texts[i] for i in order]
        outs: list[Tensor] = []
        for start in range(0, len(ordered), self._batch_size):
            batch = list(ordered[start : start + self._batch_size])
            if is_query and self._query_prefix:
                batch = [self._query_prefix + t for t in batch]
            enc = self._tok(batch, padding=True, truncation=True, max_length=self._max_length, return_tensors="pt")
            enc = {k: v.to(self._device) for k, v in enc.items()}
            out = self._model(**enc, output_hidden_states=self._layer is not None)
            hs = out.last_hidden_state if self._layer is None else out.hidden_states[self._layer]
            mask = enc["attention_mask"]
            if self._pooling == "mean":
                m = mask.unsqueeze(-1).to(hs.dtype)
                pooled = (hs * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
            else:
                # Index of the last attended position, independent of padding side.
                last = mask.shape[1] - 1 - torch.flip(mask, dims=[1]).argmax(dim=1)
                pooled = hs[torch.arange(hs.shape[0], device=hs.device), last]
            outs.append(pooled.to(torch.float32).cpu())
        if not outs:
            return torch.zeros((0, self._dim), dtype=torch.float32)
        result = torch.empty((len(texts), self._dim), dtype=torch.float32)
        result[torch.tensor(order, dtype=torch.long)] = torch.cat(outs, dim=0)
        return result


# Verified against the model card (hf://models/Qwen/Qwen3-Embedding-0.6B): Qwen3ForCausalLM backbone,
# hidden 1024, 28 layers, bf16 weights 1.19 GB, last-token pooling, query-side instruction only,
# format "Instruct: {task}\nQuery:{query}".
QWEN3_CODE_QUERY_PREFIX = "Instruct: Given a description of a code change or bug, retrieve the relevant source code\nQuery:"
PRESETS: dict[str, dict[str, object]] = {
    "Qwen/Qwen3-Embedding-0.6B": {"pooling": "last", "query_prefix": QWEN3_CODE_QUERY_PREFIX},
    "Qwen/Qwen3-Embedding-4B": {"pooling": "last", "query_prefix": QWEN3_CODE_QUERY_PREFIX},
}

_DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def build_encoder(spec: str, **options: object) -> HiddenStateEncoder:
    """Encoder from a spec string: "hashing[:dim]" or "hf:<model id or local path>[@revision]".

    HF options (override presets): layer, pooling, device, dtype ("float32"|"bfloat16"), max_length,
    batch_size, query_prefix, padding_side.
    """
    if spec == "hashing" or spec.startswith("hashing:"):
        dim = int(spec.split(":", 1)[1]) if ":" in spec else int(options.get("dim", 1024))  # type: ignore[arg-type]
        return HashingEncoder(dim=dim)
    if not spec.startswith("hf:"):
        raise ValueError(f"unknown encoder spec {spec!r} (use 'hashing[:dim]' or 'hf:<model>[@rev]')")
    name, _, revision = spec[3:].partition("@")
    opts: dict[str, object] = {**PRESETS.get(name, {}), **{k: v for k, v in options.items() if v is not None}}
    opts.pop("dim", None)
    dtype = opts.pop("dtype", "float32")
    if isinstance(dtype, str):
        if dtype not in _DTYPES:
            raise ValueError(f"dtype must be one of {sorted(_DTYPES)}")
        dtype = _DTYPES[dtype]
    return TransformersHiddenStateEncoder(name, revision=revision or None, dtype=dtype, **opts)  # type: ignore[arg-type]
