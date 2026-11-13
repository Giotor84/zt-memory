"""Thin client for the VRAM compute engine (any OpenAI-compatible server: llama.cpp, vLLM, SGLang),
the factory that picks the backend from `[engine]` (OpenAI-compatible or Amazon Bedrock, see engine_bedrock.py)
and helpers that close the greenfield loop (parse generated files, write them safely)."""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from .router import RoutedContext

if TYPE_CHECKING:
    from .engine_bedrock import BedrockEngineClient

BACKENDS = ("openai", "bedrock")


class ComputeEngineError(RuntimeError):
    """Engine unreachable or unusable answer. `empty_response`: the model returned no text;
    `max_tokens_exhausted`: because the output budget ran out; `aws_code`: Bedrock/botocore error code."""

    def __init__(self, message: str, *, empty_response: bool = False, max_tokens_exhausted: bool = False,
                 aws_code: str | None = None) -> None:
        super().__init__(message)
        self.empty_response = empty_response
        self.max_tokens_exhausted = max_tokens_exhausted
        self.aws_code = aws_code


@dataclass(frozen=True)
class GeneratedFile:
    path: str
    content: str


_FENCE = re.compile(r"^```[^\n`]*?\bpath=(?P<path>[^\s`]+)[^\n]*\n(?P<body>.*?)^```[ \t]*$", re.MULTILINE | re.DOTALL)


def parse_generated_files(text: str) -> list[GeneratedFile]:
    """Extract ```<lang> path=rel/file.ext blocks (greenfield output contract)."""
    out: dict[str, GeneratedFile] = {}
    for m in _FENCE.finditer(text):
        path = m.group("path").strip().strip("'\"")
        out[path] = GeneratedFile(path, m.group("body"))
    return list(out.values())


def write_generated_files(files: Sequence[GeneratedFile], repo_root: Path, *, overwrite: bool = True) -> list[str]:
    """Write files under repo_root; rejects absolute paths and traversal. Returns relative paths written."""
    root = Path(repo_root).resolve()
    written: list[str] = []
    for f in files:
        rel = PurePosixPath(f.path)
        if rel.is_absolute() or ".." in rel.parts or not rel.parts:
            raise ValueError(f"unsafe path from model output: {f.path!r}")
        target = (root / rel).resolve()
        if root not in target.parents:
            raise ValueError(f"path escapes repo_root: {f.path!r}")
        if target.exists() and not overwrite:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(f.content if f.content.endswith("\n") else f.content + "\n", encoding="utf-8")
        tmp.replace(target)
        written.append(str(rel))
    return written


@dataclass(frozen=True)
class ChatResult:
    text: str
    prompt_tokens: int | None
    completion_tokens: int | None
    finish_reason: str | None
    seconds: float

    @property
    def tokens_per_second(self) -> float | None:
        if not self.completion_tokens or self.seconds <= 0:
            return None
        return self.completion_tokens / self.seconds


class ComputeEngineClient:
    """OpenAI-compatible chat endpoint plus llama-server extras (/health, /props, /tokenize) when available."""

    def __init__(self, base_url: str = "http://127.0.0.1:8080/v1", *, model: str = "local", api_key: str | None = None, timeout_s: float = 900.0) -> None:
        base = base_url.rstrip("/")
        self._url = base + "/chat/completions"
        self._root = base[: -len("/v1")] if base.endswith("/v1") else base
        self._model = model
        self._api_key = api_key
        self._timeout = timeout_s
        self.supports_stop = True  # set to False the first time the server rejects "stop"

    # -- low level ----------------------------------------------------------------------------
    def _request(self, url: str, body: dict[str, object] | None = None, *, timeout: float | None = None) -> dict:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(url, data=data, headers=headers, method="POST" if body is not None else "GET")
        try:
            with urllib.request.urlopen(req, timeout=timeout or self._timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise ComputeEngineError(f"HTTP {exc.code} from compute engine: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError) as exc:
            raise ComputeEngineError(f"compute engine unreachable or invalid response at {url}: {exc}") from exc

    # -- llama-server extras ------------------------------------------------------------------
    def health(self) -> bool:
        try:
            return self._request(self._root + "/health", timeout=5).get("status") == "ok"
        except ComputeEngineError:
            return False

    def context_size(self) -> int | None:
        """Per-slot context of a llama-server (None for servers without /props)."""
        try:
            props = self._request(self._root + "/props", timeout=10)
        except ComputeEngineError:
            return None
        n = (props.get("default_generation_settings") or {}).get("n_ctx") or props.get("n_ctx")
        return int(n) if n else None

    def count_tokens(self, text: str) -> int | None:
        """Exact token count with the served model's tokenizer (None if /tokenize is unavailable)."""
        try:
            res = self._request(self._root + "/tokenize", {"content": text}, timeout=60)
        except ComputeEngineError:
            return None
        tokens = res.get("tokens")
        return len(tokens) if isinstance(tokens, list) else None

    # -- chat ---------------------------------------------------------------------------------
    def chat_full(self, messages: list[dict[str, str]], *, max_tokens: int = 2048, temperature: float = 0.2,
                  stop: Sequence[str] | None = None) -> ChatResult:
        clean = [{"role": m["role"], "content": m["content"]} for m in messages]  # only standard keys leave zt
        body: dict[str, object] = {"model": self._model, "messages": clean, "max_tokens": max_tokens, "temperature": temperature}
        if stop and self.supports_stop:
            body["stop"] = list(stop)
        t0 = time.perf_counter()
        try:
            payload = self._request(self._url, body)
        except ComputeEngineError as exc:
            if not ("stop" in body and str(exc).startswith("HTTP 400") and "stop" in str(exc).lower()):
                raise
            self.supports_stop = False  # e.g. reasoning models that reject stop sequences: drop them from now on
            del body["stop"]
            payload = self._request(self._url, body)
        seconds = time.perf_counter() - t0
        try:
            choice = payload["choices"][0]
            text = str(choice["message"]["content"] or "")
        except (KeyError, IndexError, TypeError) as exc:
            raise ComputeEngineError(f"unexpected response shape: {str(payload)[:300]}") from exc
        usage = payload.get("usage") or {}
        return ChatResult(text, usage.get("prompt_tokens"), usage.get("completion_tokens"), choice.get("finish_reason"), seconds)

    def chat(self, messages: list[dict[str, str]], *, max_tokens: int = 2048, temperature: float = 0.2, stop: Sequence[str] | None = None) -> str:
        return self.chat_full(messages, max_tokens=max_tokens, temperature=temperature, stop=stop).text

    def complete(self, routed: RoutedContext, *, system_prompt: str | None = None, max_tokens: int = 2048, temperature: float = 0.2) -> str:
        return self.chat(routed.to_chat_messages(system_prompt), max_tokens=max_tokens, temperature=temperature)


def make_engine_client(settings: Mapping[str, Any], *, usage_log: Path | None = None) -> "ComputeEngineClient | BedrockEngineClient":
    """The client for `[engine]` settings (ENGINE_DEFAULTS merged with the project's values).

    backend "openai" (default): ComputeEngineClient(url, model, timeout_s) as before. backend "bedrock": the Converse
    client; `url` is ignored, `ctx` falls back to the model profile, calls are limited by `max_calls` and logged to
    `usage_log` (usually .zt/engine_usage.jsonl)."""
    backend = str(settings.get("backend") or "openai")
    if backend == "openai":
        return ComputeEngineClient(str(settings["url"]), model=str(settings["model"]), timeout_s=float(settings["timeout_s"]))
    if backend == "bedrock":
        from .engine_bedrock import BedrockEngineClient

        ctx = settings.get("ctx")
        return BedrockEngineClient(
            str(settings["model"]), region=str(settings.get("region") or "eu-central-1"), proxy=str(settings.get("proxy") or ""),
            ctx=int(ctx) if ctx else None, timeout_s=float(settings["timeout_s"]),
            aws_retries=int(settings.get("aws_retries") or 3), max_calls=int(settings.get("max_calls") or 20),
            usage_log=usage_log)
    raise ComputeEngineError(f"unknown [engine] backend {backend!r}; choose from {', '.join(BACKENDS)}")

