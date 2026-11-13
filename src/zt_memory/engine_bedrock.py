"""Amazon Bedrock backend (Converse API) with the same interface as ComputeEngineClient.

Used by `zt code`, `zt engine` and the bug-fix benchmark when `[engine] backend = "bedrock"`. Credentials come
from the environment (instance IAM role, AWS_* variables): nothing is read from the zt config or stored here.

Cost rules: no call to the model happens implicitly (health() only builds the boto3 client), every call counts
against a per-process limit (`max_calls`) and is appended to a JSONL usage log. Reasoning models (MiniMax M2.5)
return `reasoningContent` blocks before the answer: only the `text` blocks reach the caller, so the reasoning
can never be parsed as SEARCH/REPLACE edits.

boto3 is optional (`pip install "zt-memory[bedrock]"`) and imported only when a client is built.
"""
from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .engine_client import ChatResult, ComputeEngineError
from .router import RoutedContext

CHARS_PER_TOKEN = 3.0  # conservative for code: Bedrock exposes no tokenizer
ESTIMATE_MARGIN = 1.10
PROBE_PROMPT = "Reply only with: OK"
PROBE_MAX_TOKENS = 128  # with 16, MiniMax spends the whole budget reasoning and the text stays empty
_REGION_PREFIXES = ("us.", "eu.", "apac.", "global.", "us-gov.")  # cross-region inference profile IDs
_FINISH = {"end_turn": "stop", "max_tokens": "length"}  # other stop reasons are passed through unchanged
MISSING_BOTO3 = ("boto3 is not installed: install the bedrock extra (pip install \"zt-memory[bedrock]\", "
                 "offline: from the wheelhouse) to use [engine] backend = \"bedrock\"")


class EngineBudgetExceeded(ComputeEngineError):
    """The per-process call limit was reached: stop, do not record it as a failed attempt."""

    stop_run = True


@dataclass(frozen=True)
class ModelProfile:
    name: str
    context: int | None  # None: take [engine] ctx
    max_output_tokens: int
    reasoning: bool
    extra_fields: dict[str, Any] = field(default_factory=dict)
    retry_empty_response: bool = False  # one more try when the text is empty but the budget was not exhausted


# Closed table: a model not listed here gets the generic profile (no extra request fields).
PROFILES: tuple[tuple[str, ModelProfile], ...] = (
    ("minimax.", ModelProfile("minimax-m2.5", 196_000, 8192, reasoning=True, retry_empty_response=True)),
    ("qwen.", ModelProfile("qwen3", 256_000, 8192, reasoning=False, extra_fields={"enable_thinking": False})),
    ("anthropic.", ModelProfile("claude", 200_000, 8192, reasoning=False)),
)
GENERIC_PROFILE = ModelProfile("generic", None, 8192, reasoning=False)


def model_profile(model_id: str) -> ModelProfile:
    mid = model_id.strip().lower()
    for prefix in _REGION_PREFIXES:
        if mid.startswith(prefix):
            mid = mid[len(prefix):]
            break
    return next((p for start, p in PROFILES if mid.startswith(start)), GENERIC_PROFILE)


class _CallCounter:
    """Calls to Bedrock made by this process (all clients together)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.count = 0

    def take(self, limit: int) -> int:
        with self._lock:
            if self.count >= limit:
                raise EngineBudgetExceeded(
                    f"max_calls reached: {self.count} Bedrock call(s) in this process, limit {limit} "
                    "(raise [engine] max_calls if this is intended)")
            self.count += 1
            return self.count

    def reset(self) -> None:  # tests
        with self._lock:
            self.count = 0


CALLS = _CallCounter()


class BedrockEngineClient:
    """Converse API client exposing health / context_size / count_tokens / chat_full / chat / complete."""

    def __init__(self, model: str, *, region: str = "eu-central-1", proxy: str = "", ctx: int | None = None,
                 timeout_s: float = 900.0, aws_retries: int = 3, max_calls: int = 20, usage_log: Path | None = None,
                 client: Any = None, boto3_module: Any = None, config_class: Any = None) -> None:
        """`client`, `boto3_module`, `config_class`: injectable fakes for tests (no network, no boto3 needed)."""
        self.model = model
        self.profile = model_profile(model)
        self.region = region
        self.proxy = proxy.strip()
        self.ctx = int(ctx) if ctx else None
        self.timeout_s = float(timeout_s)
        self.aws_retries = int(aws_retries)
        self.max_calls = int(max_calls)
        self.usage_log = Path(usage_log) if usage_log else None
        self.supports_stop = True  # set to False the first time the model rejects stopSequences
        self.last_error: str | None = None
        self._client = client
        self._boto3 = boto3_module
        self._config_class = config_class

    # -- client -------------------------------------------------------------------------------
    def _bedrock(self) -> Any:
        if self._client is not None:
            return self._client
        boto3, config_class = self._boto3, self._config_class
        if boto3 is None or config_class is None:
            try:
                import boto3 as _boto3  # noqa: PLC0415 - optional dependency
                from botocore.config import Config as _Config  # noqa: PLC0415
            except ImportError as exc:
                raise ComputeEngineError(MISSING_BOTO3) from exc
            boto3, config_class = boto3 or _boto3, config_class or _Config
        cfg = config_class(
            proxies={"http": self.proxy, "https": self.proxy} if self.proxy else None,  # None: HTTPS_PROXY or direct
            connect_timeout=10, read_timeout=self.timeout_s,
            retries={"max_attempts": self.aws_retries, "mode": "standard"},  # backoff on throttling
        )
        self._client = boto3.client("bedrock-runtime", region_name=self.region, config=cfg)
        return self._client

    def health(self) -> bool:
        """True when boto3 is importable and the client could be built. Never calls the model."""
        try:
            self._bedrock()
        except ComputeEngineError as exc:
            self.last_error = str(exc)
            return False
        except Exception as exc:  # noqa: BLE001 - e.g. invalid region or proxy URL from botocore
            self.last_error = f"cannot create the bedrock-runtime client: {type(exc).__name__}: {exc}"
            return False
        self.last_error = None
        return True

    # -- sizes --------------------------------------------------------------------------------
    def context_size(self) -> int | None:
        return self.ctx or self.profile.context

    def count_tokens(self, text: str) -> int | None:
        return None  # no tokenizer on Bedrock: callers fall back to estimates

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Conservative estimate when a number is needed: 3 characters per token plus 10%."""
        return math.ceil(round(len(text) / CHARS_PER_TOKEN * ESTIMATE_MARGIN, 6))  # round: 110.00000000000001 is 110

    def output_limit(self, max_tokens: int) -> int:
        return max(1, min(int(max_tokens), self.profile.max_output_tokens))

    # -- chat ---------------------------------------------------------------------------------
    def request(self, messages: Sequence[dict[str, str]], *, max_tokens: int, temperature: float,
                stop: Sequence[str] | None = None) -> dict[str, Any]:
        """The Converse arguments for an OpenAI-style message list (public for inspection and tests)."""
        system = "\n\n".join(m["content"] for m in messages if m.get("role") == "system" and m.get("content"))
        turns: list[dict[str, Any]] = []
        for m in messages:
            role = m.get("role")
            if role == "system":
                continue
            if role not in ("user", "assistant"):
                raise ComputeEngineError(f"unsupported message role for Bedrock: {role!r}")
            text = str(m.get("content") or "")
            if turns and turns[-1]["role"] == role:  # Converse requires alternating roles
                turns[-1]["content"][0]["text"] += "\n\n" + text
            else:
                turns.append({"role": role, "content": [{"text": text}]})
        inference: dict[str, Any] = {"maxTokens": self.output_limit(max_tokens), "temperature": float(temperature)}
        if stop and self.supports_stop:
            inference["stopSequences"] = list(stop)
        req: dict[str, Any] = {"modelId": self.model, "messages": turns, "inferenceConfig": inference}
        if system:
            req["system"] = [{"text": system}]
        if self.profile.extra_fields:
            req["additionalModelRequestFields"] = dict(self.profile.extra_fields)
        return req

    def chat_full(self, messages: list[dict[str, str]], *, max_tokens: int = 2048, temperature: float = 0.2,
                  stop: Sequence[str] | None = None) -> ChatResult:
        req = self.request(messages, max_tokens=max_tokens, temperature=temperature, stop=stop)
        attempts = 2 if self.profile.retry_empty_response else 1
        for attempt in range(attempts):
            try:
                result, stop_reason = self._converse(req)
            except ComputeEngineError as exc:
                if not (exc.aws_code == "ValidationException" and "stop" in str(exc).lower() and "stopSequences" in req["inferenceConfig"]):
                    raise
                self.supports_stop = False  # this model takes no stop sequences: drop them from now on
                req["inferenceConfig"].pop("stopSequences")
                result, stop_reason = self._converse(req)
            if result.text.strip():
                return result
            if stop_reason == "max_tokens":  # retrying with the same budget only spends money again
                raise ComputeEngineError(
                    f"empty answer from {self.model}: the output budget ({req['inferenceConfig']['maxTokens']} tokens) "
                    "was used up before any text (reasoning models think first): aumenta max_new_tokens",
                    empty_response=True, max_tokens_exhausted=True)
            if attempt + 1 < attempts:
                continue
        raise ComputeEngineError(f"empty answer from {self.model} (stopReason {stop_reason})", empty_response=True)

    def _converse(self, req: dict[str, Any]) -> tuple[ChatResult, str | None]:
        client = self._bedrock()
        CALLS.take(self.max_calls)
        t0 = time.perf_counter()
        try:
            resp = client.converse(**req)
        except Exception as exc:  # noqa: BLE001 - mapped below; programming errors are re-raised
            code = _aws_code(exc)
            if code is None:
                raise
            self._log(seconds=time.perf_counter() - t0, error=code)
            raise ComputeEngineError(f"Bedrock call failed ({code}): {exc}", aws_code=code) from exc
        seconds = time.perf_counter() - t0
        blocks = ((resp.get("output") or {}).get("message") or {}).get("content") or []
        text = "".join(str(b["text"]) for b in blocks if isinstance(b, dict) and "text" in b)
        reasoning = sum(len(str(((b.get("reasoningContent") or {}).get("reasoningText") or {}).get("text") or ""))
                        for b in blocks if isinstance(b, dict) and "reasoningContent" in b)
        usage = resp.get("usage") or {}
        stop_reason = resp.get("stopReason")
        self._log(seconds=seconds, input_tokens=usage.get("inputTokens"), output_tokens=usage.get("outputTokens"),
                  stop_reason=stop_reason, reasoning_chars=reasoning)
        finish = _FINISH.get(stop_reason, stop_reason) if stop_reason else None
        return ChatResult(text, usage.get("inputTokens"), usage.get("outputTokens"), finish, seconds), stop_reason

    def chat(self, messages: list[dict[str, str]], *, max_tokens: int = 2048, temperature: float = 0.2,
             stop: Sequence[str] | None = None) -> str:
        return self.chat_full(messages, max_tokens=max_tokens, temperature=temperature, stop=stop).text

    def complete(self, routed: RoutedContext, *, system_prompt: str | None = None, max_tokens: int = 2048,
                 temperature: float = 0.2) -> str:
        return self.chat(routed.to_chat_messages(system_prompt), max_tokens=max_tokens, temperature=temperature)

    def probe(self) -> str:
        """One minimal paid call (used only by `zt engine --probe`): the model's answer, stripped."""
        return self.chat_full([{"role": "user", "content": PROBE_PROMPT}], max_tokens=PROBE_MAX_TOKENS,
                              temperature=0.0).text.strip()

    # -- usage log ----------------------------------------------------------------------------
    def _log(self, *, seconds: float, input_tokens: int | None = None, output_tokens: int | None = None,
             stop_reason: str | None = None, reasoning_chars: int | None = None, error: str | None = None) -> None:
        if self.usage_log is None:
            return
        row: dict[str, Any] = {"time": datetime.now(timezone.utc).isoformat(timespec="seconds"), "model": self.model,
                               "input_tokens": input_tokens, "output_tokens": output_tokens, "stop_reason": stop_reason,
                               "seconds": round(seconds, 3)}
        if reasoning_chars:
            row["reasoning_chars"] = reasoning_chars
        if error:
            row["error"] = error
        try:
            self.usage_log.parent.mkdir(parents=True, exist_ok=True)
            with self.usage_log.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")
        except OSError:
            pass  # the log is an audit aid; a read-only folder must not lose the answer already paid for

    def describe(self) -> dict[str, Any]:
        p = self.profile
        return {"backend": "bedrock", "model": self.model, "profile": p.name, "region": self.region,
                "proxy": self.proxy or None, "context": self.context_size(), "max_output_tokens": p.max_output_tokens,
                "reasoning": p.reasoning, "extra_fields": dict(p.extra_fields), "timeout_s": self.timeout_s,
                "aws_retries": self.aws_retries, "max_calls": self.max_calls, "calls_made": CALLS.count,
                "usage_log": str(self.usage_log) if self.usage_log else None}


def _aws_code(exc: BaseException) -> str | None:
    """AWS error code for botocore errors (ClientError, connection/proxy/timeout errors), else None."""
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        code = (response.get("Error") or {}).get("Code")
        if code:
            return str(code)
    module = type(exc).__module__ or ""
    if module.startswith(("botocore", "urllib3")):
        return type(exc).__name__
    return None
