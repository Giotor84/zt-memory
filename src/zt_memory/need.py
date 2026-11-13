"""NEED protocol: the model may ask for the body of code it only saw as a signature, instead of guessing.

One mechanism for every model and provider. Round 1 sends the focused context (a few whole bodies + a map of
signatures, `[router] preset = "api"`) plus the instruction below. If the answer holds no edit blocks but
`NEED path:Lx-y` lines, zt reads those spans from disk and makes round 2 with a fresh, short prompt: the task, the
model's own notes (at most 2 lines), the requested code, the code it asked to KEEP and the headers of those files.
The round-1 bodies and the map are not sent again, so no provider-specific prompt caching is needed.

A NEED answer must come at once (no analysis, no draft patch) and end with END_NEED, which is also passed as a stop
sequence where the API accepts one. For reasoning models (MiniMax, DeepSeek R1) the stop cuts only the visible
answer: the reasoning before it is still produced.

Rounds are paid calls: `[engine] need_rounds` (default 1 with the api preset, else 0) bounds them, and on Bedrock
every call also counts against `max_calls`. With a traceback in the query the context is already pinned and NEED is
not offered.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from .router import _LANG, RoutedContext
from .spans import SpanPolicy, SpanRefused

END_NEED = "END_NEED"
MAX_NOTE_LINES = 2
NEED_CONTRACT = (
    "Some code is listed under \"Other relevant code\" only as a signature with its line range. If the change belongs in "
    "one of those, or you need a body to write an exact SEARCH section, do not guess and do not analyse further: reply "
    "at once, with no explanation and no draft edit, with only these lines:\n"
    "NEED <path>:L<start>-<end>   (up to 3, copied from that list)\n"
    "KEEP <path>:L<start>-<end>   (optional: code shown in full above that you still need)\n"
    "NOTE: <what you found so far>   (at most 2 lines)\n"
    f"{END_NEED}\n"
    "You will then receive that code and answer with the edits. If the code you need is already shown in full, answer "
    "with the edits directly."
)
NEED_LAST_ROUND = "Now answer with the edit blocks, following the output contract exactly. Do not ask for more code."
NEED_MORE_ALLOWED = ("Now answer with the edit blocks, following the output contract exactly. If you still lack one body, "
                     f"you may ask once more with NEED lines ending with {END_NEED}.")

_REQ = re.compile(r"^\s*[-*]?\s*(?P<kind>NEED|KEEP)\s*:?\s+`?(?P<path>[^\s`:]+):L?(?P<start>\d+)\s*-\s*L?(?P<end>\d+)`?", re.M)
_NOTE = re.compile(r"^\s*NOTE\s*:\s*(?P<note>.+)$", re.M)


@dataclass(frozen=True)
class SpanRequest:
    kind: str  # NEED | KEEP
    path: str
    start: int
    end: int

    def label(self) -> str:
        return f"{self.path}:L{self.start}-{self.end}"


@dataclass(frozen=True)
class NeedAnswer:
    needs: tuple[SpanRequest, ...]
    keeps: tuple[SpanRequest, ...]
    note: str


def parse_need(text: str, *, max_requests: int = 3) -> NeedAnswer:
    """NEED/KEEP lines and up to two NOTE lines, read up to END_NEED (anything after it is ignored)."""
    text = text.split(END_NEED, 1)[0]
    needs: list[SpanRequest] = []
    keeps: list[SpanRequest] = []
    for m in _REQ.finditer(text):
        start, end = int(m.group("start")), int(m.group("end"))
        if end < start:
            start, end = end, start
        path = m.group("path").replace("\\", "/")
        req = SpanRequest(m.group("kind"), path[2:] if path.startswith("./") else path, start, end)
        target = needs if req.kind == "NEED" else keeps
        if req not in target and len(target) < max_requests:
            target.append(req)
    notes = [m.group("note").strip()[:200] for m in _NOTE.finditer(text)][:MAX_NOTE_LINES]
    return NeedAnswer(tuple(needs), tuple(keeps), " / ".join(notes))


@dataclass
class NeedSettings:
    rounds: int = 1
    max_requests: int = 3
    max_lines: int = 300  # per requested span
    max_total_lines: int = 600  # all spans of one round


@dataclass
class NeedState:
    """Everything needed to build round 2 for one task."""

    settings: NeedSettings
    repo: Path
    routed: RoutedContext
    task: str
    system_prompt: str
    contract: str  # the edit contract without the NEED part
    policy: SpanPolicy | None = None  # which files may be read (exclude_globs); None = defaults for `repo`
    rounds_left: int = field(init=False)
    asked: list[list[str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.rounds_left = self.settings.rounds

    @property
    def offered(self) -> bool:
        return self.rounds_left > 0

    @property
    def stop(self) -> list[str] | None:
        """Stop sequence for the next call: only while a NEED answer is still possible."""
        return [END_NEED] if self.offered else None

    def read(self, req: SpanRequest, budget: int) -> tuple[str, int, int] | None:
        """(text, start, end) of a span zt may read (see spans.py), at most `budget` lines; None if refused."""
        policy = self.policy or SpanPolicy(self.repo)
        try:
            _, text, start, end, _ = policy.read(req.path, req.start, req.end, max_lines=min(self.settings.max_lines, budget))
        except SpanRefused:
            return None
        return text, start, end

    def resolve(self, answer: NeedAnswer) -> tuple[list[tuple[SpanRequest, str, int, int]], list[tuple[SpanRequest, str, int, int]]]:
        budget = self.settings.max_total_lines
        out: list[list[tuple[SpanRequest, str, int, int]]] = [[], []]
        for i, group in enumerate((answer.needs, answer.keeps)):
            for req in group:
                if budget <= 0:
                    break
                got = self.read(req, budget)
                if got is not None:
                    out[i].append((req, *got))
                    budget -= got[2] - got[1] + 1
        return out[0], out[1]

    def next_messages(self, answer: NeedAnswer, needed: Sequence[tuple[SpanRequest, str, int, int]],
                      kept: Sequence[tuple[SpanRequest, str, int, int]]) -> list[dict[str, Any]]:
        """Round 2: a short new prompt; the round-1 bodies and the map are not sent again."""
        self.rounds_left -= 1
        self.asked.append([r.label() for r, *_ in needed])
        closing = NEED_MORE_ALLOWED if self.rounds_left > 0 else NEED_LAST_ROUND
        paths = {r.path for r, *_ in [*needed, *kept]}
        headers = [b for b in self.routed.blocks if b.kind.name == "FILE" and b.path in paths and b.full]
        notes = "\n".join([*(f"NOTE: {answer.note}",) * bool(answer.note),
                           *(f"NEED {r.label()}" for r in answer.needs), *(f"KEEP {r.label()}" for r in answer.keeps)])
        parts = [f"## Task\n{self.task.strip()}", f"\n## Your notes from the first look at the code\n{notes}"]
        if headers:
            parts.append("\n## File headers\n" + "\n".join(
                f"### {b.path}  L{b.start_line}-{b.end_line}  [header]\n```{_lang(b.path)}\n{b.text}\n```" for b in headers))
        parts.append("\n## Code you asked for (exact spans)\n" + _spans(needed))
        if kept:
            parts.append("\n## Code you asked to keep (exact spans)\n" + _spans(kept))
        contract = self.contract + ("\n\n" + NEED_CONTRACT if self.rounds_left > 0 else "")
        parts.append(f"\n## Output contract\n{contract}\n\n{closing}")
        return [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": "\n".join(parts)}]


def _lang(path: str | None) -> str:
    return _LANG.get(PurePosixPath(path or "").suffix.lower(), "")


def _spans(items: Sequence[tuple[SpanRequest, str, int, int]]) -> str:
    return "\n".join(f"### {r.path}  L{start}-{end}\n```{_lang(r.path)}\n{text}\n```" for r, text, start, end in items)


def offer_need(routed: RoutedContext, settings: NeedSettings | None) -> bool:
    """NEED is offered only when there is a map to ask from and no traceback already pinned the answer."""
    return bool(settings and settings.rounds > 0 and not routed.pinned and any(not b.full for b in routed.blocks))


def need_settings(engine: Mapping[str, Any], router_preset: str) -> NeedSettings | None:
    """[engine] need_rounds: -1 = 1 extra round with the api preset, else 0."""
    rounds = int(engine.get("need_rounds", -1))
    if rounds < 0:
        rounds = 1 if router_preset == "api" else 0
    return NeedSettings(rounds=rounds) if rounds > 0 else None
