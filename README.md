# zt-memory

**In italiano:** [README.it.md](README.it.md).

**Local code-context engine for LLMs.** Indexes a repository by symbol (functions, classes, methods via
tree-sitter) and, for each question, returns only the code spans that matter — with exact paths and line ranges,
within a token budget. Retrieval never generates tokens: it fuses lexical search (SQLite FTS5/BM25) with embeddings
and adds a small bonus for recently edited or retrieved symbols (the "energy"). Everything runs locally: index on
SSD, encoder on CPU, no external service.

Built for the case where it is measured to help most: **making local and air-gapped models useful on your code**.

[![tests](https://github.com/Giotor84/zt-memory/actions/workflows/tests.yml/badge.svg)](https://github.com/Giotor84/zt-memory/actions/workflows/tests.yml)
![license](https://img.shields.io/badge/license-Apache--2.0-blue)
![python](https://img.shields.io/badge/python-3.11%2B-blue)

| use | command | for |
|---|---|---|
| **MCP server** | `zt mcp` | Claude Code, Cursor, Cline, local agents: *where to look* (files, lines, signatures) or the exact code → [docs/MCP.md](docs/MCP.md) |
| **local edit loop** | `zt code "..."` | a 14B model on a 16 GB GPU: context → verified SEARCH/REPLACE patch → tests → memory update → [docs/ENGINE_WINDOWS.md](docs/ENGINE_WINDOWS.md) |
| **large models over an API** | `[router] preset = "api"` + `zt code` | focused context + signature map; the model asks for missing code with `NEED` → [docs/NEED.md](docs/NEED.md) |
| **Amazon Bedrock** | `[engine] backend = "bedrock"` | environments with no GPU and no internet (e.g. SageMaker behind a proxy) → [docs/BEDROCK.md](docs/BEDROCK.md) |
| **CLI / library** | `zt route "..."` | a ready prompt for any OpenAI-compatible LLM |

## What is measured (and what is not)

Every number below comes from a script in `examples/`; reports are in `docs/` (in Italian for now).
Negative results are kept on purpose.

| measurement | result |
|---|---|
| right file in the context of a **local 14B model** | **93–100%** with zt vs 21–29% with whole files → [RELAZIONE](docs/RELAZIONE.md) |
| bug location in context, **150 real bugs** in 6 repos (Python, Go, TypeScript) | **zt 61%**, BM25 on symbols 58%, BM25 on whole files 19% (p < 0.0001), with 29–31% fewer tokens → [BENCHMARK_LOCALIZE](docs/BENCHMARK_LOCALIZE.md) |
| question = pytest output of failing tests, 42 real bugs (attrs, click, flask) | zt 60% vs 52% without traceback parsing → [BENCHMARK_TRACEBACK](docs/BENCHMARK_TRACEBACK.md) |
| questions in Italian with no code names (same 150 bugs) | zt + Qwen3 41% vs BM25 27% at equal tokens; translating the question to English helps more than the encoder |
| hybrid vs lexical only (requests, 91 questions from git history) | recall@10 0.845 vs 0.823 |
| full loop on an RTX 4060 Ti 16 GB (Qwen2.5-Coder-14B Q4_K_M, 16k ctx) | bug fixed on first try, tests green, ~10 GB VRAM, ~27 tok/s, 14 s total → `examples/e2e_fix_bug.py` |
| bugs actually **fixed** by the local 14B (14 click bugs) | **1/14** with zt, 0/14 with BM25 — zt delivers the context, the model is the limit → [BENCHMARK_BUGFIX](docs/BENCHMARK_BUGFIX.md) |
| **Claude Code** with vs without zt over MCP (45 real bugs, 137 sessions) | same quality; with `mode="locations"` (the default) cost −3%; with 4000 tokens of code +16/24%. **With a strong cloud agent zt does not save money** → [BENCHMARK_CLAUDE_CODE](docs/BENCHMARK_CLAUDE_CODE.md) |

## Install

```bash
# Optional on Linux/Windows without GPU: install lightweight CPU PyTorch first (~200 MB vs ~2.5 GB CUDA)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install "zt-memory[hf,treesitter] @ git+https://github.com/Giotor84/zt-memory"   # + ",bedrock" for Amazon Bedrock
cd your-repo
zt init --encoder hf:Qwen/Qwen3-Embedding-0.6B --rank 128   # writes .zt/config.toml, adds .zt/ to git excludes
zt sync                                                      # first run indexes everything, then only changed symbols
```

Keep files an agent must never read out of the index **before** the first sync:

```toml
# .zt/config.toml
[memory]
exclude_globs = "secrets.py, tests/fixtures/*.json"
```

### Use it from Claude Code (MCP)

```bash
claude mcp add zt-memory -- zt mcp      # finds .zt/ walking up from the session folder
```

The default `locations` mode returns files, line ranges, signatures and docstrings (≤ 500 tokens); the client then
reads only the lines it needs.

## Commands

```bash
zt sync | zt status | zt watch                  # incremental index, status, polling while editing
zt route "fix retry logic on 503"               # prompt with the selected context
zt mcp [--repo PATH] [--mode code]              # MCP server on stdio
zt engine [--probe]                             # LLM server health, context size, speed
zt code "fix retry logic on 503" --check "python -m pytest -q" --revert-on-fail
zt undo                                         # revert the last zt code edit
```

Exit codes: 0 ok, 1 error, 2 usage, 3 store busy, 4 no usable edit, 5 tests failed, 6 LLM server unreachable.

## How it works

```
question ─┬─► encoder (CPU) ► SVD rank r ► k-NN on the store (SSD, memmap) ─────────┐
          └─► code-aware tokens ► FTS5/BM25 (SQLite) ───────────────────────────────┤
              seed expansion: spec/decision ↔ code edges, file headers ◄────────────┤
              RRF: w_g/(k+1+rank_geo) + w_l/(k+1+rank_lex) · kind weight · (1 + w_E·energy)
              ► dynamic cut ► exact spans from disk (hash-verified) ► token budget ► LLM
```

- **Tracebacks first**: `File "...", line N`, `file.py:N:`, `file.ts:N:C`, `file.go:N` — symbols at those lines
  lead the context, source before tests.
- **Stable symbol identity**: `file::Class.method`, stable when code moves; overloads as `name~hash(signature)`.
- **Incremental sync**: one `stat()` per file; same bytes → no work; then a per-symbol diff.
- **Energy**: edits (embedding surprise) and retrievals heat a symbol, which cools over time (`E_t = γE + ‖v‖`,
  inertia β). Small gain, never negative in our measurements.
- **Patches**: SEARCH/REPLACE blocks, all-or-nothing, exact → trailing-space → indentation-insensitive match;
  backups and `zt undo`.

Hardware budget (16 GB RAM + 16 GB VRAM): 14B Q4_K_M + KV q8_0 16k ≈ 10 GB VRAM; Qwen3-Embedding-0.6B on CPU ≈ 5.4 GB
RSS; 15k chunks at rank 64 ≈ 40 MiB on disk.

## Known limits

- The local 14B fixes few bugs on its own (see above): zt is a context engine, not an autonomous fixer.
- With a strong cloud agent and precise logs, plain `grep` already finds the right file.
- First `zt sync` with Qwen3 on CPU is slow (~2.3 chunks/s): use `--device cuda` with the LLM server stopped.
- Changing encoder or rank rebuilds the store; recorded decisions (SPEC/DECISION) are not rebuilt from code.
- Detailed docs are in Italian ([README.it.md](README.it.md), [docs/](docs/)); English translation in progress.

## Tests

```bash
pip install -e ".[dev,hf]"
pytest
```

## Professional services

zt-memory is free under Apache-2.0. I help teams run code LLMs **on-premises or air-gapped** (local GPUs, Amazon
Bedrock behind a proxy), with index exclusions for regulated data and honest benchmarks on their own repositories.
Get in touch through my [GitHub profile](https://github.com/Giotor84).

## Related projects

- [zt-batch-reader](https://github.com/Giotor84/zt-batch-reader) — open-core closed document extraction via calibrated first-token logprobs & deterministic rules (commercial 284B MoE add-on available)
- [zt-anchor](https://github.com/Giotor84/zt-anchor) — anchoring, abstention and frozen-holdout evaluation for document AI
- [zt-airgap](https://github.com/Giotor84/zt-airgap) — tested template for LLM pipelines on SageMaker with no internet
- All projects: [github.com/Giotor84](https://github.com/Giotor84)

## License

Apache-2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).
