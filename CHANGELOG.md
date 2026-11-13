# Changelog

## 0.4.0 — first public release
- MCP server with `locations` (default), `code` and `focus` modes; optional `zt_read_span` (`--read-span`).
- `zt code` edit loop with llama.cpp or Amazon Bedrock (Converse API, proxy support, call cap).
- `NEED` protocol: the model asks for missing code in one extra short round.
- `exclude_globs` keeps files out of the index, the MCP tools and `NEED`.
- Traceback-aware ranking; symbol-level incremental sync; energy prior.
