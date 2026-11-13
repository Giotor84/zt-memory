# Starts llama-server for zt-memory: all layers on GPU, flash attention, 8-bit KV cache, one slot.
# VRAM estimate (Qwen2.5-Coder-14B Q4_K_M): weights 8.4 GiB + KV q8_0 ~0.1 MiB/token + ~1 GiB buffers
#   -Ctx 16384 -> ~11 GiB   |   -Ctx 12288 -> ~10.6 GiB      (check with: nvidia-smi)
# The context must hold prompt AND answer: zt gives the router (ctx - max_new_tokens - 128) tokens.
param(
    [string]$Model = "C:\models\qwen2.5-coder-14b-instruct-q4_k_m.gguf",
    [string]$LlamaDir = "C:\llama",
    [int]$Ctx = 16384,
    [int]$Port = 8080
)
$ErrorActionPreference = "Stop"
if (-not (Test-Path $Model)) { throw "model not found: $Model (run setup_llama.ps1)" }
# -fa takes a value in current builds: a bare "-fa" is a parse error.
# -np 1: a single slot owns the whole context (the default "auto" creates 4 slots).
& "$LlamaDir\llama-server.exe" -m $Model -ngl 99 -fa on -ctk q8_0 -ctv q8_0 -c $Ctx -np 1 --host 127.0.0.1 --port $Port
