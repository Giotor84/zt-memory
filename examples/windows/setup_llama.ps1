# Installs the newest llama.cpp CUDA build for Windows and downloads Qwen2.5-Coder-14B-Instruct Q4_K_M.
# Usage (PowerShell 5.1+):  .\examples\windows\setup_llama.ps1
# If Windows blocks the script: Unblock-File .\examples\windows\*.ps1
param(
    [string]$LlamaDir = "C:\llama",
    [string]$ModelDir = "C:\models",
    [ValidateSet("12.4", "13.4")][string]$Cuda = "12.4"
)
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"   # Invoke-WebRequest is ~10x faster without the progress bar

Write-Host "== GPU / driver"
if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) { throw "nvidia-smi not found: install the NVIDIA driver first" }
& nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used --format=csv,noheader | ForEach-Object { Write-Host "  $_" }

Write-Host "== llama.cpp (CUDA $Cuda)"
# A new release is published before its CI jobs upload the binaries: pick the newest release that already
# carries both Windows CUDA zips instead of trusting /releases/latest.
$headers = @{ "User-Agent" = "zt-memory-setup"; "Accept" = "application/vnd.github+json" }
$releases = Invoke-RestMethod "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=15" -Headers $headers
$pick = $null
foreach ($r in $releases) {
    $bin = $r.assets | Where-Object { $_.name -eq "llama-$($r.tag_name)-bin-win-cuda-$Cuda-x64.zip" } | Select-Object -First 1
    $rt = $r.assets | Where-Object { $_.name -eq "cudart-llama-bin-win-cuda-$Cuda-x64.zip" } | Select-Object -First 1
    if ($bin -and $rt) { $pick = @{ tag = $r.tag_name; bin = $bin; rt = $rt }; break }
}
if (-not $pick) {
    $names = ($releases[0].assets | Where-Object { $_.name -like "*win-cuda*" } | ForEach-Object { $_.name }) -join ", "
    throw "no recent release has win-cuda-$Cuda zips (latest offers: $names). Try -Cuda 13.4"
}
Write-Host "  release $($pick.tag): $($pick.bin.name) + $($pick.rt.name)"
New-Item -ItemType Directory -Force $LlamaDir, $ModelDir | Out-Null
Invoke-WebRequest $pick.bin.browser_download_url -OutFile "$LlamaDir\llama.zip" -Headers @{ "User-Agent" = "zt-memory-setup" }
Invoke-WebRequest $pick.rt.browser_download_url -OutFile "$LlamaDir\cudart.zip" -Headers @{ "User-Agent" = "zt-memory-setup" }
Expand-Archive "$LlamaDir\llama.zip" -DestinationPath $LlamaDir -Force
Expand-Archive "$LlamaDir\cudart.zip" -DestinationPath $LlamaDir -Force   # CUDA runtime DLLs next to the exe
Remove-Item "$LlamaDir\llama.zip", "$LlamaDir\cudart.zip"
& "$LlamaDir\llama-server.exe" --version

Write-Host "== model (8.99 GB)"
$model = Join-Path $ModelDir "qwen2.5-coder-14b-instruct-q4_k_m.gguf"
if (Test-Path $model) {
    Write-Host "  already present: $model"
} else {
    if (Get-Command hf -ErrorAction SilentlyContinue) {
        hf download Qwen/Qwen2.5-Coder-14B-Instruct-GGUF qwen2.5-coder-14b-instruct-q4_k_m.gguf --local-dir $ModelDir
    } else {
        # no huggingface CLI in this window: curl.exe ships with Windows 10+, -C - resumes an interrupted download
        $url = "https://huggingface.co/Qwen/Qwen2.5-Coder-14B-Instruct-GGUF/resolve/main/qwen2.5-coder-14b-instruct-q4_k_m.gguf"
        & curl.exe -L --fail -C - -o $model $url
        if ($LASTEXITCODE -ne 0) { throw "model download failed (curl exit $LASTEXITCODE); rerun to resume" }
    }
}
$size = (Get-Item $model).Length
if ($size -lt 8.9e9) { throw "model file looks incomplete ($size bytes): rerun the script to resume" }
Write-Host "done. start the server with: .\examples\windows\start_llama.ps1"
