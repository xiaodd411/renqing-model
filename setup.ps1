param([switch]$Cpu, [switch]$Causal)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw '需要 uv。请先从 https://docs.astral.sh/uv/getting-started/installation/ 安装，或使用已有 Python 3.11+ 建立虚拟环境。'
}
$env:UV_PYTHON_INSTALL_DIR = Join-Path $PSScriptRoot 'cache\python'
$env:UV_CACHE_DIR = Join-Path $PSScriptRoot 'cache\uv'
if (-not (Test-Path '.venv\Scripts\python.exe')) {
    uv venv --python 3.12 .venv
    if ($LASTEXITCODE -ne 0) { throw '创建虚拟环境失败' }
}
$index = if ($Cpu) { 'https://download.pytorch.org/whl/cpu' } else { 'https://download.pytorch.org/whl/cu128' }
uv pip install --python .venv\Scripts\python.exe torch==2.8.0 --index-url $index
if ($LASTEXITCODE -ne 0) { throw 'PyTorch 安装失败' }
$taskPackage = if ($Causal) { '.[causal]' } else { '.' }
uv pip install --python .venv\Scripts\python.exe -e $taskPackage
if ($LASTEXITCODE -ne 0) { throw '项目依赖安装失败' }
Write-Host '安装完成。运行 .\train.ps1，或使用 .venv\Scripts\renqing.exe。'
