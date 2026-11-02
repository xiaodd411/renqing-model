param([string]$Output = 'runs\renqing-v2', [int]$Seed = 42, [int]$Epochs = 8)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$taskPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) { $taskPython = Join-Path $PSScriptRoot '..\..\work\.venv\Scripts\python.exe' }
if (-not (Test-Path -LiteralPath $taskPython)) { throw '请先运行 .\setup.ps1' }
$env:PYTHONUTF8 = '1'
$env:HF_HOME = Join-Path $PSScriptRoot 'cache\huggingface'
$env:HF_HUB_DISABLE_TELEMETRY = '1'
if (-not (Test-Path 'data\challenge_v2.jsonl')) {
    & $taskPython scripts\build_challenge_v2.py
    if ($LASTEXITCODE -ne 0) { throw '构建挑战集失败' }
}
if (-not (Test-Path 'data\expanded-v2\manifest.json')) {
    & $taskPython scripts\prepare_expanded.py
    if ($LASTEXITCODE -ne 0) { throw '准备扩展数据失败' }
}
& $taskPython -m renqing.cli validate --data data\expanded-v2
if ($LASTEXITCODE -ne 0) { throw '扩展数据校验失败' }
& $taskPython -m renqing.cli train --data data\expanded-v2 --output $Output --epochs $Epochs --patience 3 --batch-size 24 --lr 2e-5 --balance-tasks --seed $Seed
if ($LASTEXITCODE -ne 0) { throw '训练失败' }
