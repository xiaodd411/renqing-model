param([string]$Output = 'runs\renqing-v1', [int]$Epochs = 20)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$taskPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) {
    $taskPython = Join-Path $PSScriptRoot '..\..\work\.venv\Scripts\python.exe'
}
if (-not (Test-Path -LiteralPath $taskPython)) { throw '请先运行 .\setup.ps1' }
$env:HF_HOME = Join-Path $PSScriptRoot 'cache\huggingface'
$env:HF_HUB_DISABLE_TELEMETRY = '1'
$env:PYTHONUTF8 = '1'
if (-not (Test-Path 'data\demo\manifest.json')) {
    & $taskPython scripts\build_data.py
    if ($LASTEXITCODE -ne 0) { throw '生成数据失败' }
}
& $taskPython -m renqing.cli validate --data data\demo
if ($LASTEXITCODE -ne 0) { throw '数据校验失败' }
& $taskPython -m renqing.cli train --data data\demo --output $Output --epochs $Epochs
if ($LASTEXITCODE -ne 0) { throw '训练失败，请查看上述输出' }
