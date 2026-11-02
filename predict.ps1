param([string]$Model = 'runs\renqing-v1', [string]$InputFile = 'examples\request.json')
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$taskPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) {
    $taskPython = Join-Path $PSScriptRoot '..\..\work\.venv\Scripts\python.exe'
}
if (-not (Test-Path -LiteralPath $taskPython)) { throw '请先运行 .\setup.ps1' }
$env:HF_HUB_DISABLE_TELEMETRY = '1'
$env:PYTHONUTF8 = '1'
& $taskPython -m renqing.cli predict --model $Model --input $InputFile
if ($LASTEXITCODE -ne 0) { throw '推理失败，请查看上述输出' }
