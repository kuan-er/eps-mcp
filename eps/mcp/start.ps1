$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw "Missing isolated Python environment: $python" }
& $python (Join-Path $PSScriptRoot 'server.py')
exit $LASTEXITCODE
