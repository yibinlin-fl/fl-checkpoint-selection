$ErrorActionPreference = "Stop"
Push-Location (Join-Path $PSScriptRoot '..')
try {
  python scripts/prepare_fnbench.py
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
} finally { Pop-Location }
