$ErrorActionPreference = "Stop"
python -m pytest -q
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python scripts/p1_cli.py all --config configs/phase0_smoke.json
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "Done: outputs/smoke/regret_summary.csv (SYNTHETIC DEBUG ONLY)"
