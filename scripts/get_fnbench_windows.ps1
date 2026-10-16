$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Path vendor -Force | Out-Null
if (-not (Test-Path 'vendor/FNBench/.git')) {
  git clone https://github.com/Sprinter1999/FNBench.git vendor/FNBench
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
git -C vendor/FNBench rev-parse HEAD | Out-File -Encoding utf8 vendor/FNBENCH_PINNED_COMMIT.txt
Write-Host "Upstream FNBench fetched; inspect its Apache-2.0 license."
