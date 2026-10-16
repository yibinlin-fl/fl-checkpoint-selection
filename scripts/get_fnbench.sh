#!/usr/bin/env bash
set -euo pipefail
mkdir -p vendor
if [[ -d vendor/FNBench/.git ]]; then
  echo "vendor/FNBench already cloned; not overwriting"
else
  git clone https://github.com/Sprinter1999/FNBench.git vendor/FNBench
fi
git -C vendor/FNBench rev-parse HEAD | tee vendor/FNBENCH_PINNED_COMMIT.txt
printf '\nUpstream FNBench is Apache-2.0; see vendor/FNBench/LICENSE.\n'
printf 'The P1 starter is separate code; integration into upstream must be audited.\n'
