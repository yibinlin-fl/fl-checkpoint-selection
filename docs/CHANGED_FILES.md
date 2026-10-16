# Phase-0 改动文件

相对原始保留提交 `afa9851d8473343d1bb221a657048e6f38047fd1`，实现提交 `0fe5599af8d7b28e76bdac710046570264455605`：

- `.gitattributes`
- `.gitignore`
- `README.md`
- `configs/fnbench_upstream.json`
- `configs/phase0_fnbench_smoke.json`
- `configs/phase0_smoke.json`
- `configs/phase0_v100.json`
- `configs/phase0_v100_iid.json`
- `configs/phase0_v100_trainnoise20.json`
- `docs/ACCEPTANCE_CHECKLIST.md`
- `docs/CODE_AUDIT.md`
- `docs/DECISION.md`
- `docs/FNBENCH_AUDIT.md`
- `docs/FNBENCH_INTEGRATION.md`
- `docs/V100_RUNBOOK.md`
- `p1/artifacts.py`
- `p1/audit.py`
- `p1/config.py`
- `p1/data.py`
- `p1/evaluation.py`
- `p1/fnbench.py`
- `p1/noise.py`
- `p1/plots.py`
- `p1/selection.py`
- `p1/splits.py`
- `p1/training.py`
- `patches/fnbench-lazy-agnews.patch`
- `pyproject.toml`
- `requirements-v100.txt`
- `requirements.txt`
- `scripts/check_v100.py`
- `scripts/get_fnbench.sh`
- `scripts/get_fnbench_windows.ps1`
- `scripts/p1_cli.py`
- `scripts/prepare_fnbench.py`
- `scripts/record_environment.py`
- `scripts/run_v100.sh`
- `scripts/verify_phase0_outputs.py`
- `tests/conftest.py`
- `tests/test_boundaries.py`
- `tests/test_fnbench.py`
- `tests/test_noise.py`

另有最终报告 `docs/PHASE0_REPORT.md`、本清单和 `docs/evidence/` 下的真实日志/轻量结果快照。没有将大型checkpoint或vendor副本加入主仓库；按固定SHA与patch重建vendor。原ZIP已有 outputs/smoke 完整保留。
