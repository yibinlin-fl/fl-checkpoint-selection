# P1 Phase-0：联邦 checkpoint 选择审计

正式任务书为 `CODEX_MASTER_PROMPT.md`。只研究同一训练轨迹的全局 checkpoint 选择；synthetic-debug 只用于工程测试。

当前已获取并审计官方 FNBench，上游 `838dff3023335c015dea07fa13dc42ca971d525d`。保留独立 starter，用官方 FedAvg 组件的外部适配器准备正式实验；没有运行 CIFAR/V100/120轮/多seed/FedNoRo。LICENSE 实际为 MIT，README Apache 徽章不一致，见审计报告。

## 环境与本地验证
Python3.10+；建议独立虚拟环境。本机现有 CPU 环境已可用，没有改装系统包。

```bash
python -m venv .venv
# 激活环境后（Windows 也可直接用 .venv/Scripts/python.exe）
python -m pip install -r requirements.txt
python scripts/prepare_fnbench.py
python -m pytest -q
python scripts/p1_cli.py all --config configs/phase0_smoke.json
python scripts/p1_cli.py all --config configs/phase0_fnbench_smoke.json
```

官方组件需要 scikit-learn。CIFAR 路径的 AGNews import 已通过受控 lazy import 补丁消除 torchtext 依赖；不修改官方 CE、SGD、drop_last、聚合或 BN 行为。获取失败仍可跑独立 starter，官方测试将明确 skip。Windows 不依赖 WSL，也不必更改 PowerShell 执行策略，直接运行 Python 脚本即可。

每次必须使用全新的输出目录；默认目录已有结果时，给新目录，例如：

```bash
python scripts/p1_cli.py all --config configs/phase0_smoke.json --out outputs/my_new_smoke
```

不会批量删除或覆盖旧结果。测试临时文件保留在 `.local_tmp/`，由用户按需手动清理。

## 数据与选择边界
`train -> evaluate(noisy val only) -> select -> audit(test + clean diagnostic) -> plot`。

训练只加载 official train 来源，client assignment 后固定 holdout；训练标签噪声和验证标签噪声独立。验证 uniform 率按 val 样本数加权匹配异质率。同组 checkpoint 共享所有 scenario。selector 只读取 `validation_scores.csv`，字段严格白名单、拒绝 clean/test/noise rate、异常值及残缺网格；EMA 标量来自训练时 config_used。Last 分数留空而不输出 NaN。selector 不读配置文件、manifest、private_audit 或 dataset。

选择结果用 SHA256 冻结；随后 audit 才加载官方 test，计算 `(best_saved_test_acc - selected_test_acc)*100`。clean-val/test-best 是事后诊断，不能部署或选参。文件隔离和哈希属于可复现工程约束，不是恶意访问防护沙箱。

每个输出目录包含 config_used、run_metadata、splits_manifest（原始索引与class hist）、train_log（轮次/参与者/样本量）、training_artifacts（checkpoint SHA256）、checkpoints、validation_scores、selected_models、selection_frozen、regret_summary、validation_accuracy.png，以及 private_audit 下噪声映射、clean diagnostic 与 oracle_test（含test预测hash）。outputs 不纳入 Git，报告和轻量证据纳入 Git。

## 验收与下一步
- `docs/PHASE0_REPORT.md`：实际环境、测试、命令、结果与限制。
- `docs/CODE_AUDIT.md`：核心代码和信息流审计。
- `docs/FNBENCH_AUDIT.md`：上游源码事实、推断与未验证项。
- `docs/V100_RUNBOOK.md`：候选环境安装、正式单seed命令与工作量。
- `docs/DECISION.md`：INCONCLUSIVE，只完成工程验收。

正式 CIFAR 下载和 V100 训练需后续用户授权，**本次到 Phase-0 验收为止**。
