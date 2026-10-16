# P1 Phase-0: 联邦全局模型选择与不可靠验证标签

> **项目性质：独立、可运行的科研起始工程（不是 FNBench 官方代码，也没有包含 CIFAR-10 数据）。**
> 这是用于证明或证伪论文假设的实验设施，不是已完成的新算法或毕业论文。

## 当前课题到底研究什么？

一次联邦训练产生多个全局 checkpoint。各客户端验证集的标签噪声率及类别分布不同，服务器根据含噪汇总指标选择 checkpoint，可能选中真实目标分布下性能更差的模型。我们要测的是 **实际选择损失（selection regret）**，以及简单现有策略是否足以避免这一损失。

任务边界：**只做全局 checkpoint 选择**，不做客户端本地 early stopping、超参数优化、动态聚合或新鲁棒模型。**这不是已经确认具有论文创新性的题目。**

## 下载后首先运行

Python 3.10+，建议虚拟环境。**请在项目根目录执行**：

```bash
python -m venv .venv
# Linux / macOS:
source .venv/bin/activate
# Windows PowerShell:
# .venv\\Scripts\\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -q
python scripts/p1_cli.py all --config configs/phase0_smoke.json
```

`synthetic-debug` **只验证工程可运行**，不是论文中的真实训练结论。生成：

```text
outputs/smoke/
  config_used.json
  run_metadata.json
  splits_manifest.json
  train_log.json
  checkpoints/round_0001.pt, round_0002.pt
  validation_scores.csv      # 仅 uniform/heterogeneous；不含干净验证参考
  selected_models.csv        # 只根据 validation_scores 选轮次
  regret_summary.csv         # 事后审计，无泄漏到选择器
  validation_accuracy.png
  private_audit/             # 不准输入选择器（oracle_test.csv 等）
```

## 实际 CIFAR-10 跑法（V100/A100）

> **耗时、下载数据及算力：须经你明确批准才运行。** 本工程默认没有替你运行任何 CIFAR-10 GPU 实验。

```bash
python scripts/p1_cli.py train    --config configs/phase0_v100.json
python scripts/p1_cli.py evaluate --config configs/phase0_v100.json
python scripts/p1_cli.py select   --config configs/phase0_v100.json
python scripts/p1_cli.py audit    --config configs/phase0_v100.json
python scripts/p1_cli.py plot     --config configs/phase0_v100.json
```

真实 run 的缺陷/谨慎事项：本代码为极简 FedAvg baseline，**并非 FNBench 的严格算法复现**。如果用 FNBench、FedNoRo 实现正式论文实验，**必须在原源码中完成数据划分隔离、全局 checkpoint 仪表化和日志同步**。同一模型的三个验证方案共享 checkpoint，不能分别训练再做对照。模型训练时不读取验证/测试标签，只有完成选择后，`audit` 才读取 `private_audit/oracle_test.csv`。

**重要实验控制**：官方 CIFAR-10 test 10k 仅用于冻结规则后的离线审计；训练集 50k 先做 client assignment，再划分互斥 train/validation。不能把 CIFAR-10-C 测试集用于训练。运行 `evaluation` 前训练过程只能看到训练样本，所有 clean validation 标签视图与 test oracle 均在事后计算。可从 configuration 添加 `"training_label_noise_rate":0.2` 引入**单独生成**的训练标签噪声，不要与验证标签噪声混淆。

目标量：`selection_regret_pp = (max_t test_acc_t - test_acc_selected) * 100`。分数中的测试准确率只出现在**事后审计表**，不进入模型选择器。`clean_val_oracle` 在独立 `audit` 模块内计算，**不出现在选择器的输入文件中**，不可作为部署策略。必须比较 `last`、含噪 macro/micro accuracy、含噪 CE 和历史平滑。

## 引入 FNBench 官方仓库

本 ZIP **不包含官方 FNBench 代码**：制作环境无法通过 git 完成外部克隆。Codex 可以在具备 GitHub 网络的电脑/云端环境运行：

```bash
bash scripts/get_fnbench.sh
```

这会将官方代码放到 `vendor/FNBench/`，保存上游 commit 到 `vendor/FNBENCH_PINNED_COMMIT.txt`。上游代码 Apache-2.0，归原作者所有；请遵守上游许可证和标注引用。**不要把我们的小型独立实现伪称为 FNBench 仓库**。FNBench 官方源： https://github.com/Sprinter1999/FNBench

**已核查官方 README/源码**：主要训练入口 `main_fed_LNL.py`。原版强制 CUDA，且使用 test_loader 进行测试评估，因此 Codex 必须审计训练循环和测试结果的使用方式，避免 checkpoint/参数选择泄漏。官方还提供多个 noisy-label 方法，实际参数必须基于克隆版本检查。

## Codex 完整任务

**将整个工程解压并打开为 Codex 工作区；复制 `CODEX_MASTER_PROMPT.md` 的完整文本交给 Codex。**

先完成源码审计、接口测试、2-round smoke test；请不要让 Codex 默认启动正式 120-round V100 训练。

详见 `docs/RESEARCH_PROTOCOL.md`、`docs/FNBENCH_INTEGRATION.md`、`docs/ACCEPTANCE_CHECKLIST.md`。

## Windows 补充

Windows 用户建议先读根目录 `START_HERE_先看这里.md`；`scripts/run_smoke_windows.ps1` 和 `scripts/get_fnbench_windows.ps1` 已附带。不要求为本项目安装 Linux，也无需在电脑 CPU 上启动 ResNet18 的 120 轮正式训练。
