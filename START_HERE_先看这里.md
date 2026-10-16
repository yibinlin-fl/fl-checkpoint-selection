# 先看这里：你收到的是完整 P1 起始工程，而不是一份空目录

**当前研究候选**：不同客户端验证标签质量不一致时，服务器从多个已经训练的全局 checkpoint 中选模型是否可能产生真实测试性能损失？

## 推荐操作（Windows + Codex）

1. 将整个 ZIP 解压到英文路径，例如 `D:\\research\\P1-FL-Checkpoint-Selection`。
2. 在 Codex 中打开该目录作为**工作区**，不要只上传或粘贴 ZIP 文件名。
3. 将 `CODEX_MASTER_PROMPT.md` 的全文发给 Codex；它明确要求先做源码审计和 CPU smoke，不自动启动昂贵的 V100 训练。
4. 没有 Codex 时也可在 Windows PowerShell 本地运行：

```powershell
cd D:\research\P1-FL-Checkpoint-Selection
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
.\scripts\run_smoke_windows.ps1
```

PowerShell 可能因执行策略限制激活 `.ps1`。可以用 `.venv\Scripts\python.exe -m pip install -r requirements.txt`、`.venv\Scripts\python.exe -m pytest -q` 和 `.venv\Scripts\python.exe scripts/p1_cli.py all --config configs/phase0_smoke.json`，不必修改系统执行策略。

## 官方 FNBench 下载

在 Codex 可联网环境中：

```powershell
.\scripts\get_fnbench_windows.ps1
```

或 Bash：

```bash
bash scripts/get_fnbench.sh
```

**IMPORTANT**：因创建此 ZIP 的执行环境无法直接克隆 GitHub，上游 FNBench 原代码**没有包含在压缩包中**。但 ZIP 已包含可独立运行的 FedAvg、小规模验证、离线 checkpoint 选择的完整工程。官方源码通过上述脚本获取，然后按照 `CODEX_MASTER_PROMPT.md` 中的约束接入，才能用于更严格的论文级对照。

## 看哪些文件

- `CODEX_MASTER_PROMPT.md`：完整给 Codex 的任务说明
- `README.md`：安装与运行方法
- `p1/`：可运行的训练/验证/选模/审计实现
- `tests/`：数据隔离等自动测试
- `configs/phase0_smoke.json`：2轮 CPU 调试（已在打包前真实运行）
- `configs/phase0_v100.json`：正式 CIFAR-10 配置草案，不要自动启动
- `docs/`：严格研究方案与验收清单

**现在不是要让 Codex 立即跑满 120 轮训练，而是先让它验收项目和 FNBench 源码，再提交代码。**
