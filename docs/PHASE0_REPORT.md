# Phase-0 实际交付报告（2026-10-17，Asia/Shanghai）

## 结论与执行边界
完成起始工程审计、环境检查、官方源码获取/审计、最小外部适配、全部单元测试和真实两轮CPU smoke；停止在工程验收。科学结论 **INCONCLUSIVE**。没有下载CIFAR、运行真实GPU训练、100–120轮、多seed或FedNoRo，没有新增loss/聚合器/创新模块。

工程目录：`D:/AVibCodingProjects/P1_FL_Checkpoint_Selection_Starter_20261017/P1-FL-Checkpoint-Selection`。任务书原文保留未修改；按其第一/二阶段执行，在可获取上游后实施了允许的第三阶段审计和最小仪表化。

## 版本与环境
- 原ZIP不是Git仓库；新建分支 `phase0-audit`。原始完整代码保存提交 `afa9851d8473343d1bb221a657048e6f38047fd1`，没有覆盖起始代码。
- 最终实际测试的代码提交 `0fe5599af8d7b28e76bdac710046570264455605`。本报告/证据后续另作纯交付提交，最终提交SHA以 `git rev-parse HEAD` 和交付消息为准；不会在文件中伪造自引用SHA。smoke元数据记录上述实际代码SHA。
- 官方FNBench克隆成功：`838dff3023335c015dea07fa13dc42ca971d525d`，完整本地副本 `vendor/FNBench`；准确pin在 `configs/fnbench_upstream.json`，补丁纳入主Git。
- Windows11，Python3.12.7（`D:/anaconda3/python.exe`）；torch2.12.0+cpu，torchvision0.27.0+cpu，numpy1.26.4，pytest7.4.4，sklearn1.5.1，matplotlib3.9.2。
- RTX3050系列，4GB，驱动566.07；nvidia-smi所示CUDA12.7是驱动支持版本，**torch CUDA runtime=None、CUDA unavailable**，本次仅CPU。物理RAM16,905,818,112 bytes（约15.75GiB）；运行前可用约3.25GB；D盘空闲约64GB。CIM内存查询被沙箱拒绝，改用psutil只读查询成功。
- 网络：GitHub真实clone成功。`bash scripts/get_fnbench.sh` 首次因Windows WSL `Bash/E_ACCESSDENIED`失败，PowerShell脚本被本机签名策略拒绝；没有改执行策略，直接 `git clone https://github.com/Sprinter1999/FNBench.git vendor/FNBench` 成功，后用跨平台Python精确准备脚本验证可重复应用补丁。
- 没有安装或升级系统包。原始官方import实报缺torchtext；lazy AGNews补丁后导入成功。候选V100环境不使用上游不配套torch/torchvision版本，详见手册，尚未安装验证。

实际机器报告见 `evidence/environment.json`，现有开发环境包快照 `evidence/environment_packages.txt`（不是可跨平台直接安装的锁文件）。

## 实际测试结果
1. 未修代码原12项测试：首次9 passed/3 errors，3项均为pytest默认临时目录无法创建；将TEMP/TMP改至工作区后 **12 passed in 5.04s**。原始独立starter两轮smoke成功，loss2.3016/2.2987；保留旧输出，不作为新结果。
2. 最终扩展测试 **29 passed，0 failed，0 skipped，7.79s**；1条官方DataLoader在CPU无accelerator时pin_memory无效的警告。见 `evidence/pytest_final.txt`。
3. 独立starter最终两轮CPU smoke：train loss2.3016/2.2987；72 train、24 val、32独立synthetic test；2 checkpoints、16条public score、10项选择、12条事后regret记录。见 `evidence/smoke_starter.txt`、`starter_verification.json`。
4. 官方FedAvg组件最终两轮CPU smoke：原生ResNet18、IID、4 clients、每轮全部参与、CE/SGD/weight_decay/drop_last/BN聚合使用上游实现；loss2.459915/2.415407，样本/记录数量同上。见 `evidence/smoke_fnbench.txt`、`fnbench_verification.json`。
5. 官方同seed重放，仅将val噪声率改为[.6,.4,.2,0]、noise seed改为3031：split manifest逐字节相同，两份checkpoint SHA256均完全相同，两个checkpoint各自test predictions SHA256完全相同，test分数一致。检查程序实际退出0；配置/日志/证明在evidence。**不是增加随机种子或真实大规模实验。**
6. 官方组件直接调用与适配器同输入/RNG状态逐项权重及loss一致；checkpoint reload预测一致、BN整数buffer语义保留。完整原main的RNG消费并未逐位复现，详见官方审计。
7. 带守卫的端到端单测确认freeze前没有test source访问；只有CSV的孤立目录也能选模；NaN/Inf/非法noise/异常配置/clean scenario/重复缺失分数/冻结结果篡改均拒绝。精确转移矩阵测试验证uniform noise公式及阈值，不将随机噪声必然改变排序作为前提。
8. `git diff --check` 通过；图像已实际查看，标题显式SYNTHETIC DEBUG ONLY。全部数值/曲线均为真实调试运行。

**所有smoke使用synthetic-debug，不是CIFAR-10，不是P1科研结论，不能据此给出GO/NO-GO或显著性。** 官方debug test两轮accuracy恰为.09375，各选择regret为0，不能据此判定课题无效；独立debug也只用于验证流水线。

## 关键修复与源码结论
- train/evaluate原先会读取或提前评分test；现在严格train -> noisy validation -> 冻结选择 -> audit test/clean。
- selector只读白名单CSV及自己的输出摘要，不读配置/私有oracle/原始dataset；EMA由训练config_used冻结标量提供。支持Last/macro/micro/CE/EMA，clean/test-best仅事后诊断。
- 新增来源命名空间与split断言、真实label view固化、输出覆盖保护、checkpoint/config/split哈希和选择冻结校验、样本/参与者日志、test prediction hash；修复CLI --out错位、Last NaN、直接API绕过schema。
- 官方入口 `main_fed_LNL.py` 强制CUDA，每轮聚合后访问test但FedAvg不以test更新训练或选最优模型；无全局torch.save。checkpoint仪表化放在每轮官方global load_state_dict之后。
- 主调用链和逐文件审计见 `CODE_AUDIT.md`；官方训练/测试/聚合/noise CLI/BN/分区细节、源码推断和未验证项见 `FNBENCH_AUDIT.md`。
- 实际LICENSE为MIT，与README Apache-2.0徽章冲突，已如实纠正交付说明，没有修改上游许可证。

## 实际复现命令
以下在工程目录执行；测试使用2个CPU线程。已执行命令与关键早期失败也列于上文。

```powershell
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
python scripts/prepare_fnbench.py
python -m pytest -q
python scripts/p1_cli.py all --config configs/phase0_smoke.json
python scripts/p1_cli.py all --config configs/phase0_fnbench_smoke.json
python scripts/p1_cli.py all --config docs/evidence/noise_replay_config.json
python scripts/verify_phase0_outputs.py outputs/phase0_smoke_final_20261017
python scripts/verify_phase0_outputs.py outputs/phase0_fnbench_smoke_final_20261017 --compare outputs/phase0_fnbench_noise_replay_20261017
python scripts/record_environment.py
python -m pip freeze
git diff --check
```

这些目录已经存在。复跑训练须使用新 `--out outputs/unique_name`，相应调整验证路径；不会删除或覆盖旧文件。tests的临时目录也全部保留，不做批量清理。

## 产物实际位置
- `outputs/phase0_smoke_final_20261017/`
- `outputs/phase0_fnbench_smoke_final_20261017/`
- `outputs/phase0_fnbench_noise_replay_20261017/`

三个目录均含config、metadata（代码SHA和官方SHA）、split、train日志、checkpoint、selection CSV/冻结hash、regret、图和private_audit。每目录2个checkpoint。本机原始checkpoint保留（官方每个约45MB），不把大文件提交Git；轻量结果/图/manifest/私有诊断快照在 `docs/evidence/{starter,fnbench,fnbench_replay}` 纳入Git。SHA列表见 verification JSON，可验证本机文件未变。

完整改动文件表见 `CHANGED_FILES.md`。主仓库忽略vendor、outputs、.local_tmp；vendor只有已记录loader补丁及Python自动缓存，算法文件未改。没有GitHub远程，不创建PR、不推送。

## 后续命令、工作量与未验证边界
V100候选安装命令和GPU检查见 `V100_RUNBOOK.md`。**后续获得授权后的**正式命令：

```bash
bash scripts/run_v100.sh --approved-formal-run
```

先安装候选环境、授权下载官方CIFAR并通过短pilot；此命令会执行1seed、10 clients、120 rounds、每轮1 local epoch、约7.4万batch updates、24个checkpoint，约1.1GB checkpoint磁盘。真实吞吐未测，不虚报小时数。IID及20%训练噪声是独立已备配置，不能自动串跑。

尚未完成：真实CIFAR indices与内容检查、V100环境/短pilot/收敛检查、正式训练与统计显著性、多seed、FedNoRo与其他文献baseline。官方Dirichlet每类固定seed且有最小样本重试局限，保留算法原样并记录；source namespace断言不等价于已实际运行真实数据。文件哈希不构成恶意篡改安全边界，不用python -O运行断言验收。

当前工程无阻塞CPU验收的错误；科研有效性证据仍不足。等待用户验收及下一条明确指令，不启动正式GPU训练。
