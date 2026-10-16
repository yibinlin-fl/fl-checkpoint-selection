# 请 Codex 完整执行以下任务（P1 Phase-0 科研项目）

你现在是我的科研工程 Agent，目标是**审计和可复现实验**，不是创造一个新算法。请先通读本工作区的 `README.md`、`docs/RESEARCH_PROTOCOL.md`、`docs/FNBENCH_INTEGRATION.md`、`docs/ACCEPTANCE_CHECKLIST.md`、`configs/`、`p1/` 和 `tests/`，然后严格按阶段执行。

## 研究背景与唯一问题

研究方向：联邦学习客户端验证数据质量异质性导致的**全局 checkpoint 选择偏差**。在同一次 FL 训练获得同一组全局模型的情况下，不同客户端的验证标签可能具有不同噪声率、客户端类别分布可能不同。服务器若只使用含噪 validation loss/accuracy，是否会选中对**预先指定目标分布**表现更差的全局 checkpoint？相比最后一轮、含噪加权 accuracy、含噪 CE、EMA 平滑以及已有无验证选择方法，损失是否稳定、显著且不可被简单手段解决？

这只是候选研究假设。不要预设结果一定支持课题，不要调配置凑出正结果；出现 NO-GO 请如实报告，绝不虚构实验或改写历史。

**任务范围冻结**：只研究已经训练得到的多个全局 checkpoint 之间的 model selection，不研究本地 early stopping、客户端选择、新 FL 聚合器、新鲁棒增强、新 loss、改变模型架构以提高 accuracy，也不研究超参数搜索或自动重训练。

## 第一阶段：工作区与代码审计（必须先做）

1. 检查 `git status`、当前分支、代码是否改动、Python、PyTorch、torchvision、CUDA/GPU/内存/磁盘、网络状态。仅汇报可验证的真实信息；若工作区还不是 git repo，安全地初始化并在新分支工作，不覆盖我的代码。
2. 先审计 ZIP 中的**独立 P1 起始工程**，指出其功能、限制、关键调用链、潜在科学错误和数据泄漏风险。要理解：这不是官方 FNBench 代码；`synthetic-debug` 是工程调试数据。
3. 如果能访问 GitHub，执行 `bash scripts/get_fnbench.sh`，克隆官方 `https://github.com/Sprinter1999/FNBench` 到 `vendor/FNBench`，记录真实 SHA，许可证 Apache-2.0。网络不可用时报告错误并继续本地测试，不要编造已克隆的结果。
4. 在 `vendor/FNBench` 可用时阅读 README、main_fed_LNL.py、utils/options.py、data/partition.py、data/datasets/loader.py、flcore/update.py、flcore/federation.py、flcore/test.py 等实际存在的文件。说明 client 数、参与率、epoch、label noise CLI 的实际定义、测试集访问时机及模型保存方式。特别注意原入口可能强制 CUDA、可能每轮用 test_loader。**不要直接拿官方测试集选 checkpoint**。
5. 产出 `docs/FNBENCH_AUDIT.md`；分开写已核实事实、源码推断、目前未验证项。若缺依赖，先记录实际错误及可重现修复方式。

## 第二阶段：本地可运行性和数据隔离测试（无需 GPU）

1. 运行 `python -m pytest -q`，修复真实代码 bug，不改科学假设。
2. 运行 `python scripts/p1_cli.py all --config configs/phase0_smoke.json`。这只执行两轮 CPU synthetic debug 数据，目标是输出真实 checkpoint、`validation_scores.csv`、`selected_models.csv`、`regret_summary.csv`、图、配置、记录。**这不是 CIFAR-10 真实结果，不能声称证明研究现象**。
3. 验证 train/val index 完全互斥，CIFAR-10 官方 test 始终独立；固定随机种子时 splits/noise 可复现；生成不同验证标签视图不得影响任何训练参数和 checkpoint。
4. 检查保存/加载 checkpoint 后相同输入预测一致；检查 no NaN，记录轮次和 sample counts。原始训练 loss、FedAvg weighting、参与客户端规则无意外变化。
5. 最重要：选择器只允许读取 `validation_scores.csv`（字段白名单，且仅允许 uniform/heterogeneous scenario），**不得访问** `private_audit/oracle_test.csv`、干净验证真标签、注入噪声真值或其派生信息。真正的测试准确率仅能在冻结选择结果后，由 `audit` 计算 regret。
6. 统一对称噪声数学自检：C 类、翻转至其余 C-1 类，`E[Acc_noisy]=(1-Cη/(C-1))*Acc_clean+η/(C-1)`。在统一 η 小于 (C-1)/C 的期望下，模型准确率排序保持；不要将“标签噪声必然导致选模错误”作为事实。

## 第三阶段：与 FNBench 的正式集成（有原仓库后执行）

1. 优先让实验以 FNBench 官方 FedAvg 为主，而不是把这个独立小框架冒称官方实现。只对其做数据集划分、冻结的验证噪声视图、全局 checkpoint、验证指标导出等**只读仪表化**；不改原损失、optimizer、aggregation。
2. 使用 CIFAR-10，10 clients，Dirichlet α=0.5，ResNet18（官方实装版本）、固定 train/validation 留出，建议比例 80/20。原始 CIFAR 50k 训练集合规切分；官方 CIFAR 10k 测试集仅离线 oracle audit，绝不参与选轮次/调参。
3. 比较同一组 checkpoint 的 (a) clean val [**仅诊断上界**]、(b) 客户端相同率的对称标签噪声、(c) 客户端异质率。建议 hetero 预注册 `[0,0,0,0,.2,.2,.4,.4,.6,.6]`，uniform 率严格按 val 样本数加权匹配，而不是简单求算术平均。不同 scenario 不能重新训练同一模型，各场景使用同一组 model checkpoint；噪声只注入 val 标签、不改变 train labels。
4. 训练标签噪声单独控制（可从 clean train 开始，再固定 training noise 20% 作为第二条件）；不要把 train label corruption 与 validation noise 混同。客户端 class hist/样本量固定并保存。目标分布先冻结为官方 CIFAR-10 iid test；对 clean val 下分布不一致可能带来的性能偏差单独报告。新增 IID 对照，但不能通过事后挑选 alpha 或 seed 找最有利结果。
5. 暂定 100–120 global rounds、每 5 轮一个 checkpoint、1 seed 先行。如果已有官方参数名不一致，以源码为准修正、记录。再计划 3–5 seeds 和另一种鲁棒算法 FedNoRo；**没有得到我明确批准，不得启动 100+ 轮/大量 GPU、下载超大数据或执行 3–5 seeds**。
6. Selection-only 策略：Last、macro noisy accuracy、按 val n 加权 noisy accuracy、noisy CE、EMA。提供 clean validation 和 oracle test best 仅用于事后诊断，绝不送给实际选模模块。如果加入现有鲁棒选择/无验证早停，严格区分已直接实现和自行适配。

## 第四阶段：指标、可追溯结果与 Go/No-Go

输出：`config_used.json`、git SHA、FNBench upstream SHA、`splits_manifest.json`、`checkpoints`、`validation_scores.csv`、`selected_models.csv`、`private_audit/oracle_test.csv`、`regret_summary.csv`、验证曲线、正式的 `docs/DECISION.md`。`selection_regret_pp=100*(best_test_acc_among_saved - selected_test_acc)`；真实测试只用于冻结选择后的评判。测试集与训练集、验证集的泄漏检查要写为程序断言。

决策必须允许：
- `NO-GO`：异质噪声基本不影响实际 checkpoint，或者简单最后轮/含噪 accuracy/EMA 就足够，或效应只在不合理的极端噪声下出现。
- `FURTHER-AUDIT`：真实 FL 训练、多随机种子出现明显的、稳定的非零 regret，且不能被简单策略消除。内部参考阈值：≥2 个百分点、5 seeds 里至少 4 seeds；这不是领域通用指标，不允许事后调条件。
- `INCONCLUSIVE`：数据或计算资源不足以判断、结果不稳定、训练没有收敛/数据太少。

不要把通过实验测试宣称为“已有论文创新性”，后续还需审最近 FL checkpoint model selection、FedES、Label Wave、Noisy Early Stopping 及 related work。

## 工程质量与执行边界

- 优先小而可靠的变更，保留官方 baseline 和实验 config 可复现性；用已有框架，不为了重写而重写。
- 不允许硬编码虚构的 accuracy，也不得伪造已经跑过 V100 的日志。
- 不要将 validation 标签真值、test oracle、真实噪声率传入选模算法。
- 图/CSV 必须对应真实运行的数据；Synthetic debug 要用明显标签标记。
- 网络访问、包安装可按允许权限进行；昂贵 GPU 训练与大型数据下载先征求确认。
- 任何 code patch 要运行 tests + 2-round smoke，说明改动理由，带上重现命令。

## 你本次的停止条件（非常重要）

**本次只完成第一阶段和第二阶段；如果 FNBench 可获取，可以做第三阶段的代码审计和最小补丁，但不能直接启动正式 120-round 训练。**

最后请报告：
1. 原仓库版本、是否成功克隆 FNBench、环境/CUDA 状态；
2. 我收到的 ZIP 本身测试结果和修复内容；
3. 源码审计：真实训练入口、训练评估是否泄漏、checkpoint 储存点；
4. 运行过的真实命令、通过的测试数、输出实际路径；
5. 完整的 Git commit SHA、分支、未提交改动；
6. V100 上的下一条具体执行命令及预计工作量；
7. 任何未解决 blocker 与科学有效性风险；
8. 等待我下一条指令，不要自作主张跑大规模训练。
