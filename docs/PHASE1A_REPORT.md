# P1 Phase-1A 本地准备与真实数据审计报告

2026-10-17，Asia/Shanghai。**状态：本机可执行准备完成，OpenI V100 环节未运行，Phase-1A 尚未整体通过。** 用户已授权真实数据、V100环境验证和2–5轮smoke；120轮、多个seed及新方法仍未授权。

## 实际版本与环境
- 当前分支 `phase1a-data-v100`；Phase-0基线 `9d916a6e425b7bb10d8c9b0cefccf849755fbefa`。
- 真实数据审计/仪表化实现 `0443d662788a38378577ee330f41ad4a6434d5f6`；真实数据测试及成本估计完善 `19a5698a6b99429652a9effdc2937005d8c61bea`；OpenI可执行代码提交 **`20d48d5f42ebc10de0cb6b1a36aed89c98cc6d2b`**。最终报告/证据单独提交，最终交付SHA见git rev-parse HEAD及交付消息，避免自引用SHA。
- FNBench上游仍固定 `838dff3023335c015dea07fa13dc42ca971d525d`；官方算法文件未修改，vendor仍只有lazy AGNews导入补丁。
- 本机 Windows、Python3.12.7，PyTorch2.12.0+cpu、torchvision0.27.0+cpu，GPU RTX3050 Laptop 4GB，驱动566.07。本机没有V100、CUDA-enabled Torch或c2net。nvidia-smi的CUDA12.7是驱动能力，不是已安装PyTorch CUDA runtime。
- 会话刷新后PATH的python曾切换到 `D:/software/python/python.exe`，该解释器缺torch/pytest。没有误装依赖，后续验证明确用 `D:/anaconda3/python.exe`；不能将该环境错误当作项目测试失败。
- OpenI CLI在 `D:/anaconda3/envs/pytorch/Scripts/openi.exe`，SDK3.0.1；默认日志在沙箱外写失败。只在临时Python进程将logging输出重定向到工作区，成功读取dataset upload --help；没有更改凭据、系统PATH或升级SDK，没有执行上传/登录认证。凭据文件存在不代表当前认证有效。
- 没有配置任何GitHub/OpenI git remote，没有创建远端仓库、push或启动云任务；目标owner/repo_id、镜像、平台挂载和账号资源尚未提供，不能宣称已核验远端版本。OpenI可以直接使用自己的代码仓库，无需GitHub中转。

## 真实 CIFAR-10 下载与内容核验
最初Toronto下载约50KB/s，保留部分归档后中断；同站分段请求最终SSL UNEXPECTED_EOF失败，保留部分文件，没有批量删除。随后采用 [PaddlePaddle官方源码列出的同名归档镜像](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/dataset/cifar.py)，使用torchvision的标准下载/解压工具，并强制 [CIFAR官方MD5](https://www.cs.toronto.edu/~kriz/cifar.html) 校验。只是传输镜像变化，没有重新编码、换版本、丢弃或重新标注样本。

- 有效归档：`data_cache/cifar-10-python.tar.gz`，**170,498,071 bytes**。
- 官方MD5：`c58f30108f718f92721af3b95e74349a`，通过。
- SHA256：`6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce`。
- 五个train batch、test_batch及batches.meta均逐一通过官方MD5，另记录SHA256/bytes。完整来源在 `docs/phase1a_evidence/download_provenance.json`。

## 真实划分与数据隔离
冻结10 clients、Dirichlet alpha=.5、validation_fraction=.2、seed42；继续调用官方partition，再固定客户端内部holdout。没有挑seed或改变alpha。原始50000条官方train完整分配为 **40000 train + 10000 validation**；官方10000条test保持独立来源。

| client | train n | validation n |
|---|---:|---:|
| 0 | 4185 | 1046 |
| 1 | 4532 | 1133 |
| 2 | 3406 | 851 |
| 3 | 2388 | 597 |
| 4 | 2726 | 682 |
| 5 | 3486 | 872 |
| 6 | 5993 | 1498 |
| 7 | 3156 | 789 |
| 8 | 5891 | 1473 |
| 9 | 4237 | 1059 |
| 合计 | 40000 | 10000 |

实际断言/审计：
- train内部、val内部与跨client索引均唯一；train/val交集为空，合并完整覆盖0..49999。
- train/val使用official_train命名空间，test使用official_test，test indices完整0..9999；不把同名整数误当同一来源样本。
- 全量60000张图逐像素SHA256审计：train内部、train/val之间、official train/test之间的**完全相同图像数量均0**。不是语义近重复检测，不能声称排除所有近重复图像。
- 对真实数据重复相同seed的官方assignment/holdout，结果逐项相同；noise view生成前后原始train图像与标签hash相同。
- 数据审计读取了test文件做完整性/像素重复检查；没有计算任何模型输出或test accuracy，未据test调整配置。真实test模型性能仍只能在冻结选择后audit。
- official_dirichlet每类重置numpy seed的上游局限仍保留；本次没有用其他seed绕过它。

实际文件目录 `outputs/phase1a_cifar10_prepared_seed42/`：splits_manifest.json（索引与class hist）、source_indices.npz（不含object arrays）、registered_settings.json、dataset_statistics.json、prepared_artifacts.json及private_audit中的真实冻结label view/噪声实现率/像素审计。完全相同轻量快照已纳入 `docs/phase1a_evidence/prepared_data/`。训练开始前会验证其hash、登记配置、数据文件MD5/SHA及重新计算的split一致性。

## 训练噪声与三个验证条件
训练标签噪声固定 **0.0**，后续不能根据结果改为20%。heterogeneous固定 `[0,0,0,0,.2,.2,.4,.4,.6,.6]`；validation noise seed2031。按真实val样本量加权得到uniform计划率 **0.27448**，不是异质率的算术平均0.24。

这批固定val实际翻转数：clean0、uniform2736、heterogeneous2831；不同实际数来自独立Bernoulli采样，不通过事后更换seed匹配。诊断重放仅固定noise seed+1=2032，计划率、训练标签、split均不变。

clean/uniform/heterogeneous使用同一组训练checkpoint：实际selector仅读取uniform/heterogeneous CSV，clean为冻结选择后的诊断；test最佳仅事后regret参照。OpenI离线噪声重放复制并校验同一批checkpoint，不是三次训练；报告明确区分“共享模型证明”与“独立重新训练的GPU逐位可复现性”。后者尚未进行，不能以文件复制冒充重新训练重放。

**尚无真实CIFAR checkpoint**，所以真实GPU checkpoint SHA/test prediction不变性、真实选择结果和regret尚无实测结果。当前已验证真实数据构造不改训练输入，以及Phase-0 synthetic独立训练不变性。

## 工程实现与真实测试
- 新的Phase-1A配置只允许2–5轮、每轮保存；OpenI入口进一步冻结为恰好2轮。所有预注册科学参数逐项比较，拒绝120轮、改seed/alpha/训练噪声/CPU等越界配置。
- V100预检检查真实名称、compute capability7.0、wheel sm_70、显存及小CUDA运算。本机实际预检退出1，正确在训练前停止；没有偷偷回退RTX/CPU，正式输出目录未创建。记录见 `local_v100_preflight.txt`。
- 官方训练仅增加每轮耗时、有效处理样本、吞吐、checkpoint保存时间/大小、训练轮次内PyTorch峰值allocated/reserved显存；不修改CE/SGD/weight_decay/drop_last/聚合/BN语义。另准备真实训练模型与checkpoint reload logits一致性检查。
- 最终全部 **59 passed，0 failed，0 skipped，9.06s**，1条官方CPU pin_memory无accelerator警告。含实际CIFAR索引/噪声审计、阶段授权边界、禁止非V100、危险解压拒绝、输入manifest、缺c2net、上传失败不分析、成功回传顺序等。OpenI成功/失败路径用mock测试，**不代表真实云端上传通过**。
- 修改后再次完成FNBench原组件两轮CPU synthetic regression，loss2.459915/2.415407；同Phase-0 checkpoint SHA与test predictions一致。真实GPU/CIFAR训练没有运行。完整测试日志 `docs/phase1a_evidence/pytest_openi_delivery.txt`，smoke日志 `cpu_openi_delivery.txt`。
- `git diff --check`、CLI --help、input ZIP/内部manifest哈希验证、官方FNBench bundle独立克隆均通过。

## OpenI 输入、回传与启动卡
已按用户指定 [openi-experiments Skill](C:/Users/asus/.codex/skills/openi-experiments/SKILL.md) 实现合同；完整可照填卡见 **OPENI_PHASE1A_LAUNCH_CARD.md**，未知远端字段明确待核验。

代码包 `outputs/openi_packages/p1_phase1a_code_delivery.zip`：316,496 bytes，SHA256 `fcea182f78d71c2411812442273444caa3ecf4ca6f93a6fc8cd3e097d63c3851`。

输入包 `outputs/openi_packages/p1_phase1a_input_v1.zip`：174,029,096 bytes，SHA256 `fefafd3e61464c3795dfcdc904020370f4911c94b6e8ee2c6feae2404faab946`。包含官方归档+已审计prepared目录+FNBench.bundle，不含Token或模型。使用Skill流式manifest工具实际create/verify通过。该包未上传，避免在未确认repo_id/可见性前公开研究内容。

入口 `scripts/openi_phase1a.py` 要求c2net output context存在，校验整个ZIP和内层文件，拒绝路径逃逸/链接/旧run_id覆盖，训练后在昂贵分析前先上传含2个checkpoint的recovery；失败状态保持scientific_evidence=false，上传失败不启动分析。SDK返回正常只表示调用返回，下载后才算远端完整性验收。完整的status/结果命名与下载绝对目录见启动卡。

预创建接收目录：
`D:/AVibCodingProjects/P1_FL_Checkpoint_Selection_Starter_20261017/P1-FL-Checkpoint-Selection/outputs/openi_downloads/p1-phase1a-s42-r2-20261017-v1`

没有创建远端仓库、公开数据、推送代码、CLI上传或启动云端任务。还需确定OpenI代码仓库、数据集repo_id/可见性、镜像与平台资源，才能执行这些外部步骤。

## 实际执行命令（工程目录）
```powershell
& 'D:/anaconda3/python.exe' scripts/download_cifar10.py --source paddle
& 'D:/anaconda3/python.exe' scripts/prepare_phase1a_data.py --config configs/phase1a_v100_smoke.json
& 'D:/anaconda3/python.exe' scripts/run_phase1a.py --config configs/phase1a_v100_smoke.json
# 上一条按预期在本机V100预检失败，没有训练
$env:OMP_NUM_THREADS='2'; $env:MKL_NUM_THREADS='2'
& 'D:/anaconda3/python.exe' -m pytest -q
& 'D:/anaconda3/python.exe' scripts/p1_cli.py all --config configs/phase0_fnbench_smoke.json --out outputs/phase1a_openi_delivery_smoke
& 'D:/anaconda3/python.exe' scripts/openi_phase1a.py --help
& 'D:/anaconda3/python.exe' scripts/build_openi_input.py
```
另实际执行Skill artifact_manifest.py create/verify、git bundle创建/独立clone与git archive代码ZIP。数据/输出已有时复跑要用新的路径，不清理旧目录。SDK CLI help使用工作区日志重定向子进程；没有读取或输出Token值。

## Phase-1B 配置与成本
完整待授权配置 **configs/phase1b_v100.json**：保持上述训练/数据/seed条件，120 global rounds、每5轮保存共24个，strict deterministic执行。仍是单seed clean-train FedAvg，没有算法新增。**不允许本次运行。**

后续Phase-1A真实验收通过并另行批准后，命令：
```bash
bash scripts/run_v100.sh --approved-formal-run
```

本机真实split对应每轮621个SGD batch、有效39744训练样本（官方drop_last）。两轮1242 updates；120轮74520 updates。官方ResNet18 CPU权重文件每个44,771,211 bytes，24个约1.07GB仅供存储规划，**不是GPU checkpoint I/O实测**。

真实GPU每轮时间、峰值显存、吞吐、OOM、loader表现及实际I/O均**待测**，不外推CPU时间到V100。云端成功后cost_estimate.json将记录每轮训练/保存耗时、allocated/reserved峰值，以及120轮训练+24次checkpoint保存的估计；短smoke稳态样本很少，不含安装/下载/离线分析时间，需明确不确定性。

## 尚未完成与下一步
OpenI V100镜像/依赖安装和c2net真实上下文、2轮真实训练、checkpoint加载、三个真实验证场景、冻结后test/regret、真实GPU重放与成本均待云端执行。当前只能确认真实输入和本地工程准备有效，不能宣称Phase-1A通过或P1链条成立。等待仓库/镜像信息及可执行的OpenI操作入口；不启动Phase-1B。
