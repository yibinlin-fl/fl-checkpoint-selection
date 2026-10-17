# P1 Phase-1A OpenI 启动卡（待核验，尚不可宣称云端已就绪）

任务唯一标识：`p1-phase1a-s42-r2-20261017-v1`。角色：真实 CIFAR-10 工程 smoke，单 seed42、两轮全局 FedAvg。Phase-1A 训练获授权；120轮/多seed未授权。私有数据集上传、代码推送、创建网页任务未执行，目标仓库与镜像未提供。

## 代码身份
分支 `phase1a-data-v100`；可执行代码提交 `20d48d5f42ebc10de0cb6b1a36aed89c98cc6d2b`。GitHub/OpenI远端均未创建/配置，没有push或fetch远端状态。OpenI选用的仓库/分支/commit：**待核验**。不将本地提交冒称已上传。可以使用OpenI自己的代码仓库，不依赖GitHub。

可审阅代码ZIP：
`D:/AVibCodingProjects/P1_FL_Checkpoint_Selection_Starter_20261017/P1-FL-Checkpoint-Selection/outputs/openi_packages/p1_phase1a_code_delivery.zip`

316,496 bytes，SHA256 `fcea182f78d71c2411812442273444caa3ecf4ca6f93a6fc8cd3e097d63c3851`；根目录直接含p1/scripts/configs等。代码ZIP没有.git，若采用上传文件方式创建OpenI仓库，应额外记录OpenI新commit；原代码身份可通过代码ZIP hash及git archive commit追溯。

## 输入包
输入为官方 CIFAR-10 原始归档、冻结的真实split/噪声映射，以及官方 FNBench Git bundle。没有模型checkpoint、Token或账户凭据。

本地完整路径：
`D:/AVibCodingProjects/P1_FL_Checkpoint_Selection_Starter_20261017/P1-FL-Checkpoint-Selection/outputs/openi_packages/p1_phase1a_input_v1.zip`

精确174,029,096 bytes；SHA256：
`fefafd3e61464c3795dfcdc904020370f4911c94b6e8ee2c6feae2404faab946`

包内根目录 `p1_phase1a_input/`，包含 `input_manifest.json`、`data_cache/cifar-10-python.tar.gz`、`prepared/`、`FNBench.bundle`。包及内部各文件hash已本机核验；FNBench bundle已实际克隆验证到SHA `838dff3023335c015dea07fa13dc42ca971d525d`，不需要云端联网重新下载上游源码。

OpenI dataset owner/repo_id、显示名、可见性、远端文件名与复用状态：**待核验**。建议私有；不能默认公开。未上传。旧代码包p1_phase1a_code_c3e5785.zip已被delivery版取代，不上传旧代码包。相似的旧Phase-0输出不是输入包；不要上传初次Toronto下载的 `.partial_phase1a` 或失败range文件。只上传上述SHA对应的ZIP。

## 环境与网页照填区
- 任务名称：`p1-phase1a-s42-r2-20261017-v1`
- 代码仓库/分支/commit：**待提供/远端核验**
- 镜像名/tag：**待提供**；要求Python3.10、配套torch2.5.1/torchvision0.20.1 cu118候选环境、可导入c2net及git。必须在真实镜像验证，不能填CPU环境镜像。
- 数据集：**待提供 owner/repo_id 与可见性**
- GPU：V100 × 1；实际型号/显存/空闲量待 `require_v100()` 检查，不能以RTX3050替代。
- 启动文件：`scripts/openi_phase1a.py`

| 实际参数名（保留下划线） | 精确填写值 | 说明 |
|---|---|---|
| run_id | p1-phase1a-s42-r2-20261017-v1 | 重跑需新run_id |
| mode | phase1a | 只有该合法选项，不接受formal |
| data_source | 空字符串 | 从c2net.dataset_path查找唯一archive_name；若平台不接受空参数，可省略，默认也是空 |
| archive_name | p1_phase1a_input_v1.zip | 远端文件名必须一致；多个匹配会拒绝 |
| input_bytes | 174029096 | 字节数，不是MB |
| input_sha256 | fefafd3e61464c3795dfcdc904020370f4911c94b6e8ee2c6feae2404faab946 | 整个ZIP的SHA256 |
| work_root | /tmp/p1_openi_runs | 容器本地临时空间，镜像容量/可写性待核验 |
| config | configs/phase1a_v100_smoke.json | 两轮配置，入口再次检查必须恰好2轮 |

Windows路径不填写进data_source。c2net输出挂载由prepare()提供，上传API在昂贵训练前必须存在。不能把本机OpenI Token当作云端c2net上下文。

内部固定：10 clients、全部参与、Dirichlet alpha=.5、80/20 holdout、seed42、官方ResNet18、clean训练标签、hetero=[0,0,0,0,.2,.2,.4,.4,.6,.6]、noise seed2031、uniform=.27448（按真实val样本数加权）、EMA=.6、bs64、local_ep1、lr.01/momentum.9/weight_decay.0005。每轮保存checkpoint。离线重放只把noise seed固定加1为2032，不重新训练，不搜索条件。

## 容器安装/本地终端等价命令
镜像环境准备参考 `docs/V100_RUNBOOK.md`；本项目未在OpenI安装过这些依赖，也未验证任何镜像tag。不要在未知基础镜像中静默覆盖torch或c2net。

```bash
python scripts/openi_phase1a.py --run_id p1-phase1a-s42-r2-20261017-v1 --mode phase1a --data_source "" --archive_name p1_phase1a_input_v1.zip --input_bytes 174029096 --input_sha256 fefafd3e61464c3795dfcdc904020370f4911c94b6e8ee2c6feae2404faab946 --work_root /tmp/p1_openi_runs --config configs/phase1a_v100_smoke.json
```

## 资源/产物合同
本机没有V100，没有每轮GPU耗时/峰值显存/吞吐/OOM数据，不能承诺耗时或填付费预算。账号资源、费率、余额、任务时限：**待核验**。两轮实际处理约79,488训练样本（1,242个batch更新，官方drop_last）；模型副本与数据加载需要足够CPU RAM，建议先核验至少8GB内存与10GB空闲磁盘。该容量建议是工程余量，不是实测GPU占用。

流程：验证c2net与真实V100 -> 验证ZIP/内部manifest/官方MD5 -> 核对预注册split与数据 -> 两轮训练 -> 完成checkpoint roundtrip -> **先打包并上传含checkpoint的恢复包** -> 再evaluate/select/audit/plot -> 同组checkpoint离线噪声重放 -> 成本记录 -> 上传正式包。

精确远端产物名：
- `p1_p1-phase1a-s42-r2-20261017-v1_recovery_outputs.tar.gz`
- `p1_p1-phase1a-s42-r2-20261017-v1_outputs.tar.gz`
- `p1_p1-phase1a-s42-r2-20261017-v1_status.json`
- 失败时另有 `p1_p1-phase1a-s42-r2-20261017-v1_failed_outputs.tar.gz`

恢复包包含metadata、实际config、训练日志、两份checkpoint、roundtrip证明，状态training_complete_analysis_pending且scientific_evidence=false。恢复包上传报错则不开始分析，尝试回传失败状态。正式包含完整选择、事后oracle、regret、cost与离线重放；本短smoke始终scientific_evidence=false、scientific_verdict=INCONCLUSIVE。包内有逐文件bytes/SHA256清单。

SDK正常返回只标记upload_call_returned，不冒称远端文件可下载或hash一致。需下载后独立审计。恢复包是模型权重恢复，不包含完整optimizer/RNG/loader续跑状态；不支持无损接续训练。容器硬销毁不能保证finally执行，分析前回传用于降低结果丢失风险。

## 下载与验收
已创建下载目录：
`D:/AVibCodingProjects/P1_FL_Checkpoint_Selection_Starter_20261017/P1-FL-Checkpoint-Selection/outputs/openi_downloads/p1-phase1a-s42-r2-20261017-v1`

1. 核验并选择实际代码commit、镜像和私有数据集后，再照表创建短任务；本卡不是已成功创建任务的证明。
2. 查看开头run_id/环境/数据hash；两轮训练后先确认恢复包，再等正式结果。
3. 将两类包、status及平台日志原样下载到上述目录，保留原包，不覆盖其他run。
4. 交回本项目核对sha、模型数/轮次、源split、选择冻结顺序、预测/重放及完整状态；网页exit0、上传100%均不是充分验收。
5. Phase-1A云端验收通过后，才决定是否批准 `bash scripts/run_v100.sh --approved-formal-run`，当前禁止执行。
