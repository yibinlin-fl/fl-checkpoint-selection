# 当前交接索引（2026-10-17）

- Phase-0有条件通过；Phase-1A获授权（真实CIFAR + V100两轮smoke），Phase-1B120轮未获授权。
- 当前报告：PHASE1A_REPORT.md；OpenI启动卡：OPENI_PHASE1A_LAUNCH_CARD.md。
- 已完成：官方CIFAR MD5/SHA审计、40k/10k/10k索引隔离、clean train与三验证label view冻结、59项测试、两轮CPU回归、OpenI安全输入/恢复先回传合同、私有待上传包与下载目录。
- 未完成：OpenI代码仓库/镜像/dataset repo_id确认、OpenI输入上传、真实V100运行及成本；本机RTX3050且CPU Torch，不能冒充云端通过。
- 可执行代码包SHA对应20d48d5f42ebc10de0cb6b1a36aed89c98cc6d2b；输入包和哈希见启动卡。真实数据预处理源提交0443d662788a38378577ee330f41ad4a6434d5f6，源split与科学配置未改变。
- 训练配置configs/phase1a_v100_smoke.json：10 clients/alpha.5/seed42/2 rounds/clean train。正式候选configs/phase1b_v100.json，不执行。
- 独立诊断重放只更改validation noise seed2031->2032，不重训；不要冒称已证明GPU独立重训逐位一致。
- 所有数据/权重/传输ZIP保留在outputs或data_cache，Git提交轻量证据；禁止批量删除。不要覆盖旧run_id、输入或原始下载包。
- 下一步：确认用户的OpenI私有仓库与镜像，按指定Skill继续；无新授权不公开数据或启动长任务。收到结果后保留原包并核验完整性/恢复包/status，再验收Phase-1A。

## GitHub 上传更新（2026-10-17）

用户已明确选择公开上传。GitHub仓库 https://github.com/yibinlin-fl/fl-checkpoint-selection 的 main 已推送并核验基线提交 `68e8172629b8fe83c221a380a164aa05c15dc404`。本地分支 phase1a-data-v100 跟踪 origin/main。数据集原始图像、checkpoints、vendor工作副本和传输ZIP未上传；已提交的报告与轻量审计证据随代码公开。OpenI数据集上传及V100任务仍未执行。后续同步本节的提交仅更新文档，不改变已测实现。
