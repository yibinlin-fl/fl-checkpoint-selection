# 官方 FNBench 集成协议

上游：https://github.com/Sprinter1999/FNBench

目前提供两个并行路径：

A. 本包自包含简化 FedAvg（可立即 smoke，可跑 CIFAR-10 baseline）。

B. 官方 FNBench（推荐用于论文级 FedAvg/FedNoRo 对照）。需要通过 `bash scripts/get_fnbench.sh` 克隆。**B 还未完成集成，不得称已跑 FNBench 结果。**

先读上游实际 commit 的 README、main_fed_LNL.py、utils/options.py、data/partition.py、flcore/federation.py、flcore/update.py、flcore/test.py、data/datasets/loader.py。

源码审计关注：

1. `main_fed_LNL.py` 强制 CUDA（纯 CPU 不可按原命令跑）。
2. train 与 validation/test 划分：FNBench 原版 `dataset_train` / `dataset_test` 不代表已提供独立 val；必须添加 index manifest，不能复用官方 test 作为调参集。
3. 记录每轮是否调用 `test_img` 以及是否以其结果提前停止/选择 checkpoint；论文实验必须隔离。
4. `args.num_users`、`args.frac`、`args.epochs`、`args.noise_group_num`、`args.group_noise_rate` 实际语义，配置 10 clients 要满足噪声组个数求和等于客户端数。
5. 对含有 BN running buffers 的聚合方案保持算法原本语义，确保新增 checkpoint 存取不改变训练轨迹。
6. 新增验证评价脚本只接收训练中间 checkpoint，能够读取 `splits_manifest.json` 和固定噪声映射，但不能读取官方 test labels 去选模型。
7. 按相同实验配置和随机种子对比仪表化前/后无意外训练路径差异，记录上游 SHA、环境、日志、差别和无法完全复现的因素。

完成集成前先提交 `docs/FNBENCH_AUDIT.md`，再做最小补丁，复现 FedAvg 的 2-round smoke，随后才能征询是否跑正式 V100 实验。
