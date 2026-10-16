# FNBench 最小适配状态

已克隆官方 upstream `838dff3023335c015dea07fa13dc42ca971d525d`，来源 `https://github.com/Sprinter1999/FNBench`。详见 FNBENCH_AUDIT.md。README Apache-2.0 徽章与实际 MIT LICENSE 冲突，保持原文件且如实记录。

`python scripts/prepare_fnbench.py` 固定 SHA、只应用 `patches/fnbench-lazy-agnews.patch`；在 CIFAR 路径避免无关 torchtext 依赖。上游 `data/loader.py` 是实际路径。

`p1/fnbench.py` 是外部数据/日志适配器，只用原 BaseLocalUpdate、LocalModelWeights、官方 build_model 与分区函数，不改 optimizer/loss/aggregation。官方 ResNet18 返回 tuple，离线评估沿用官方 normalization。global load_state_dict 后保存 CPU 权重（含BN buffers）；无 train/test eval。

官方组件已做 synthetic CPU 两轮 smoke 及相同输入/RNG 的仪表化等价测试；没有执行原 main CUDA 入口或真实 CIFAR。新增 holdout 与完整设seed是明确实验控制，与原 main 的完整随机数消费不保证一致。CPU 成功不等价于官方论文复现或 V100 通过。

入口配置 `configs/phase0_v100.json`（clean train），IID control 和独立20% training noise 是后续需单独授权的配置。FedNoRo 未适配。所有当前结果仅工程调试，不允许宣称完成论文级验证。
