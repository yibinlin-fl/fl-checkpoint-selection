# 验收清单（用户拿到 Codex 结果后逐项核对）

- [ ] 仓库有分支、环境报告、上游 FNBench commit；无联网时正确解释阻塞。
- [ ] `python -m pytest -q` 完全通过。
- [ ] CPU synthetic 2-round smoke 跑通；明确写“调试数据，不是科学证据”。
- [ ] 真实 CIFAR-10 的 train/val/test ID 不重叠，train val indices 已固化。
- [ ] 仅验证标签发生改变，同一训练轨迹、模型文件 SHA 和测试 predictions 不变。
- [ ] `validation_scores.csv` 只有 uniform/heterogeneous 选择器字段，不含 clean-scenario、test truth、clean truth、真实注入噪声率。
- [ ] `selected_models.csv` 不调用 `private_audit`、测试数据或原始干净标签。
- [ ] `audit` 在选模结束后生成 selection regret；包含最后轮和 clean 诊断参照。
- [ ] 同时有 micro 与 macro、CE、EMA 以及 Last checkpoint。
- [ ] 固定单位 pp，明确目标评估分布及 selection objective。
- [ ] 没有把模拟结果伪称真实 CIFAR/GPU 实验。
- [ ] 先交第一次报告/commit，再由用户授权执行 120 轮、3–5 seeds。
