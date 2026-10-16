# Phase-0 验收清单（2026-10-17）

- [x] 起始工程安全初始化为Git，保存原始版本，分支 phase0-audit。
- [x] 完整阅读任务书和所有核心代码/配置/测试/协议。
- [x] 官方FNBench成功获取，精确SHA和实际MIT LICENSE已记录。
- [x] 全部单测通过；实际数量/日志见 PHASE0_REPORT.md 和 evidence/。
- [x] 独立starter与官方FedAvg组件的2-round CPU synthetic smoke通过。
- [x] train/val互斥/覆盖；official_train与official_test源分开；mock CIFAR loader 验证访问路径。
- [ ] **真实CIFAR内容/实际拆分与V100运行尚未验证**，未获准下载或正式训练。
- [x] 验证噪声仅改变val view；同seed重放校验checkpoint/test predictions不变，见实际证据。
- [x] selector严格CSV白名单，只有uniform/heterogeneous；隔离目录不含config或private_audit也可选模。
- [x] test/clean diagnostic只能在冻结选择之后计算；包含Last及clean/test事后上界。
- [x] macro/micro accuracy、CE、EMA、Last均实现，regret单位pp。
- [x] synthetic图和报告有调试标注，无伪造CIFAR/GPU/创新性结论。
- [x] V100候选安装、单seed命令、IID/独立training noise配置备妥；尚未实测。
- [ ] 用户验收并决定是否批准下一阶段；当前仅 INCONCLUSIVE。
