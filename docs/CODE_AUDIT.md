# 核心工程审计（Phase-0）

范围：全部 p1 模块、scripts、configs、原始12项测试、README/研究协议/验收文档；官方 FedAvg 调用链另见 FNBENCH_AUDIT。未开展新算法设计。

## 功能与调用链
独立 starter：p1_cli -> training.run_training -> make_splits -> local_update(CE/SGD) -> fedavg -> global checkpoint。官方 backend：相同CLI -> fnbench.run_fnbench_training -> 官方partition -> 固定holdout -> 官方BaseLocalUpdate -> 官方LocalModelWeights.average -> checkpoint。后续两者共用 evaluate -> selection -> audit -> plot。

配置/数据：config 校验正整数、噪声率/EMA边界、数据集/后端、分区与模型。data.load_partition 明确 train 布尔源，CIFAR 默认不下载；synthetic 有独立seed/source namespace。splits 对全量覆盖、跨client唯一性、train/val互斥、test独立源作断言；manifest 固化索引、class histogram、target distribution。noise 用局部Generator，uniform按真实val数量加权，映射保存在private_audit。禁止改seed/alpha救结果。

## 发现与修复
1. 起始训练函数 load_images 同时加载 train/test（虽不使用 test）。拆为按source加载，训练和evaluate不实例化 test。通过mock loader与端到端访问守卫测试验证。
2. 起始 evaluate 在select之前生成test及clean分数，存在人为反馈风险。移到冻结后的audit；缺少或变更选择冻结文件拒绝audit。
3. 起始选择模块读完整config且import evaluation。拆出纯文件工具，选择器只读白名单CSV及自己的输出摘要，EMA仅接收冻结标量。用只有CSV的孤立目录实际测试选择成功。
4. 起始selector直调API绕过schema；NaN/Inf、重复、缺client/变n、scenario轮次不齐无校验。现在两入口统一校验。仅从CSV自身不能发现所有轮次都一致缺失的同一个client；正式scores由完整manifest循环导出。
5. 起始Last的selection_score为NaN，改为无数值的空字段。其余指标要求有限值。oracle额外拒绝重复和非有限accuracy。
6. 起始复用output会覆盖配置、混入陈旧checkpoint；改为非空目录拒绝训练。checkpoint集合/config/split哈希保存并在evaluate/audit验证；selected结果及scores哈希冻结，拒绝重选。
7. 起始CLI --out只影响下游不影响train，现全流程一致；select用config_used里的EMA，避免后来修改原配置改变规则。
8. 原图无synthetic警示；现显式 SYNTHETIC DEBUG ONLY。新train_log记录参与client及样本数。
9. Windows pytest默认临时目录写失败，tests/conftest.py使用工作区内全新目录，不触发旧目录递归清理。未批量删除任何文件；原outputs/smoke最初由原始基线命令更新，后续运行均使用新目录。
10. 官方与starter ResNet、normalization、weight_decay/drop_last、BN聚合不同，不能混称。正式配置切到官方组件；starter保留用于轻量调试。详见官方源码等价测试。
11. 官方无全局保存、每轮test、缺torchtext、依赖配对错误及不完整seed控制已记录；只对无关AGNews导入加最小补丁，数据与checkpoint外部适配。

## 科学有效性与残余风险
- 未下载/训练真实CIFAR，真实40k/10k split、官方test内容完整性、V100兼容性、性能/收敛尚待后续；不能把namespace断言或mock测试称为已验证真实数据无泄漏。
- CIFAR loader在实际使用时做官方MD5；test命名空间与train来源分离。离线evaluate持有clean训练来源标签用于生成噪声，但selector从不接收；私有审计数据不能用于人工挑选参数。
- FNBench原Dirichlet在每类内重置numpy seed，跨seed分区可能不变，小样本循环风险保留且显式说明；synthetic禁用其小样本Dirichlet路径，正式不通过事后调参数规避。
- assert是验收断言，请勿使用 python -O（会移除assert）。文件哈希不是防恶意篡改的安全系统。运行不支持中断恢复/追加，失败后使用新目录。
- 偏斜val与目标IID test存在分布不一致，clean val也可能有regret；已备IID配置，均未运行。uniform对称噪声在阈值内期望排序不变，有精确转移矩阵测试。
- 官方轮次配置120、local_ep=1与默认5有区别；单独20%训练噪声配置保留，其flip机制与validation Bernoulli机制明确分离。
- 没有实现FedNoRo、FedES、Label Wave或其他无验证方法；不声称具备完整论文baseline覆盖。当前决策只能INCONCLUSIVE。
