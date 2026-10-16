# Phase-0 预注册研究协议（内部草案，冻结前核对）

## 研究问题
10-client FL，train/val 样本固定，网络参数 checkpoint 序列固定；均匀对称验证噪声与客户端异质对称验证噪声相比，是否诱发可重复的选择后测试后悔值（pp）？

## 数学 sanity check
C 分类，对称翻转至其他 C-1 类的概率为 η，有 E[Acc_noisy]=(1-Cη/(C-1))Acc_clean+η/(C-1)。若统一 η < (C-1)/C，排序期望不变。不能以“所有标签噪声必然让选模失效”为论文假设。现实中有限验证集产生统计误差；异质噪声 × domain/label skew 改变有效权重，可能改变排序，但不保证。

## 第 0 阶段
Synthetic CPU smoke：只校验信息边界/数据结构/是否能运行；不用于报告科学现象。

## 第 1 阶段
CIFAR-10、ResNet18、10 clients、Dirichlet α=0.5、FedAvg，先 1 seed 跑完真实 checkpoint 序列（120 轮，每 5 轮存）；3 个验证条件只改变标签，不影响训练轨迹。记录 noise 实际翻转率和类别分布。训练标签噪声设置必须另外定义并冻结，建议增加 clean train/noisy train 两种情况，不能把两组训练结果混为一组对照。预先固定目标评估分布（默认官方 CIFAR-10 test 的样本宏平均准确率，即每样本等权）。

## 评估与选择
同一 checkpoint 评估各 client validation。不允许用 clean 验证标签、实际标签注入比例、官方测试标签选 checkpoint。输出 selection-only 表后独立选择，再由 audit 读取 test oracle 算 regret。测 `last`、macro/micro noisy val accuracy、noisy CE、EMA；clean-val 是仅在独立事后 audit 中计算的诊断基准，不进入 deployment selector CSV。**注意当目标是 IID 测试分布而 val 受 Dirichlet 影响时，即使 val label clean 也可能引入 distribution-mismatch selection gap**，需要 IID/control 分区、目标权重匹配检查。

## 决策
- 倾向 NO-GO：已知简单方案消除主要 regret，或 regret≈0，或仅极端人为配置才出现；不做后验 cherry-pick。
- 继续审计：3–5 seeds、多算法、多数据下明显稳定的 regret，简单方法不够且有实际部署需求。参考内部门槛：≥2pp 且 4/5 seeds，非论文通用标准。
- 必须在正式立项前对比 FedES、Label Wave、Noisy Early Stopping、其他可靠选择与无验证 stopping，避免将已知结果当创新。
