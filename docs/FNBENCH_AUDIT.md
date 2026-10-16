# FNBench 源码审计（2026-10-17；Phase-0）

## 已核实事实
- 官方仓库 https://github.com/Sprinter1999/FNBench ，本地成功完整克隆；上游 SHA `838dff3023335c015dea07fa13dc42ca971d525d`，上游提交时间 2026-02-25。初始 vendor git status 干净。
- README 标记 Apache-2.0，但该 SHA 的 LICENSE 全文是 MIT License，Copyright (c) 2025 Xuefen。以实际文件留存；不得将 Apache 徽章当作许可证正文。
- 入口 main_fed_LNL.py；真实 loader 是 data/loader.py，不存在 data/datasets/loader.py。
- utils/options.py：epochs=120 表示 global rounds；num_users=100；frac=.1；local_ep=5；local_bs=64；lr=.01；momentum=.9；weight_decay=.0005。main 强制 all_clients=False，m=max(int(frac*num_users),1)，np.random.choice 不放回，保留采样顺序。
- main 在 CUDA 不可用时主动 exit，虽有 --gpu -1，CPU 机器仍被这个独立检查拦截。
- 分区 data/partition.py：IID 不整除会丢弃余数；Dirichlet 每类 np.random.seed(k)，故 assignment 并非由实验 seed 完全控制；min_size>=100 重试，小数据可能无限循环。最小适配保持官方分区；synthetic 只能使用 IID 且整除。
- 没有独立 val；train/test 分别是官方 CIFAR train/test。训练增强 RandomCrop/HorizontalFlip，Normalize ImageNet mean/std；test 无随机增强。
- noise_group_num 求和必须等于 num_users；单个 group_noise_rate 被复制为上下限；每组 rate=min+i*(max-min)/n（不包含上界）。抽取 floor(n*rate) 个 train indices，再用 noisify_label 翻至其他类别。该 CLI 是训练标签噪声，绝非 validation 噪声。输入长度检查的逻辑不能可靠覆盖所有错误组合。
- BaseLocalUpdate 使用 CE，SGD(lr,momentum,weight_decay)，每次本地调用重新创建 optimizer，DataLoader shuffle=True/drop_last=True；每 client 持有 net1/net2。is_svd_loss=False 时没有附加表示正则。
- LocalModelWeights 在参与者之间按 train 样本数 FedAvg；若各 client 等长则将所有权重缩放为1。FedAvg 对所有 state 项包括 BN 整数 counter 加权后 torch.div；load_state_dict 再转换 counter dtype。独立 starter 对整数取首 client，二者不等价，正式适配必须调用官方聚合。
- model_arch/model_resnet.py 是自定义 ResNet18，BasicBlock [2,2,2,2]、3x3 stride1 stem，forward 返回 (logits,features)，不能用 starter torchvision ResNet 冒充。
- main 每轮 aggregation/load_state_dict 后调用 test_img(test_loader)，保存 test acc/loss/precision/recall/F1 列表，打印最后10轮均值；FedAvg 路径没有以 test 指标更新 optimizer、early stop 或选最佳 checkpoint。该路径没有 torch.save 全局 checkpoint。
- 上游 main 只显式设 numpy seed，没有完整设置 Python/Torch seed。真实入口并不能仅凭 --seed 保证轨迹复现。
- 本机直接 import flcore.update 实际报 ModuleNotFoundError: torchtext。utils/__init__.py -> data/loader.py -> AGNews 顶层 import 是原因，CIFAR 不需要该模块。
- requirements.txt 同时要求 torch==2.0.0、torchvision==0.11.1+cu113，版本组合不配套；不能直接当作可安装的 V100 锁文件。

## 最小补丁及源码推断
- 先写此审计，再将 AGNews import 移入 AGNews 分支；不修改任何算法。补丁存于 patches/，vendor 保留完整上游文件和 LICENSE。
- 采用外部 FedAvg adapter 调用原 BaseLocalUpdate、LocalModelWeights、build_model 和分区函数；只提供独立 holdout、checkpoint、manifest、冻结选模后 test audit。
- 分区后固定每客户端80/20 holdout；训练样本减少是预注册数据控制，不能称为原始50k训练结果的数值复现。
- 额外完整设 seed 是明确的复现控制；与原 main 的随机数消费、移除 test DataLoader 遍历后的随机数状态不保证逐位一致。测试对相同隔离数据与 RNG 状态的官方组件路径做仪表化前后比较。
- 没有使用测试准确率进行实际 checkpoint 选择；删去训练中 test 访问是隔离要求。私有目录只是工程边界，不是针对恶意用户的操作系统权限隔离。

## 目前未验证项
- 尚未运行 CIFAR-10、V100、正式120轮、FedNoRo、多seed；不能声称与作者论文数值复现或支持P1假设。
- 官方 CIFAR bytes 的完整性由 torchvision 数据集 MD5 验证，未下载数据，本机只测试 source namespace 和 synthetic indices 断言；真实拆分需在获准后生成并复核。
- V100 安装与吞吐尚未实测，正式时间只能在获准 pilot 后估计。
