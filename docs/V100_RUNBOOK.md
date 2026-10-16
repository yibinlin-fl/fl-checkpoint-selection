# V100 运行手册（候选环境；尚未在 V100 实测）

本次只交付 Phase-0；下列安装、数据下载和正式训练命令均未执行。请先验收，然后单独授权。不要一次运行多个条件或多个 seeds。

## 环境安装
Linux、Python 3.10、建议 V100 16GB 或以上。使用独立环境，不覆盖系统环境：

```bash
python3.10 -m venv .venv-v100
source .venv-v100/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements-v100.txt
python scripts/prepare_fnbench.py
python scripts/check_v100.py
python scripts/record_environment.py > v100_environment.json
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest -q
```

PyTorch/torchvision 配对命令依据 [PyTorch 官方历史版本安装页](https://pytorch.org/get-started/previous-versions/#v251)。这是面向 Volta 的保守候选版本，不使用上游不配套 requirements，不需要 CIFAR 路径无关的 torchtext。`check_v100.py` 必须验证 wheel 有 sm_70 并实际执行 CUDA 运算；仅 nvidia-smi 能运行不足以通过。当前开发机 torch2.12 CPU 的通过结果不代表此候选 GPU 环境已验收。

## 下载（只在后续获得授权时）
官方 CIFAR-10 压缩包约170MB，展开和后续产物需要额外磁盘空间；不会下载其他大型数据集。正式配置 download_data=false，缺数据即报错；不要在训练中静默下载。

```bash
python -c "from torchvision.datasets import CIFAR10; CIFAR10('data_cache', train=True, download=True)"
```

torchvision 官方 loader 执行 MD5 完整性校验。下载归档包含 train/test，但训练过程只实例化 train=True，audit 才实例化 train=False。

## 下一条正式命令（本次没有执行）

```bash
bash scripts/run_v100.sh --approved-formal-run
```

脚本进行只读 GPU preflight、精确版本准备，然后调用：

```bash
python scripts/p1_cli.py all --config configs/phase0_v100.json
```

该配置使用官方 FNBench ResNet18 / BaseLocalUpdate / LocalModelWeights：10 clients，全部参与、Dirichlet alpha=.5、客户端固定80/20 holdout、seed42、clean training labels、120 rounds、每轮1 local epoch、bs64、lr.01、momentum.9、weight_decay.0005、无额外 loss/表示正则/学习率 schedule。官方默认 local_ep=5，这里按原预注册 starter 草案明确设为1；是实验配置选择，不是修改算法。全局 round 一基记录，checkpoints 为5,10,...,120共24个；Last 为保存的120轮。

数据规模约40k训练/10k验证（逐 client rounding），官方独立10k test。约120 x sum_client floor(n_train_client/64) 次 SGD 更新（约7.4万，精确值以实际 manifest 为准）；每个 checkpoint 三种验证视图与 test audit；24份 FP32 ResNet18 checkpoint 约1.1GB，环境、数据和临时产物另计，建议至少10GB空闲。没有实测 V100 吞吐，不能给出可信小时数。

首次获准应先执行**另行批准的短 CIFAR pilot**校验环境、实际split和每轮耗时，再决定是否执行上面完整命令；本次没有执行 pilot，也没有训练真实 CIFAR。

## 分步检查和复用
```bash
python scripts/p1_cli.py train --config configs/phase0_v100.json
python scripts/p1_cli.py evaluate --config configs/phase0_v100.json
python scripts/p1_cli.py select --config configs/phase0_v100.json
python scripts/p1_cli.py audit --config configs/phase0_v100.json
python scripts/p1_cli.py plot --config configs/phase0_v100.json
```

不要在 all 后重复 train/evaluate/select；冻结结果拒绝重选。每次运行必须使用新的 output_dir，可用 `--out outputs/unique_run_name`，不覆盖或清理旧结果。evaluate 只算 uniform/heterogeneous 指标；select 读取 CSV、使用 config_used 中冻结 EMA；audit 在选择冻结校验后才加载 test/clean diagnostic。

IID 对照 `configs/phase0_v100_iid.json` 和独立20%训练噪声条件 `configs/phase0_v100_trainnoise20.json` 已备好，仍需后续授权，不能据已有 regret 调 alpha/seed/噪声。训练20%是官方 floor(n*.2) 对称翻转；验证噪声是独立 Bernoulli 视图，两者不能混淆。uniform 验证率严格按客户端 val 样本数匹配 heterogeneous 预期率；实际翻转率可能因抽样不同。目标预先固定为官方 IID test，每样本等权，clean val 只作为分布偏移诊断上界。FedNoRo 尚未适配/运行。
