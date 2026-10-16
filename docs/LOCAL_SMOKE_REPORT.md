# 起始工程的打包前测试（2026-10-17）

环境：本工具执行环境，PyTorch 2.10.0+cpu / torchvision 0.25.0+cpu / pytest 9.0.2。

真实执行：

```bash
python -m pytest -q
# 12 passed in 2.58s
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python scripts/p1_cli.py all --config configs/phase0_smoke.json
```

烟测日志：

```text
[train] round=1/2 train_loss=2.3016
[train] round=2/2 train_loss=2.2987
Training complete; TEST SET NOT EVALUATED during training.
[offline-evaluate] round=1 completed
[offline-evaluate] round=2 completed
Offline scoring saved: public selection signals and separate private audit.
Selection complete. This module did NOT read private_audit files.
Audited decision table saved. Test oracle used AFTER all choices are fixed.
Saved validation_accuracy.png
```

**非常重要**：以上仅为 `synthetic-debug` 两轮 CPU 工程自测，不能证明 P1 的科研假设，也不能当作真实 CIFAR-10/FNBench 论文实验。下次换系统需要重跑以校验环境。
