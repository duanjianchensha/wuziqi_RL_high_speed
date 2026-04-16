# 训练对比基准（自动生成）

运行（项目根目录、conda 环境 `ship`）：

```powershell
python -u scripts/quick_train_compare.py --fast
```

完整对比（更久）：

```powershell
python -u scripts/quick_train_compare.py --games 200 --playout 200
```

仅复测已有 `final_policy.pth`（需上次完整跑过并生成各目录下的 `last_train_metrics.json`）：

```powershell
python -u scripts/quick_train_compare.py --eval-only --eval-playout 160 --eval-games 10
```

通过条件（任一满足即退出码 0）：对纯 MCTS 胜率更高、或「新 vs 旧」头对头胜率 > 0.5、或末轮训练 loss 更低（更好拟合当前回放池）。

结果摘要：`last_compare_result.json`。
