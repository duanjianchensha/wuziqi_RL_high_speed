"""
pretrain.py — 规则预训练入口

使用规则策略自弈数据（models/human_games/rule_data）对神经网络做监督预训练，
为后续 AlphaZero 自弈提供更高的起点。

用法:
  python pretrain.py                                     # 默认参数
  python pretrain.py --epochs 80 --batch 512
  python pretrain.py --data-dir models/human_games/rule_data --out models/current_policy.pth
  python pretrain.py --from-model models/best_policy.pth # 从指定权重热身继续
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    from gomoku.config import config
    from gomoku.data_utils import load_npz_files
    from gomoku.neural_net import PolicyValueFunction

    p = argparse.ArgumentParser(description="规则数据监督预训练")
    p.add_argument(
        "--data-dir",
        default=config.RULE_DATA_DIR,
        help=f"game_*.npz 所在目录（默认: {config.RULE_DATA_DIR}）",
    )
    p.add_argument("--epochs", type=int, default=80, help="训练轮数（默认 80）")
    p.add_argument("--batch", type=int, default=None, help="batch size（默认用 config.BATCH_SIZE）")
    p.add_argument("--lr", type=float, default=None, help="学习率（默认用 config.LR）")
    p.add_argument(
        "--out",
        default=config.CURRENT_POLICY,
        help=f"输出权重路径（默认: {config.CURRENT_POLICY}）",
    )
    p.add_argument(
        "--from-model",
        default=None,
        help="初始权重路径；默认优先 current_policy.pth，其次 best_policy.pth，否则随机初始化",
    )
    args = p.parse_args()

    batch_cfg = int(args.batch or config.BATCH_SIZE)
    lr = float(args.lr or config.LR)
    data_dir = args.data_dir

    # ── 加载数据 ────────────────────────────────────────────────
    states, probs, zs = load_npz_files(data_dir)
    n = int(states.shape[0])
    if n == 0:
        print(
            f"[pretrain] 目录无有效数据: {data_dir}\n"
            "请先运行: python scripts/gen_rule_data.py",
            flush=True,
        )
        sys.exit(1)

    # 样本数少于 batch 时自动缩放
    batch = min(batch_cfg, n)
    if batch == n and n > 1:
        batch = min(max(8, n // 4), n)
    if batch < batch_cfg:
        print(
            f"[pretrain] 样本数={n}，batch 从 {batch_cfg} 调整为 {batch}",
            flush=True,
        )

    # ── 加载模型 ────────────────────────────────────────────────
    load_path = args.from_model
    if load_path is None:
        if os.path.exists(config.CURRENT_POLICY):
            load_path = config.CURRENT_POLICY
        elif os.path.exists(config.BEST_POLICY):
            load_path = config.BEST_POLICY

    print(
        f"[pretrain] 数据目录={data_dir}  样本={n}  epochs={args.epochs}  batch={batch}  lr={lr}",
        flush=True,
    )
    policy = PolicyValueFunction(model_path=load_path if load_path else None)
    if load_path:
        policy.reset_optimizer()
        print(
            f"[pretrain] 已加载权重: {load_path}（优化器已重置，防止旧 Adam 状态拉偏 loss）",
            flush=True,
        )
    else:
        print("[pretrain] 随机初始化网络权重", flush=True)

    # ── 训练循环 ────────────────────────────────────────────────
    rng = np.random.default_rng(42)
    for ep in range(args.epochs):
        perm = rng.permutation(n)
        total_loss = 0.0
        steps = 0
        for start in range(0, n, batch):
            idx = perm[start: start + batch]
            info = policy.train_step(states[idx], probs[idx], zs[idx], lr=lr)
            total_loss += info["loss"]
            steps += 1
        avg = total_loss / max(steps, 1)
        print(f"  epoch {ep + 1:3d}/{args.epochs}  mean_loss={avg:.4f}", flush=True)

    # ── 保存权重 ────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    policy.save(args.out)
    print(f"[pretrain] 已保存: {args.out}", flush=True)
    print("下一步: python train.py  （可附加 --resume 使用 best_policy 继续自弈）", flush=True)


if __name__ == "__main__":
    main()
