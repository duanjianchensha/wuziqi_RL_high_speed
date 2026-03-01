"""
train.py — 训练入口

用法:
  python train.py               # 默认参数训练
  python train.py --games 200   # 训练 200 局
  python train.py --games 1000 --playout 600  # 更多局数 + 更多模拟
"""

import argparse
import multiprocessing as mp


def parse_args():
    p = argparse.ArgumentParser(description="五子棋 AlphaZero 训练")
    p.add_argument("--games",   type=int, default=None,
                   help="总自弈局数（默认读 config.N_SELFPLAY_GAMES）")
    p.add_argument("--playout", type=int, default=None,
                   help="每步 MCTS 模拟次数（默认读 config.N_PLAYOUT_TRAIN）")
    p.add_argument("--workers", type=int, default=None,
                   help="并行 worker 数（默认 CPU核心数-1）")
    p.add_argument("--batch",   type=int, default=None,
                   help="训练 batch size")
    p.add_argument("--resume",  action="store_true",
                   help="从 best_policy.pth 继续训练（默认行为）")
    return p.parse_args()


def main():
    args = parse_args()

    # 在 import 之前应用命令行覆盖
    from gomoku.config import config
    if args.games:
        config.N_SELFPLAY_GAMES = args.games
    if args.playout:
        config.N_PLAYOUT_TRAIN = args.playout
    if args.workers:
        config.N_WORKERS = args.workers
    if args.batch:
        config.BATCH_SIZE = args.batch

    # 打印配置摘要
    print("=" * 55)
    print("  五子棋 AlphaZero 训练 (Phase 1: Pure Python MCTS)")
    print("=" * 55)
    print(f"  棋盘大小:       {config.BOARD_SIZE}×{config.BOARD_SIZE}")
    print(f"  获胜条件:       {config.N_IN_ROW} 连子")
    print(f"  训练目标局数:   {config.N_SELFPLAY_GAMES}")
    print(f"  每步MCTS模拟:   {config.N_PLAYOUT_TRAIN}")
    print(f"  并行Worker数:   {config.N_WORKERS}")
    print(f"  BatchSize:      {config.BATCH_SIZE}")
    print(f"  设备:           {config.DEVICE}")
    print("=" * 55)

    from gomoku.coach import Coach
    coach = Coach()
    coach.run(config.N_SELFPLAY_GAMES)


if __name__ == "__main__":
    # Windows 多进程必需的 spawn 保护
    mp.set_start_method("spawn", force=True)
    main()
