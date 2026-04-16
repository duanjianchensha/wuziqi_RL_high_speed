"""
train.py — 训练入口

用法:
  python train.py               # 默认参数训练
  python train.py --games 200   # 训练 200 局
  python train.py --games 1000 --playout 600  # 更多局数 + 更多模拟
"""

import argparse
import multiprocessing as mp
import os
import json
import sys


def parse_args():
    p = argparse.ArgumentParser(description="五子棋 AlphaZero 训练")
    p.add_argument(
        "--games",
        type=int,
        default=None,
        help="总自弈局数（默认读 config.N_SELFPLAY_GAMES）",
    )
    p.add_argument(
        "--playout",
        type=int,
        default=None,
        help="每步 MCTS 模拟次数（默认读 config.N_PLAYOUT_TRAIN）",
    )
    p.add_argument(
        "--workers", type=int, default=None, help="并行 worker 数（默认 CPU核心数-1）"
    )
    p.add_argument("--batch", type=int, default=None, help="训练 batch size")
    p.add_argument(
        "--resume", action="store_true", help="从 best_policy.pth 继续训练（默认行为）"
    )
    p.add_argument(
        "--fresh",
        action="store_true",
        help="不读取 train_state.json（局数/lr_mult 等），网页一键训练会附带本参数",
    )
    p.add_argument(
        "--mix-expert-ratio",
        type=float,
        default=None,
        help="混合回放：每 batch 中规则数据占比 0~1，默认读 config.MIX_EXPERT_REPLAY_RATIO",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # 在 import 之前应用命令行覆盖
    from gomoku.config import config

    if args.games:
        config.N_SELFPLAY_GAMES = args.games
    if args.playout:
        config.N_PLAYOUT_TRAIN = args.playout
        config.ENABLE_PLAYOUT_SCHEDULE = False
    if args.workers:
        config.N_WORKERS = args.workers
    if args.batch:
        config.BATCH_SIZE = args.batch
    if args.mix_expert_ratio is not None:
        config.MIX_EXPERT_REPLAY_RATIO = max(
            0.0, min(1.0, float(args.mix_expert_ratio))
        )

    # 持久化本次训练配置，供 Web 端动态难度读取
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    profile_path = os.path.join(config.MODEL_DIR, "latest_train_profile.json")
    with open(profile_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "n_selfplay_games": int(config.N_SELFPLAY_GAMES),
                "n_playout_train": int(config.N_PLAYOUT_TRAIN),
                "n_workers": int(config.N_WORKERS),
                "batch_size": int(config.BATCH_SIZE),
                "mix_expert_replay_ratio": float(config.MIX_EXPERT_REPLAY_RATIO),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # 配置日志：同时输出到控制台和文件（train.log）
    import logging

    log_path = os.path.join(config.MODEL_DIR, "train.log")
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_path, encoding="utf-8"),
    ]
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s", handlers=handlers
    )
    logger = logging.getLogger(__name__)

    # 兼容未改写 Coach 中 print 的输出，将 stdout tee 到日志文件
    class _Tee:
        def __init__(self, *streams):
            self.streams = streams

        def write(self, data):
            for s in self.streams:
                try:
                    s.write(data)
                except Exception:
                    pass

        def flush(self):
            for s in self.streams:
                try:
                    s.flush()
                except Exception:
                    pass

    sys.stdout = _Tee(sys.stdout, open(log_path, "a", encoding="utf-8"))
    sys.stderr = _Tee(sys.stderr, open(log_path, "a", encoding="utf-8"))

    # 打印配置摘要
    logger.info("%s", "=" * 55)
    logger.info("%s", "  五子棋 AlphaZero 训练 (Phase 1: Pure Python MCTS)")
    logger.info("%s", "=" * 55)
    logger.info("  棋盘大小:       %dx%d", config.BOARD_SIZE, config.BOARD_SIZE)
    logger.info("  获胜条件:       %d 连子", config.N_IN_ROW)
    logger.info("  训练目标局数:   %d", config.N_SELFPLAY_GAMES)
    logger.info("  每步MCTS模拟:   %d", config.N_PLAYOUT_TRAIN)
    logger.info("  并行Worker数:   %d", config.N_WORKERS)
    logger.info("  BatchSize:      %d", config.BATCH_SIZE)
    if float(getattr(config, "MIX_EXPERT_REPLAY_RATIO", 0.0)) > 0:
        logger.info(
            "  混合回放:       规则数据占比=%.0f%%（无 npz 时自动纯自弈）",
            100.0 * float(config.MIX_EXPERT_REPLAY_RATIO),
        )
    else:
        logger.info("  混合回放:       关闭")
    logger.info("  设备:           %s", config.DEVICE)
    logger.info("%s", "=" * 55)

    from gomoku.coach import Coach

    coach = Coach(fresh_start=bool(args.fresh))
    coach.run(config.N_SELFPLAY_GAMES)


if __name__ == "__main__":
    # Windows 多进程必需的 spawn 保护
    mp.set_start_method("spawn", force=True)
    main()
