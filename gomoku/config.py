"""
gomoku/config.py
全局超参数配置，支持 CPU-only 降级模式。
Phase 1: 纯 Python MCTS + PyTorch 神经网络
Phase 2 (TODO): C++ 高性能 MCTS 加速
"""
import os
import torch


class Config:
    # ───────── 棋盘 ─────────
    BOARD_SIZE: int = 8          # 棋盘大小（8×8 便于 CPU 快速训练）
    N_IN_ROW:   int = 5          # 五子棋胜利条件

    # ───────── 神经网络 ─────────
    # CPU 模式使用较小网络；迁移 GPU 后可调大
    N_FILTERS:      int = 64     # 残差块滤波器数（GPU 可改为 128/256）
    N_RES_BLOCKS:   int = 5      # 残差块数量
    L2_CONST:       float = 1e-4 # L2 正则化系数
    LR:             float = 1e-3 # 初始学习率

    # ───────── MCTS ─────────
    C_PUCT:          float = 5.0  # PUCT 探索常数
    N_PLAYOUT_TRAIN: int   = 400  # 自弈时每步模拟次数（CPU 降级；GPU 可调至 800）
    N_PLAYOUT_EVAL:  int   = 800  # 评估时模拟次数
    N_PLAYOUT_HUMAN: int   = 400  # Web 对战时（简单=200, 中=400, 难=800）
    DIRICHLET_ALPHA: float = 0.3  # Dirichlet 噪声 α
    DIRICHLET_EPS:   float = 0.25 # 噪声混合比例
    TEMP_THRESHOLD:  int   = 15   # 前 N 步高温探索，之后 → 0

    # ───────── 自弈与训练 ─────────
    N_SELFPLAY_GAMES:  int = 500   # 总自弈局数达到此值前持续迭代
    BUFFER_SIZE:       int = 10000 # 回放池容量
    BATCH_SIZE:        int = 256   # 训练 mini-batch 大小（GPU 可调至 512）
    EPOCHS_PER_UPDATE: int = 5     # 每次更新时训练 epoch 数
    KL_TARGET:         float = 0.02 # 自适应学习率 KL 散度目标
    LR_MULTIPLIER:     float = 1.0  # 学习率动态倍率
    CHECK_FREQ:        int = 50    # 每隔多少局评估一次（对比纯 MCTS）
    EVAL_GAMES:        int = 10    # 评估时对战局数
    WIN_RATIO_THRESHOLD: float = 0.55  # 超过此胜率才更新 Best Model

    # ───────── 多进程自弈 ─────────
    # 工作进程数：默认使用 CPU 核心数（留 1 核给主进程）
    N_WORKERS: int = max(1, (os.cpu_count() or 2) - 1)
    GAMES_PER_WORKER: int = 1    # 每个 worker 每轮完成的局数

    # ───────── 硬件 ─────────
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    # ───────── 路径 ─────────
    MODEL_DIR:       str = "models"
    BEST_POLICY:     str = os.path.join("models", "best_policy.pth")
    CURRENT_POLICY:  str = os.path.join("models", "current_policy.pth")
    CHECKPOINT_DIR:  str = os.path.join("models", "checkpoints")


config = Config()

if __name__ == "__main__":
    print("=== Gomoku AlphaZero Config ===")
    for k, v in vars(Config).items():
        if not k.startswith("_"):
            print(f"  {k:30s} = {v}")
    print(f"\n  Device: {config.DEVICE}")
    print(f"  Workers: {config.N_WORKERS}")
