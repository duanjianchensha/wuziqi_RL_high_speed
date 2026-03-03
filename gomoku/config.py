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
    BOARD_SIZE: int = 15  # 棋盘大小（15×15 便于 CPU 快速训练）
    N_IN_ROW: int = 5  # 五子棋胜利条件

    # ───────── 神经网络 ─────────
    # CPU 模式使用较小网络；迁移 GPU 后可调大
    N_FILTERS: int = 128  # 残差块滤波器数（平衡性能与容量：128对15x15通常已经足够）
    N_RES_BLOCKS: int = 7  # 残差块数量
    L2_CONST: float = 1e-4  # L2 正则化系数
    LR: float = 2e-3  # 初始学习率（重置以跳出局部最优）

    # ───────── MCTS ─────────
    C_PUCT: float = 5.0  # PUCT 探索常数
    N_PLAYOUT_TRAIN: int = 300  # 自弈时每步模拟次数（提速默认值）
    ENABLE_PLAYOUT_SCHEDULE: bool = True  # 是否启用按训练进度自动调度 playout
    PLAYOUT_EARLY: int = 200  # 前期 playout
    PLAYOUT_MID: int = 300  # 中期 playout
    PLAYOUT_LATE: int = 400  # 后期 playout
    PLAYOUT_STAGE1_RATIO: float = 0.30  # 前期结束进度比例
    PLAYOUT_STAGE2_RATIO: float = 0.70  # 中期结束进度比例
    N_PLAYOUT_EVAL: int = 800  # 评估时模拟次数
    N_PLAYOUT_HUMAN: int = 400  # Web 对战时（简单=200, 中=400, 难=800）
    DIRICHLET_ALPHA: float = 1.0  # Dirichlet 噪声 α（8x8 棋盘建议加大探索）
    DIRICHLET_EPS: float = 0.25  # 噪声混合比例
    TEMP_THRESHOLD: int = 15  # 前 N 步高温探索，之后 → 0

    # ───────── Web 难度配置 ─────────
    # Web界面不同难度对应的MCTS模拟次数倍数（相对于训练时的N_PLAYOUT_TRAIN）
    WEB_DIFFICULTY_MULTIPLIERS: dict = {
        "easy": 0.6,  # 简单：训练模拟次数的0.6倍
        "medium": 1.0,  # 中等：训练模拟次数的1.0倍
        "hard": 1.8,  # 困难：训练模拟次数的1.8倍
    }
    # 或者直接指定绝对模拟次数（优先级高于倍数，如果设置了非None值）
    WEB_DIFFICULTY_PLAYOUTS: dict = {
        "easy": None,  # None表示使用倍数计算，设置数字则使用绝对值
        "medium": None,
        "hard": None,
    }

    # ───────── 自弈与训练 ─────────
    N_SELFPLAY_GAMES: int = 5000  # 总自弈局数达到此值前持续迭代
    BUFFER_SIZE: int = 50000  # 回放池容量（增加数据多样性和学习跨度）
    BATCH_SIZE: int = 512  # 训练 mini-batch 大小（增加单次训练质量）
    EPOCHS_PER_UPDATE: int = 10  # 每次更新时训练 epoch 数（确保样本被消化）
    KL_TARGET: float = 0.02  # 自适应学习率 KL 散度目标
    LR_MULTIPLIER: float = 1.0  # 学习率动态倍率（重置倍率）
    CHECK_FREQ: int = 50  # 每隔多少局评估一次（对比纯 MCTS）
    EVAL_GAMES: int = 10  # 评估时对战局数
    WIN_RATIO_THRESHOLD: float = 0.55  # 超过此胜率才更新 Best Model

    # ───────── 多进程自弈 ─────────
    # 工作进程数：默认使用 CPU 核心数（留 1 核给主进程）
    N_WORKERS: int = min(16, max(1, (os.cpu_count() or 2) // 2))
    GAMES_PER_WORKER: int = 1  # 每个 worker 每轮完成的局数

    # ───────── 硬件 ─────────
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    # Worker自弈推理时的设备。若设为设备如 "cuda"，多进程下会占用大量显存，此时建议降低 N_WORKERS (如 2-4)
    WORKER_DEVICE: str = "cpu"

    # ───────── 路径 ─────────
    MODEL_DIR: str = "models"
    BEST_POLICY: str = os.path.join("models", "best_policy.pth")
    CURRENT_POLICY: str = os.path.join("models", "current_policy.pth")
    CHECKPOINT_DIR: str = os.path.join("models", "checkpoints")


config = Config()

if __name__ == "__main__":
    print("=== Gomoku AlphaZero Config ===")
    for k, v in vars(Config).items():
        if not k.startswith("_"):
            print(f"  {k:30s} = {v}")
    print(f"\n  Device: {config.DEVICE}")
    print(f"  Workers: {config.N_WORKERS}")
