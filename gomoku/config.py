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
    # AlphaZero 论文推荐: alpha ≈ 10 / N_actions = 10/225 ≈ 0.044；使用 0.03 减少初期过度均匀探索
    DIRICHLET_ALPHA: float = 0.03  # Dirichlet 噪声 α（15×15 棋盘，225个动作）
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
    BATCH_SIZE: int = 1024  # 训练 mini-batch 大小（增加单次训练质量）
    EPOCHS_PER_UPDATE: int = 20  # 每次更新时训练 epoch 数（确保样本被消化）
    KL_TARGET: float = 0.02  # 自适应学习率 KL 散度目标
    LR_MULTIPLIER: float = 1.0  # 学习率动态倍率（重置倍率）
    # LR 热身：前 LR_WARMUP_GAMES 局内学习率从 LR*0.1 线性升至 LR，
    # 避免模型在经验极少时接受过大参数更新（早期数据噪声高）。
    LR_WARMUP_GAMES: int = 300
    CHECK_FREQ: int = 10  # 每隔多少局评估一次（对比纯 MCTS）
    EVAL_GAMES: int = 10  # 评估时对战局数
    WIN_RATIO_THRESHOLD: float = 0.55  # 超过此胜率才更新 Best Model
    # 梯度裁剪：防止梯度爆炸导致 NaN loss（尤其在训练早期或恢复训练时）
    GRAD_CLIP: float = 5.0

    # ───────── 认输机制（Resignation）─────────
    # 当 MCTS 根节点价值估计（己方视角）连续 RESIGN_PATIENCE 步低于阈值时认输。
    # 可显著缩短明局势已定的局（节省 30-50% 自弈时间）。
    # 设为 None 禁用认输。
    RESIGN_THRESHOLD: float = -0.85  # 低于此值认为必输
    RESIGN_PATIENCE: int = 5  # 连续多少步才认输（避免误判）
    RESIGN_MIN_MOVE: int = 30  # 至少走满多少步后才允许认输
    # 以 NO_RESIGN_PROB 概率禁用本局认输，用于检测假阳性误认输
    NO_RESIGN_PROB: float = 0.1

    # ───────── 多进程自弈 ─────────
    # 工作进程数：默认使用 CPU 核心数（留 1 核给主进程）
    N_WORKERS: int = min(12, max(1, (os.cpu_count() or 2) - 2))
    GAMES_PER_WORKER: int = (
        5  # 每个 worker 每轮完成的局数    # 当 WORKER_DEVICE="cuda" 时，每个 worker 独立占坐一份显存。
    )
    # 此上限防止过多 worker 导致 GPU OOM（设为 None 则不限制）。
    MAX_CUDA_WORKERS: int = (
        6  # 批量叶节点推理：每次收集 LEAF_BATCH_SIZE 个叶节点后一次性调用 NN，
    )
    # 避免 300 次单独 forward 的 GPU 启动开销。建议值：16-32。
    LEAF_BATCH_SIZE: int = 16

    # ───────── 硬件 ─────────
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    # Worker自弈推理设备。CUDA可大幅提升worker推理速度，但每个worker占用一份显存。
    # 若显存有限（<8GB）或worker数>4，建议保持"cpu"或降低N_WORKERS至2-4。
    WORKER_DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    # ───────── 路径 ─────────
    MODEL_DIR: str = "models"
    BEST_POLICY: str = os.path.join("models", "best_policy.pth")
    CURRENT_POLICY: str = os.path.join("models", "current_policy.pth")
    CHECKPOINT_DIR: str = os.path.join("models", "checkpoints")
    # 训练状态持久化：保存 lr_mult / n_games_played，寞机恢复训练时继续上次进度。
    TRAIN_STATE_PATH: str = os.path.join("models", "train_state.json")


config = Config()

if __name__ == "__main__":
    print("=== Gomoku AlphaZero Config ===")
    for k, v in vars(Config).items():
        if not k.startswith("_"):
            print(f"  {k:30s} = {v}")
    print(f"\n  Device: {config.DEVICE}")
    print(f"  Workers: {config.N_WORKERS}")
