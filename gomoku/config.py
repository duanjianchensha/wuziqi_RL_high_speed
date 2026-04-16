"""
gomoku/config.py
全局超参数配置，支持 CPU-only 降级模式。
Phase 1: 纯 Python MCTS + PyTorch 神经网络
Phase 2 (TODO): C++ 高性能 MCTS 加速
"""

import os
import torch


class Config:
    # 硬件向参数可与本机同步：python scripts/recommend_train_params.py --json models/recommended_train.json

    # ───────── 棋盘 ─────────
    # 15×15 职业级人类极强；要达到/超越需长期自弈（通常 10^5~10^7 局量级）+ 高 playout 与足够算力。
    BOARD_SIZE: int = 15  # 棋盘大小（15×15 便于 CPU 快速训练）
    N_IN_ROW: int = 5  # 五子棋胜利条件

    # ───────── 神经网络 ─────────
    # CPU 模式使用较小网络；迁移 GPU 后可调大
    N_FILTERS: int = 128  # 残差块滤波器数（平衡性能与容量：128对15x15通常已经足够）
    N_RES_BLOCKS: int = 7  # 残差块数量
    L2_CONST: float = 1e-4  # L2 正则化系数
    LR: float = 1.5e-3  # 略降以配合更多 epoch/batch 步，减少震荡

    # ───────── MCTS ─────────
    C_PUCT: float = 5.0  # PUCT 探索常数
    # 自弈 playout 越高，MCTS 策略标签越准、学棋越快（更耗算力）。在 GPU 上建议 ≥400。
    N_PLAYOUT_TRAIN: int = 420
    ENABLE_PLAYOUT_SCHEDULE: bool = True  # 是否启用按训练进度自动调度 playout
    PLAYOUT_EARLY: int = 273  # 与 scripts/recommend_train_params.py（8G 显存档）一致
    PLAYOUT_MID: int = 420
    PLAYOUT_LATE: int = 504
    PLAYOUT_STAGE1_RATIO: float = 0.30  # 前期结束进度比例
    PLAYOUT_STAGE2_RATIO: float = 0.70  # 中期结束进度比例
    # 评估时双方使用相同次数；应 ≥ 训练主强度，否则「能下赢纯 MCTS」的判别力不足。
    N_PLAYOUT_EVAL: int = 567
    N_PLAYOUT_HUMAN: int = 400  # Web 对战时（简单=200, 中=400, 难=800）
    # AlphaZero 论文推荐: alpha ≈ 10 / N_actions = 10/225 ≈ 0.044；使用 0.03 减少初期过度均匀探索
    DIRICHLET_ALPHA: float = 0.03  # Dirichlet 噪声 α（15×15 棋盘，225个动作）
    DIRICHLET_EPS: float = 0.20  # 再略降：更快从「乱试」过渡到跟 MCTS 主策略
    TEMP_THRESHOLD: int = 20  # 前20步保持探索温度，覆盖更多开局变化（15×15平均局长~55步）

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
    # 池子过大时旧策略样本占比高，网络「追着过时标签」跑，棋力曲线显得不涨。
    # 缩小后 FIFO 更快被新自弈填满，策略迭代更猛（代价：长远多样性略减，可再调大）。
    BUFFER_SIZE: int = 13824
    BATCH_SIZE: int = 384  # 略减 batch → 同轮次内梯度步更多
    # >0 时采样偏向队列末尾（最新自弈），减轻「旧策略标签」拖累，利于持续自我超越；0=均匀随机
    RECENCY_SAMPLE_ALPHA: float = 0.5  # 降低新数据偏置（2.0 会造成回音壁，泛化差）
    EPOCHS_PER_UPDATE: int = 6  # 减少 epoch 防振荡（12 对小 batch 新数据会反复过拟合）
    KL_TARGET: float = (
        0.02  # 自适应 LR 的 KL 目标：KL>0.04 降 LR，KL<0.01 升 LR
    )
    USE_ADAPTIVE_LR: bool = (
        True  # 启用 KL 自适应学习率，KL 过高时自动踩刹车防振荡
    )
    LR_MULTIPLIER: float = 1.0  # 学习率动态倍率（重置倍率）
    # LR 热身：前 LR_WARMUP_GAMES 局内学习率从 LR*0.1 线性升至 LR，
    # 避免模型在经验极少时接受过大参数更新（早期数据噪声高）。
    LR_WARMUP_GAMES: int = 180  # 更快升到满 LR，前期少拖节奏
    # LR 阶梯衰减：与 playout 调度联动，训练进度推进时同步降 LR，
    # 避免中后期 LR 过大导致 loss 振荡。
    # 前期 (0~30%): LR×1.0；中期 (30~70%): LR×0.5；后期 (70%+): LR×0.25
    LR_SCHEDULE: bool = True
    CHECK_FREQ: int = 6  # 略增评估频率，便于早保存「历史最优」权重
    EVAL_GAMES: int = 30  # 增大局数降低方差（14局标准误≈±13%，30局≈±9%）
    WIN_RATIO_THRESHOLD: float = 0.55  # 强阈值：用于日志；实际保存见 MONOTONIC_BEST_SAVE
    # True：评估胜率只要创新高就写入 best_policy（避免长期达不到 0.55 导致永不存盘）
    MONOTONIC_BEST_SAVE: bool = True
    # 每轮训练后对「弱」纯 MCTS 快评（局数少、模拟低），日志里更早看到胜率抬头；
    # 正式 CHECK_FREQ 评估仍用 N_PLAYOUT_EVAL，两者分工不同。
    ENABLE_QUICK_PROGRESS_EVAL: bool = True
    QUICK_EVAL_PLAYOUT: int = 200  # 提高快评难度，避免每轮都 4/4 满分导致 best_policy 持续被覆盖
    QUICK_EVAL_GAMES: int = 4
    # 每 N 轮训练后才跑快评（1=每轮；2=隔轮，省时间）
    QUICK_EVAL_EVERY_N_ITER: int = 1
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
    # 与 recommend_train_params 8G 显存档对齐：并行 worker 上限 3，且不超过 CPU 核数-1
    N_WORKERS: int = min(
        3,
        min(8, max(1, (os.cpu_count() or 2) - 1)),
    )
    GAMES_PER_WORKER: int = 6  # 每轮每 worker 多几局，提高样本吞吐
    # 当 WORKER_DEVICE="cuda" 时，每个 worker 独立占用一份显存。
    # 此上限防止过多 worker 导致 GPU OOM（设为 None 则不限制）。
    MAX_CUDA_WORKERS: int = 3
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
    # 训练状态持久化：保存 lr_mult / n_games_played，断点恢复训练时继续上次进度。
    TRAIN_STATE_PATH: str = os.path.join("models", "train_state.json")
    # 规则策略自弈数据目录（scripts/gen_rule_data.py 生成，供监督预训练与混合回放）
    RULE_DATA_DIR: str = os.path.join("models", "human_games", "rule_data")
    # 自弈阶段混合回放：每个训练 batch 中规则数据占比，0=关闭
    MIX_EXPERT_REPLAY_RATIO: float = 0.15

    # ───────── Web 界面偏好 ─────────
    WEB_UI_PREFS_PATH: str = os.path.join("models", "web_ui_prefs.json")


config = Config()

if __name__ == "__main__":
    print("=== Gomoku AlphaZero Config ===")
    for k, v in vars(Config).items():
        if not k.startswith("_"):
            print(f"  {k:30s} = {v}")
    print(f"\n  Device: {config.DEVICE}")
    print(f"  Workers: {config.N_WORKERS}")
