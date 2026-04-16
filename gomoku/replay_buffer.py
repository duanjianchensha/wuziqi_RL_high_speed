"""
gomoku/replay_buffer.py
经验回放池（FIFO，固定容量）

每条数据 = (state: np.ndarray(4,N,N), mcts_prob: np.ndarray(N*N), winner: float)

增强策略：存储原始样本，在 sample() 时随机选取 8 种几何变换之一。
相比在自弈时生成 8x 样本，此方案：
  1) buffer 实际容纳 8x 更多唯一局面（相同 buffer_size 对应原始局面数由1/8增至全量）
  2) 同一 batch 内不会同时包含同一局面的 8 种变换，增强 batch 多样性
"""

import random
from collections import deque
from typing import Tuple, List
import numpy as np

from gomoku.config import config


def _augment_one(
    state: np.ndarray, prob: np.ndarray, winner: float
) -> Tuple[np.ndarray, np.ndarray, float]:
    """随机选取 8 种几何变换之一（4旋转 × 2翻转），按需增强一个样本。"""
    n = state.shape[1]
    flip = np.random.randint(2)
    rot = np.random.randint(4)
    prob_2d = prob.reshape(n, n)
    s = np.flip(state, axis=2) if flip else state
    p = np.flip(prob_2d, axis=1) if flip else prob_2d
    s = np.rot90(s, rot, axes=(1, 2)).copy()
    p = np.rot90(p, rot).flatten().copy()
    return s.astype(np.float32), p.astype(np.float32), winner


def _translate_one(
    state: np.ndarray, prob: np.ndarray, winner: float, max_shift: int = 2
) -> Tuple[np.ndarray, np.ndarray, float]:
    """对专家数据做随机平移增强（用于缓解位置过拟合）。

    平移逻辑：
      - 平移量从 [-max_shift, max_shift] 均匀采样（排除 (0,0)）
      - 超出棋盘边界的棋子被截断（相当于该棋子不存在）
      - ch2（最后落子）同步平移；若超出边界则清零
      - ch3（当前玩家常数面）不变
      - prob 中超出边界的动作清零后重新归一化；若归一化失败则回退原始 prob
    """
    n = state.shape[1]
    # 生成非零偏移
    dr = np.random.randint(-max_shift, max_shift + 1)
    dc = np.random.randint(-max_shift, max_shift + 1)
    if dr == 0 and dc == 0:
        return state.astype(np.float32), prob.astype(np.float32), winner

    new_state = np.zeros_like(state, dtype=np.float32)
    # 计算源/目标切片
    r_src = slice(max(0, -dr), min(n, n - dr))
    c_src = slice(max(0, -dc), min(n, n - dc))
    r_dst = slice(max(0, dr), min(n, n + dr))
    c_dst = slice(max(0, dc), min(n, n + dc))

    # 前两个通道（棋子位置）直接平移
    new_state[0, r_dst, c_dst] = state[0, r_src, c_src]
    new_state[1, r_dst, c_dst] = state[1, r_src, c_src]
    # ch2（最后落子单热）：同步平移，超出则清零
    new_state[2, r_dst, c_dst] = state[2, r_src, c_src]
    # ch3（当前玩家常数面）：不受位置影响，直接复制
    new_state[3] = state[3]

    # prob 平移：超出边界的动作概率置 0，然后归一化
    prob_2d = prob.reshape(n, n).astype(np.float32)
    new_prob_2d = np.zeros((n, n), dtype=np.float32)
    new_prob_2d[r_dst, c_dst] = prob_2d[r_src, c_src]
    # 只在合法（未被占据）的位置保留概率
    occupied = (new_state[0] + new_state[1]) > 0
    new_prob_2d[occupied] = 0.0
    total = new_prob_2d.sum()
    if total > 1e-8:
        new_prob_2d /= total
    else:
        # 极端情况：平移后几乎所有概率都超出边界，退回原始 prob
        new_prob_2d = prob_2d.copy()

    return new_state, new_prob_2d.flatten().copy(), winner


class ReplayBuffer:
    def __init__(self, capacity: int = config.BUFFER_SIZE):
        self._buf: deque = deque(maxlen=capacity)
        self.capacity = capacity

    def push(self, data: List[Tuple]) -> None:
        """批量加入一局游戏的所有原始 (state, prob, winner) 样本（无预增强）。"""
        self._buf.extend(data)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """随机采样 batch_size 条，每条应用一种随机几何变换，返回三个 numpy 数组。"""
        nbuf = len(self._buf)
        if nbuf == 0:
            raise ValueError("ReplayBuffer 为空，无法采样")
        k = min(batch_size, nbuf)
        alpha = float(getattr(config, "RECENCY_SAMPLE_ALPHA", 0.0))
        buf_list = list(self._buf)
        if alpha <= 0.0:
            batch = random.sample(buf_list, k)
        else:
            # 下标 0 最旧，nbuf-1 最新；指数权重使新棋谱被抽中概率更高
            idx = np.arange(nbuf, dtype=np.int64)
            denom = max(nbuf - 1, 1)
            w = np.exp(alpha * idx.astype(np.float64) / denom)
            w /= w.sum()
            chosen = np.random.choice(idx, size=k, replace=False, p=w)
            batch = [buf_list[int(i)] for i in chosen]
        aug = [_augment_one(s, p, w) for s, p, w in batch]
        states, probs, winners = zip(*aug)
        return (
            np.stack(states, axis=0).astype(np.float32),
            np.stack(probs, axis=0).astype(np.float32),
            np.array(winners, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self._buf)

    def ready(self, batch_size: int) -> bool:
        """缓冲池是否积累了足够的数据供训练。"""
        return len(self._buf) >= batch_size
