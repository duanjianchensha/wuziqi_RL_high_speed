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


class ReplayBuffer:
    def __init__(self, capacity: int = config.BUFFER_SIZE):
        self._buf: deque = deque(maxlen=capacity)
        self.capacity = capacity

    def push(self, data: List[Tuple]) -> None:
        """批量加入一局游戏的所有原始 (state, prob, winner) 样本（无预增强）。"""
        self._buf.extend(data)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """随机采样 batch_size 条，每条应用一种随机几何变换，返回三个 numpy 数组。"""
        batch = random.sample(self._buf, min(batch_size, len(self._buf)))
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
