"""
gomoku/replay_buffer.py
经验回放池（FIFO，固定容量）

每条数据 = (state: np.ndarray(4,N,N), mcts_prob: np.ndarray(N*N), winner: float)
"""

import random
from collections import deque
from typing import Tuple, List
import numpy as np

from gomoku.config import config


class ReplayBuffer:
    def __init__(self, capacity: int = config.BUFFER_SIZE):
        self._buf: deque = deque(maxlen=capacity)
        self.capacity = capacity

    def push(self, data: List[Tuple]) -> None:
        """批量加入一局游戏的所有 (state, prob, winner) 样本（含8×扩充）。"""
        self._buf.extend(data)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """随机采样 batch_size 条，返回三个 numpy 数组。"""
        batch = random.sample(self._buf, min(batch_size, len(self._buf)))
        states, probs, winners = zip(*batch)
        return (
            np.stack(states,  axis=0).astype(np.float32),
            np.stack(probs,   axis=0).astype(np.float32),
            np.array(winners,         dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self._buf)

    def ready(self, batch_size: int) -> bool:
        """缓冲池是否积累了足够的数据供训练。"""
        return len(self._buf) >= batch_size
