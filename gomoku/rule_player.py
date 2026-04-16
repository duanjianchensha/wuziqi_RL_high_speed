"""
gomoku/rule_player.py
基于规则的五子棋策略玩家，用于生成高质量监督预训练数据。

设计目标：
  - 轻量无 GPU，每局耗时 <1 秒（CPU）
  - 覆盖完整棋型威胁评估（活四/冲四/活三/眠三/活二/眠二）
  - 支持返回软概率分布，比 one-hot 标签提供更丰富的监督信号
  - 支持噪声注入，保证生成数据的多样性

典型用法（配合 scripts/gen_rule_data.py）：
  player = RulePlayer(noise_eps=0.15, score_temp=0.5)
  scores = get_action_scores(board)   # 全局评分向量
  probs  = scores_to_probs(scores, board.availables, temp=0.5)
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple

from gomoku.config import config


# ────────────────────────────────────────────────
# 常量
# ────────────────────────────────────────────────

DIRECTIONS = [(1, 0), (0, 1), (1, 1), (1, -1)]

# 防守权重：稍高于 1.0，表示防守优先于同等强度的进攻
BLOCK_WEIGHT = 1.1

# 棋型得分表  key=(连子数, 开放端数)
# 连子数以 n_in_row 为上限；开放端数 0/1/2
_SCORE_5 = 1_000_000   # 五连/赢
_SCORE_TABLE = {
    # 四
    (4, 2): 100_000,   # 活四：下一步必胜
    (4, 1): 10_000,    # 冲四：对方必须应
    (4, 0): 0,         # 双端封死的四，无价值
    # 三
    (3, 2): 5_000,     # 活三：威胁极大
    (3, 1): 500,       # 眠三
    (3, 0): 0,
    # 二
    (2, 2): 100,       # 活二
    (2, 1): 10,        # 眠二
    (2, 0): 0,
    # 单子
    (1, 2): 2,
    (1, 1): 1,
    (1, 0): 0,
}


# ────────────────────────────────────────────────
# 核心棋型检测
# ────────────────────────────────────────────────

def _scan_direction(
    arr: np.ndarray, n: int, r: int, c: int, player: int,
    dr: int, dc: int, n_in_row: int
) -> Tuple[int, int]:
    """
    在方向 (dr, dc) 上，以 (r, c)（已有 player 棋子）为中心，
    统计双向连续棋子数和端点开放情况。

    返回 (count, open_ends)：
      count     = 包含 (r,c) 的连续最大长度（上限 n_in_row）
      open_ends = 两端中空格端点数 (0/1/2)
    """
    # 正方向
    cnt_fwd = 0
    fr, fc = r + dr, c + dc
    while 0 <= fr < n and 0 <= fc < n and arr[fr, fc] == player:
        cnt_fwd += 1
        fr += dr
        fc += dc
    fwd_open = int(0 <= fr < n and 0 <= fc < n and arr[fr, fc] == 0)

    # 反方向
    cnt_bwd = 0
    br, bc = r - dr, c - dc
    while 0 <= br < n and 0 <= bc < n and arr[br, bc] == player:
        cnt_bwd += 1
        br -= dr
        bc -= dc
    bwd_open = int(0 <= br < n and 0 <= bc < n and arr[br, bc] == 0)

    total = 1 + cnt_fwd + cnt_bwd
    opens = fwd_open + bwd_open
    # 超长连（长连）同样算赢
    total = min(total, n_in_row)
    return total, opens


def _position_score(
    arr: np.ndarray, n: int, n_in_row: int,
    r: int, c: int, player: int
) -> float:
    """
    假设 player 在 (r, c) 落子后，评估该落子自身形成的威胁得分。
    调用前 arr[r, c] 必须已设为 player（临时修改）。
    """
    score = 0.0
    for dr, dc in DIRECTIONS:
        cnt, opens = _scan_direction(arr, n, r, c, player, dr, dc, n_in_row)
        if cnt >= n_in_row:
            return float(_SCORE_5)
        score += _SCORE_TABLE.get((cnt, opens), 0)
    return score


def get_action_scores(board) -> np.ndarray:
    """
    计算所有位置的综合评分向量（长度 = size²）。
      综合分 = 进攻分 + BLOCK_WEIGHT × 防守分
    非法位置得分为 -inf。

    额外加：
      • 位置中心度奖励（鼓励靠近棋盘中心开局）
      • 邻近度奖励（优先在已有棋子附近落子）
    """
    n = board.size
    n_in_row = board.n_in_row
    arr = board.board      # numpy view，直接修改后恢复
    avail = board.availables
    player = board.current_player
    opp = 3 - player

    scores = np.full(n * n, -np.inf, dtype=np.float64)
    center = (n - 1) / 2.0

    for action in avail:
        r, c = divmod(action, n)

        # ── 进攻得分 ──
        arr[r, c] = player
        atk = _position_score(arr, n, n_in_row, r, c, player)
        arr[r, c] = 0

        if atk >= _SCORE_5:
            # 直接赢，无需继续
            scores[action] = float(_SCORE_5)
            continue

        # ── 防守得分（假设对手落子后的威胁） ──
        arr[r, c] = opp
        dfn = _position_score(arr, n, n_in_row, r, c, opp)
        arr[r, c] = 0

        # ── 位置奖励：中心 + 邻居 ──
        dist = abs(r - center) + abs(c - center)
        center_bonus = max(0.0, n - dist)  # 越靠中心越高

        # 邻近已有棋子的加成（空棋盘时偏向中心）
        neighbor_bonus = 0.0
        for dr2, dc2 in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]:
            nr, nc = r + dr2, c + dc2
            if 0 <= nr < n and 0 <= nc < n and arr[nr, nc] != 0:
                neighbor_bonus += 3.0

        scores[action] = atk + BLOCK_WEIGHT * dfn + center_bonus + neighbor_bonus

    return scores


def scores_to_probs(
    scores: np.ndarray,
    avail: List[int],
    temperature: float = 0.5,
) -> np.ndarray:
    """
    将评分向量转换为归一化软概率分布（长度 = size²）。
    仅在 avail 位置上有正概率，其余位置为 0。

    temperature 越低 → 越贪心；temperature=0.5 在训练时效果较好。
    """
    if not avail:
        return np.zeros(len(scores), dtype=np.float32)

    avail_arr = np.array(avail, dtype=np.int64)
    raw = scores[avail_arr].astype(np.float64)

    # 处理全 -inf（空棋盘极端情况）
    finite_mask = np.isfinite(raw)
    if not finite_mask.any():
        probs_avail = np.ones(len(avail)) / len(avail)
    else:
        raw = np.where(finite_mask, raw, raw[finite_mask].min() - 1e6)
        # 温度 softmax（数值稳定版）
        scaled = raw / max(temperature, 1e-6)
        scaled -= scaled.max()
        exp_s = np.exp(np.clip(scaled, -500, 0))
        total = exp_s.sum()
        probs_avail = exp_s / total if total > 0 else np.ones(len(avail)) / len(avail)

    full = np.zeros(len(scores), dtype=np.float32)
    full[avail_arr] = probs_avail.astype(np.float32)
    return full


# ────────────────────────────────────────────────
# 玩家封装
# ────────────────────────────────────────────────

class RulePlayer:
    """
    基于威胁评估的五子棋规则玩家。

    参数：
      noise_eps   : Dirichlet 噪声混合比例（0 = 无噪声，确定性）
      noise_alpha : Dirichlet 浓度（越小越集中于少数位置）
      score_temp  : 分数 → 概率的 softmax 温度
      deterministic: True = 始终取最高分（用于测试，无随机性）
    """

    def __init__(
        self,
        noise_eps: float = 0.15,
        noise_alpha: float = 0.3,
        score_temp: float = 0.5,
        deterministic: bool = False,
    ):
        self.noise_eps = noise_eps
        self.noise_alpha = noise_alpha
        self.score_temp = score_temp
        self.deterministic = deterministic
        self._player: int = 1

    def set_player_ind(self, p: int) -> None:
        self._player = p

    def reset_player(self) -> None:
        pass

    def get_action(self, board) -> int:
        """返回一个合法落子动作。"""
        avail = board.availables
        if not avail:
            raise RuntimeError("没有合法动作")
        if len(avail) == 1:
            return avail[0]

        scores = get_action_scores(board)

        if self.deterministic:
            return int(np.argmax(scores))

        probs = scores_to_probs(scores, avail, self.score_temp)

        # 混入 Dirichlet 噪声
        if self.noise_eps > 0 and len(avail) > 1:
            noise_vec = np.zeros_like(probs)
            nd = np.random.dirichlet(
                np.ones(len(avail)) * self.noise_alpha
            )
            for i, a in enumerate(avail):
                noise_vec[a] = nd[i]
            probs = (1.0 - self.noise_eps) * probs + self.noise_eps * noise_vec
            s = probs.sum()
            if s > 0:
                probs /= s

        return int(np.random.choice(len(probs), p=probs))

    def get_soft_probs(self, board) -> np.ndarray:
        """
        返回全棋盘软概率向量（长度 size²），不含噪声。
        用于生成训练标签（比 one-hot 信息量更丰富）。
        """
        scores = get_action_scores(board)
        return scores_to_probs(scores, board.availables, self.score_temp)
