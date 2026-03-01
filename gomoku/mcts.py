"""
gomoku/mcts.py
Phase 1: 纯 Python MCTS 实现（AlphaZero PUCT 公式）

阶段说明：
  Phase 1 - 纯 Python，单线程 MCTS（当前文件）
  Phase 2 - C++ 并行 MCTS（见 src/ 目录，TODO）

PUCT 选择公式：
  U(s,a) = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
  选择：a* = argmax U(s,a)
"""

import math
import numpy as np
from typing import Optional, Dict, List, Tuple, Callable

from gomoku.config import config


# ────────────────────────────────────────────────
# MCTSNode
# ────────────────────────────────────────────────
class MCTSNode:
    """MCTS 树节点。"""

    __slots__ = ("parent", "children", "n_visits", "_W", "_Q", "_P")

    def __init__(self, parent: Optional["MCTSNode"], prior_p: float):
        self.parent:   Optional["MCTSNode"] = parent
        self.children: Dict[int, "MCTSNode"] = {}
        self.n_visits: int   = 0
        self._W:       float = 0.0   # 累计价值
        self._Q:       float = 0.0   # 均值价值
        self._P:       float = prior_p

    # ── PUCT 值 ────────────────────────────────
    def get_value(self, c_puct: float) -> float:
        parent_n = self.parent.n_visits if self.parent else 1
        u = c_puct * self._P * math.sqrt(parent_n) / (1 + self.n_visits)
        return self._Q + u

    # ── 扩展 ───────────────────────────────────
    def expand(self, action_priors: List[Tuple[int, float]]) -> None:
        for action, prob in action_priors:
            if action not in self.children:
                self.children[action] = MCTSNode(self, prob)

    # ── 选择 ───────────────────────────────────
    def select(self, c_puct: float) -> Tuple[int, "MCTSNode"]:
        return max(self.children.items(), key=lambda kv: kv[1].get_value(c_puct))

    # ── 反向传播 ────────────────────────────────
    def update(self, leaf_value: float) -> None:
        self.n_visits += 1
        self._W += leaf_value
        self._Q  = self._W / self.n_visits

    def update_recursive(self, leaf_value: float) -> None:
        if self.parent:
            self.parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def is_root(self) -> bool:
        return self.parent is None


# ────────────────────────────────────────────────
# MCTS 搜索
# ────────────────────────────────────────────────
class MCTS:
    """
    Monte Carlo Tree Search，使用 policy_value_fn 进行叶节点扩展与价值估计。
    """

    def __init__(self, policy_value_fn: Callable, c_puct: float = config.C_PUCT,
                 n_playout: int = config.N_PLAYOUT_TRAIN):
        self._policy_value_fn = policy_value_fn
        self._c_puct  = c_puct
        self._n_playout = n_playout
        self._root    = MCTSNode(None, 1.0)

    def _playout(self, board) -> None:
        """
        从根节点执行一次模拟（选择 → 扩展 → 估值 → 反向传播）。
        board 为临时副本，原 board 不变。
        """
        node = self._root
        while True:
            if node.is_leaf():
                break
            action, node = node.select(self._c_puct)
            board.do_move(action)

        # 调用神经网络：得到 (action_probs, value)
        action_priors, leaf_value = self._policy_value_fn(board)

        if board.game_over():
            w = board.winner
            if w == 0:
                leaf_value = 0.0
            else:
                # 注意：board.current_player 已切换，需反转
                leaf_value = 1.0 if w != board.current_player else -1.0
        else:
            node.expand(action_priors)

        node.update_recursive(-leaf_value)

    def get_move_probs(self, board, temp: float = 1e-3) -> Tuple[np.ndarray, np.ndarray]:
        """
        运行 n_playout 次模拟，返回根节点处各动作的概率分布。
        temp: 温度，高温（=1）探索，低温（→0）利用
        返回: (actions, probs) 均为 numpy 数组
        """
        for _ in range(self._n_playout):
            board_copy = board.copy()
            self._playout(board_copy)

        children = self._root.children
        acts   = np.array(list(children.keys()), dtype=np.int32)
        visits = np.array([n.n_visits for n in children.values()], dtype=np.float32)

        if temp < 0.01:
            # 低温 → 贪心，直接取访问次数最高的动作
            best = np.argmax(visits)
            probs = np.zeros_like(visits)
            probs[best] = 1.0
        else:
            # 数值稳定的温度采样：log-space 避免大幂次溢出
            log_v = np.log(visits + 1e-10) / temp
            log_v -= log_v.max()             # softmax 稳定化
            probs  = np.exp(log_v)
            probs  = probs / probs.sum()
        probs = probs / probs.sum()   # 二次归一化，消除浮点误差
        return acts, probs

    def update_with_move(self, last_move: int) -> None:
        """
        将树根移至 last_move 对应子节点，实现树复用（减少重复计算）。
        last_move == -1 时完全重置树。
        """
        if last_move != -1 and last_move in self._root.children:
            new_root = self._root.children[last_move]
            new_root.parent = None
            self._root = new_root
        else:
            self._root = MCTSNode(None, 1.0)


# ────────────────────────────────────────────────
# MCTSPlayer：封装 MCTS 供游戏/评估调用
# ────────────────────────────────────────────────
class MCTSPlayer:
    """
    AlphaZero 风格 MCTS 玩家，支持：
      • 自弈模式（is_selfplay=True）：加 Dirichlet 噪声，返回动作概率
      • 对弈模式（is_selfplay=False）：高温或低温选择
    """

    def __init__(self, policy_value_fn: Callable,
                 c_puct: int    = config.C_PUCT,
                 n_playout: int = config.N_PLAYOUT_TRAIN,
                 is_selfplay: bool = False):
        self.policy_value_fn = policy_value_fn
        self.c_puct     = c_puct
        self.n_playout  = n_playout
        self.is_selfplay = is_selfplay
        self.player: Optional[int] = None
        self.mcts = MCTS(policy_value_fn, c_puct, n_playout)

    def set_player_ind(self, p: int) -> None:
        self.player = p

    def reset_player(self) -> None:
        self.mcts.update_with_move(-1)

    def get_action(self, board, temp: float = 1e-3,
                   return_prob: bool = False):
        """
        选择落子动作。
        return_prob=True 时返回 (action, full_prob_vector)，
        否则只返回 action。
        """
        avail = board.availables
        n_sq  = board.size * board.size
        move_probs = np.zeros(n_sq, dtype=np.float32)

        if not avail:
            raise RuntimeError("没有可用动作！")

        acts, probs = self.mcts.get_move_probs(board, temp)
        move_probs[acts] = probs

        if self.is_selfplay:
            # 自弈：混入 Dirichlet 噪声增加探索
            noise = np.random.dirichlet(
                config.DIRICHLET_ALPHA * np.ones(len(probs)))
            mixed = (1 - config.DIRICHLET_EPS) * probs + config.DIRICHLET_EPS * noise
            mixed = mixed / mixed.sum()          # 归一化，消除浮点误差
            move = int(np.random.choice(acts, p=mixed))
            self.mcts.update_with_move(move)          # 复用树
        else:
            move = int(np.random.choice(acts, p=probs))
            self.mcts.update_with_move(-1)             # 对弈时每步重置

        if return_prob:
            return move, move_probs
        return move

    def __repr__(self) -> str:
        return f"MCTSPlayer(player={self.player}, n_playout={self.n_playout})"


# ────────────────────────────────────────────────
# 纯 MCTS（无神经网络，用于评估基准）
# ────────────────────────────────────────────────
def _rollout_policy(board) -> Tuple[List[Tuple[int, float]], float]:
    """随机 rollout 到终局，返回均匀策略 + rollout 价值。"""
    avail = board.availables
    priors = [(a, 1.0 / len(avail)) for a in avail] if avail else []

    # rollout
    b = board.copy()
    while not b.game_over():
        move = np.random.choice(b.availables)
        b.do_move(move)

    w = b.winner
    if w == 0:
        value = 0.0
    else:
        value = 1.0 if w == board.current_player else -1.0
    return priors, value


class PureMCTSPlayer:
    """纯 MCTS 玩家（无神经网络），用于评估新模型是否有提升。"""

    def __init__(self, n_playout: int = config.N_PLAYOUT_EVAL):
        self.n_playout = n_playout
        self.player: Optional[int] = None
        self.mcts  = MCTS(_rollout_policy, c_puct=5.0, n_playout=n_playout)

    def set_player_ind(self, p: int) -> None:
        self.player = p

    def reset_player(self) -> None:
        self.mcts.update_with_move(-1)

    def get_action(self, board, **kwargs) -> int:
        avail = board.availables
        if not avail:
            raise RuntimeError("没有可用动作！")
        acts, probs = self.mcts.get_move_probs(board, temp=1e-3)
        move = int(np.random.choice(acts, p=probs))
        self.mcts.update_with_move(-1)
        return move

    def __repr__(self) -> str:
        return f"PureMCTSPlayer(player={self.player}, n_playout={self.n_playout})"


# ────────────────────────────────────────────────
# Phase 2 C++ 加速开关（编译 C++ 模块后自动启用）
# ────────────────────────────────────────────────
def _try_import_cpp():
    """
    尝试导入 Phase 2 C++ 加速模块 gomoku_cpp。
    若未编译则静默返回 None。
    运行 scripts/build.bat 编译后此函数即自动开始返回 C++ 模块。
    """
    try:
        import gomoku_cpp
        return gomoku_cpp
    except ImportError:
        return None


_CPP_MOD = _try_import_cpp()


def make_mcts_player(policy_value_fn: Callable,
                     c_puct: float = config.C_PUCT,
                     n_playout: int = config.N_PLAYOUT_TRAIN,
                     is_selfplay: bool = False) -> MCTSPlayer:
    """
    工厂函数：
      • 若 C++ 加速模块已编译 (gomoku_cpp.so/.pyd)，则返回 C++ 版 MCTSPlayer 包装。
      • 否则返回纯 Python MCTSPlayer（Phase 1 默认）。
    """
    if _CPP_MOD is not None and not is_selfplay:
        # C++ 版包装器（对弈时使用，自弈仍用 Python 以支持 Dirichlet 噪声）
        return _CppPlayerWrapper(_CPP_MOD, policy_value_fn, c_puct, n_playout)
    return MCTSPlayer(policy_value_fn, c_puct, n_playout, is_selfplay)


class _CppPlayerWrapper:
    """将 gomoku_cpp.MCTSPlayer 适配为与 MCTSPlayer 相同的接口。"""

    def __init__(self, cpp_mod, policy_value_fn, c_puct, n_playout):
        import numpy as np

        # C++ 推理函数适配：输入 (4,N,N) np.ndarray → (probs(N*N,), value)
        def infer(state_arr: np.ndarray):
            return policy_value_fn.__self__._infer_numpy(state_arr)  # type: ignore

        self._cpp = cpp_mod.MCTSPlayer(policy_value_fn, c_puct, n_playout)
        self.player: Optional[int] = None

    def set_player_ind(self, p: int) -> None:
        self.player = p

    def reset_player(self) -> None:
        self._cpp.reset()

    def get_action(self, board, temp: float = 1e-3, return_prob: bool = False):
        import numpy as np
        # C++ board 需单独构建；此处仍使用 Python Board，传入状态给 C++ MCTS
        # （C++ MCTSPlayer 接受 gomoku_cpp.Board，简化实现：fallback to Python）
        # 完整 C++ 集成需要统一棋盘表示，留作 Phase 2 完整集成
        raise NotImplementedError("C++ bridge: 请在 coach.py 中直接调用 gomoku_cpp")

    def __repr__(self) -> str:
        return f"CppMCTSPlayer(player={self.player})"
