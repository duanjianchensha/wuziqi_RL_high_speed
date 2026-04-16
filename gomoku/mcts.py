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
        self.parent: Optional["MCTSNode"] = parent
        self.children: Dict[int, "MCTSNode"] = {}
        self.n_visits: int = 0
        self._W: float = 0.0  # 累计价值
        self._Q: float = 0.0  # 均值价值
        self._P: float = prior_p

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
        self._Q = self._W / self.n_visits

    def update_recursive(self, leaf_value: float) -> None:
        if self.parent:
            self.parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    # ── Virtual Loss（批量 MCTS 用）──────────────
    def apply_vl(self, weight: float = 1.0) -> None:
        """沿路径施加虚拟损失，使后续 playout 不倾向于重复选此节点。"""
        self.n_visits += 1
        self._W -= weight
        self._Q = self._W / self.n_visits

    def undo_vl(self, weight: float = 1.0) -> None:
        """在真实反向传播完成后移除虚拟损失。"""
        self.n_visits -= 1
        self._W += weight
        self._Q = self._W / self.n_visits if self.n_visits > 0 else 0.0

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

    支持两种推理模式：
      • 单步模式（默认）：每次 playout 单独调用 policy_value_fn（batch=1）
      • 批量模式：提供 batch_infer_fn 后，每 LEAF_BATCH_SIZE 个叶节点
        打包一次前向传播，显著减少 GPU/CPU 推理调用次数
    """

    def __init__(
        self,
        policy_value_fn: Callable,
        c_puct: float = config.C_PUCT,
        n_playout: int = config.N_PLAYOUT_TRAIN,
        batch_infer_fn=None,
    ):
        self._policy_value_fn = policy_value_fn
        # batch_infer_fn(states: np.ndarray(B,4,N,N)) -> (probs(B,N²), values(B,))
        self._batch_infer_fn = batch_infer_fn
        self._c_puct = c_puct
        self._n_playout = n_playout
        self._root = MCTSNode(None, 1.0)

    # ── 选择到叶节点 ────────────────────────────────
    def _select_to_leaf(self, board):
        """从根节点向下选择直到叶节点，返回 (leaf_node, board_at_leaf)。"""
        node = self._root
        while not node.is_leaf():
            if board.game_over():
                break
            action, node = node.select(self._c_puct)
            board.do_move(action)
        return node, board

    def _select_to_leaf_vl(self, board):
        """
        带 Virtual Loss 的选择：沿路径每个节点立即施加 VL，使同 batch 内后续
        playout 选择其他路径（提高叶节点多样性）。
        返回 (leaf_node, board_at_leaf, path_nodes)。
        """
        node = self._root
        path = [node]
        node.apply_vl()
        while not node.is_leaf():
            if board.game_over():
                break
            action, node = node.select(self._c_puct)
            board.do_move(action)
            node.apply_vl()
            path.append(node)
        return node, board, path

    # ── 根节点价值（认输检测用）───────────────────
    def get_root_value(self) -> float:
        """返回根节点均值价值（当前玩家视角）。须在 get_move_probs() 之后调用。"""
        return self._root._Q

    def _playout(self, board) -> None:
        """
        从根节点执行一次模拟（选择 → 扩展 → 估值 → 反向传播）。
        board 为临时副本，原 board 不变。
        """
        node, board = self._select_to_leaf(board)

        # 调用神经网络：得到 (action_probs, value)
        action_priors, leaf_value = self._policy_value_fn(board)

        if board.game_over():
            w = board.winner
            if w == 0:
                leaf_value = 0.0
            else:
                # board.current_player 已切换到下一位玩家：
                # 若 winner == current_player，说明"当前要走的人"赢了 → +1
                # 若 winner != current_player，说明"当前要走的人"输了 → -1
                leaf_value = 1.0 if w == board.current_player else -1.0
        else:
            node.expand(action_priors)

        node.update_recursive(-leaf_value)

    def get_move_probs(
        self, board, temp: float = 1e-3, add_noise: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        运行 n_playout 次模拟，返回根节点处各动作的概率分布。
        temp: 温度，高温（=1）探索，低温（→0）利用
        返回: (actions, probs) 均为 numpy 数组
        """
        # --- 强制展开根节点 ---。避免批量推理时第一批 16 个模拟全部撞在未展开的根节点导致浪费。
        if self._root.is_leaf() and not board.game_over():
            if self._batch_infer_fn is not None:
                state = board.get_current_state()
                states = np.stack([state], axis=0)  # (1, 4, N, N)
                all_probs, all_values = self._batch_infer_fn(states)
                act_probs = all_probs[0][board.availables]
                if act_probs.sum() > 0:
                    act_probs /= act_probs.sum()
                action_priors = list(zip(board.availables, act_probs))
                leaf_value = float(all_values[0])
            else:
                action_priors, leaf_value = self._policy_value_fn(board)
            self._root.expand(action_priors)
            self._root.update_recursive(-leaf_value)

        # --- AlphaZero 标准做法：在根节点 P 上注入 Dirichlet 噪声，以引导 MCTS 探索未知分支 ---
        if add_noise and not self._root.is_leaf():
            actions = list(self._root.children.keys())
            noise = np.random.dirichlet(config.DIRICHLET_ALPHA * np.ones(len(actions)))
            for i, act in enumerate(actions):
                c = self._root.children[act]
                c._P = (1 - config.DIRICHLET_EPS) * c._P + config.DIRICHLET_EPS * noise[
                    i
                ]

        if self._batch_infer_fn is not None:
            self._run_batched_playouts(board)
        else:
            for _ in range(self._n_playout):
                board_copy = board.copy()
                self._playout(board_copy)

        children = self._root.children
        acts = np.array(list(children.keys()), dtype=np.int32)
        visits = np.array([n.n_visits for n in children.values()], dtype=np.float32)

        if temp < 0.01:
            # 低温 → 贪心，直接取访问次数最高的动作
            best = np.argmax(visits)
            probs = np.zeros_like(visits)
            probs[best] = 1.0
        else:
            # 数值稳定的温度采样：log-space 避免大幂次溢出
            log_v = np.log(visits + 1e-10) / temp
            log_v -= log_v.max()  # softmax 稳定化
            probs = np.exp(log_v)
            probs = probs / probs.sum()
        probs = probs / probs.sum()  # 二次归一化，消除浮点误差
        return acts, probs

    def _run_batched_playouts(self, board) -> None:
        """
        批量叶节点推理（含 Virtual Loss）。

        每收集 LEAF_BATCH_SIZE 个叶节点打包一次 NN 前向；同时在选择阶段
        对路径施加 Virtual Loss，使同 batch 内后续 playout 倾向于探索
        不同路径，从而提高叶节点多样性（避免 batch 内全部选同一叶节点）。
        VL 在真实反向传播前撤销，不影响树的最终统计。
        """
        batch_size = config.LEAF_BATCH_SIZE
        sims_done = 0
        while sims_done < self._n_playout:
            this_batch = min(batch_size, self._n_playout - sims_done)
            sims_done += this_batch

            # Phase 1：带 Virtual Loss 的选择阶段
            # nn_pending / term_pending 元素格式：(node, board_copy, path)
            nn_pending = []
            term_pending = []

            for _ in range(this_batch):
                bc = board.copy()
                node, bc, path = self._select_to_leaf_vl(bc)
                if bc.game_over():
                    term_pending.append((node, bc, path))
                else:
                    nn_pending.append((node, bc, path))

            # Phase 2：批量 NN 推理（非终局叶节点）
            if nn_pending:
                states = np.stack(
                    [bc.get_current_state() for _, bc, _ in nn_pending], axis=0
                )  # (B, 4, N, N)
                all_probs, all_values = self._batch_infer_fn(states)
                for (node, bc, path), probs, value in zip(
                    nn_pending, all_probs, all_values
                ):
                    # 撤销 Virtual Loss 后再做真实反向传播
                    for n in path:
                        n.undo_vl()
                    if node.is_leaf():
                        avail = bc.availables
                        act_probs = probs[avail]
                        if act_probs.sum() > 0:
                            act_probs /= act_probs.sum()
                        node.expand(list(zip(avail, act_probs)))
                    node.update_recursive(-float(value))

            # Phase 3：终局反向传播
            for node, bc, path in term_pending:
                for n in path:
                    n.undo_vl()
                w = bc.winner
                lv = 0.0 if w == 0 else (1.0 if w == bc.current_player else -1.0)
                node.update_recursive(-lv)

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

    def __init__(
        self,
        policy_value_fn: Callable,
        c_puct: int = config.C_PUCT,
        n_playout: int = config.N_PLAYOUT_TRAIN,
        is_selfplay: bool = False,
        batch_infer_fn=None,
    ):
        self.policy_value_fn = policy_value_fn
        self.c_puct = c_puct
        self.n_playout = n_playout
        self.is_selfplay = is_selfplay
        self.player: Optional[int] = None
        self.mcts = MCTS(
            policy_value_fn, c_puct, n_playout, batch_infer_fn=batch_infer_fn
        )

    def set_player_ind(self, p: int) -> None:
        self.player = p

    def reset_player(self) -> None:
        self.mcts.update_with_move(-1)

    def get_root_value(self) -> float:
        """返回 MCTS 根节点均值价值（当前玩家视角）。用于认输检测。"""
        return self.mcts.get_root_value()

    def get_action(self, board, temp: float = 1e-3, return_prob: bool = False):
        """
        选择落子动作。
        return_prob=True 时返回 (action, full_prob_vector)，
        否则只返回 action。
        """
        avail = board.availables
        n_sq = board.size * board.size
        move_probs = np.zeros(n_sq, dtype=np.float32)

        if not avail:
            raise RuntimeError("没有可用动作！")

        acts, probs = self.mcts.get_move_probs(board, temp, add_noise=self.is_selfplay)
        move_probs[acts] = probs

        if self.is_selfplay:
            # 噪声已经加到了 MCTS 的根节点先验概率中，MCTS 的搜索结果已经受到了噪声的正确引导
            # 这里只需根据 MCTS 搜索得出的分布正常采样即可
            move = int(np.random.choice(acts, p=probs))
            self.mcts.update_with_move(move)  # 复用树
        else:
            move = int(np.random.choice(acts, p=probs))
            self.mcts.update_with_move(-1)  # 对弈时每步重置

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
        self.mcts = MCTS(_rollout_policy, c_puct=5.0, n_playout=n_playout)

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


def make_mcts_player(
    policy_value_fn: Callable,
    c_puct: float = config.C_PUCT,
    n_playout: int = config.N_PLAYOUT_TRAIN,
    is_selfplay: bool = False,
) -> MCTSPlayer:
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
