"""
gomoku/game.py
五子棋棋盘环境（8×8，胜利条件：先在行、列、对角线上形成五连子）

玩家编号：1 = 黑棋（先手），2 = 白棋
动作：int，= row * board_size + col
状态特征（4 通道）：
  ch0 = 当前玩家落子位置
  ch1 = 对手落子位置
  ch2 = 最后一步落子位置（全 1 表示，0 表示空局）
  ch3 = 当前玩家指示符（全 1 = 黑棋，全 0 = 白棋）
"""

import numpy as np
from typing import List, Optional, Tuple
from gomoku.config import config


class Board:
    """棋盘状态，提供强化学习环境接口。"""

    def __init__(self, size: int = config.BOARD_SIZE, n_in_row: int = config.N_IN_ROW):
        self.size    = size
        self.n_in_row = n_in_row
        self.reset()

    # ──────────────────────────────────────────────
    # 基础操作
    # ──────────────────────────────────────────────
    def reset(self) -> None:
        """重置棋盘到初始状态。"""
        self.board: np.ndarray = np.zeros((self.size, self.size), dtype=np.int8)
        self.current_player: int = 1          # 1=黑, 2=白
        self.last_move: Optional[int] = None  # 最近落子动作
        self.move_count: int = 0
        self.winner: Optional[int] = None     # None=对战中, 0=平局, 1/2=胜者

    def copy(self) -> "Board":
        b = Board(self.size, self.n_in_row)
        b.board         = self.board.copy()
        b.current_player = self.current_player
        b.last_move     = self.last_move
        b.move_count    = self.move_count
        b.winner        = self.winner
        return b

    # ──────────────────────────────────────────────
    # 动作空间
    # ──────────────────────────────────────────────
    @property
    def availables(self) -> List[int]:
        """返回所有合法动作列表（扁平化下标）。"""
        return [r * self.size + c
                for r in range(self.size)
                for c in range(self.size)
                if self.board[r, c] == 0]

    def action_to_rc(self, action: int) -> Tuple[int, int]:
        return divmod(action, self.size)

    def rc_to_action(self, r: int, c: int) -> int:
        return r * self.size + c

    # ──────────────────────────────────────────────
    # 落子
    # ──────────────────────────────────────────────
    def do_move(self, action: int) -> None:
        r, c = self.action_to_rc(action)
        assert self.board[r, c] == 0, f"落子位置 ({r},{c}) 已有棋子！"
        self.board[r, c] = self.current_player
        self.last_move   = action
        self.move_count += 1
        # 检测胜负
        if self._check_winner(r, c):
            self.winner = self.current_player
        elif not self.availables:
            self.winner = 0            # 平局
        # 切换玩家
        self.current_player = 3 - self.current_player  # 1↔2

    # ──────────────────────────────────────────────
    # 胜负判定
    # ──────────────────────────────────────────────
    def _check_winner(self, r: int, c: int) -> bool:
        player = self.board[r, c]
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = 1
            for sign in (1, -1):
                nr, nc = r + sign * dr, c + sign * dc
                while 0 <= nr < self.size and 0 <= nc < self.size and self.board[nr, nc] == player:
                    count += 1
                    nr += sign * dr
                    nc += sign * dc
            if count >= self.n_in_row:
                return True
        return False

    def game_over(self) -> bool:
        return self.winner is not None

    # ──────────────────────────────────────────────
    # 神经网络特征输入（4 通道 × size × size）
    # ──────────────────────────────────────────────
    def get_current_state(self) -> np.ndarray:
        """
        输出 float32 数组 shape=(4, size, size)，供神经网络使用。
        """
        state = np.zeros((4, self.size, self.size), dtype=np.float32)
        current  = self.current_player
        opponent = 3 - current
        state[0] = (self.board == current).astype(np.float32)   # 己方棋子
        state[1] = (self.board == opponent).astype(np.float32)  # 对手棋子
        if self.last_move is not None:
            lr, lc = self.action_to_rc(self.last_move)
            state[2, lr, lc] = 1.0                              # 最近落子
        if current == 1:
            state[3] = 1.0                                      # 当前玩家为黑棋
        return state

    # ──────────────────────────────────────────────
    # 对称扩充（8 种变换：4 旋转 × 翻转）
    # ──────────────────────────────────────────────
    @staticmethod
    def augment_data(
        state: np.ndarray, mcts_prob: np.ndarray, winner: float
    ) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """
        对 (state, mcts_prob, winner) 进行 8 种几何变换，返回扩充后的列表。
        state:      (4, N, N)
        mcts_prob:  (N*N,)
        """
        n = state.shape[1]
        prob_2d = mcts_prob.reshape(n, n)
        samples = []
        for flip in range(2):
            s    = np.flip(state,   axis=2) if flip else state.copy()
            p_2d = np.flip(prob_2d, axis=1) if flip else prob_2d.copy()
            for rot in range(4):
                s_r    = np.rot90(s,    rot, axes=(1, 2))
                p_r    = np.rot90(p_2d, rot)
                samples.append((s_r.copy(), p_r.flatten().copy(), winner))
        return samples

    # ──────────────────────────────────────────────
    # 调试输出
    # ──────────────────────────────────────────────
    def __str__(self) -> str:
        symbols = {0: ".", 1: "●", 2: "○"}
        header  = "   " + " ".join(str(c) for c in range(self.size))
        rows    = [header]
        for r in range(self.size):
            row = f"{r:2d} " + " ".join(symbols[self.board[r, c]] for c in range(self.size))
            rows.append(row)
        return "\n".join(rows)


class Game:
    """编排两个玩家之间的完整对局。"""

    def __init__(self, board: Optional[Board] = None):
        self.board = board if board else Board()

    def start_play(self, player1, player2, start_player: int = 1, verbose: bool = False):
        """
        player1 / player2 需实现 get_action(board) -> int 接口。
        start_player: 1=黑先, 2=白先
        返回: winner（0=平局, 1=黑赢, 2=白赢）
        """
        self.board.reset()
        if start_player == 2:
            self.board.current_player = 2
        players = {1: player1, 2: player2}
        player1.set_player_ind(1)
        player2.set_player_ind(2)
        if verbose:
            print(self.board)
        while not self.board.game_over():
            p = players[self.board.current_player]
            move = p.get_action(self.board)
            self.board.do_move(move)
            if verbose:
                print(f"Player {3 - self.board.current_player} → {self.board.action_to_rc(move)}")
                print(self.board)
        winner = self.board.winner
        if verbose:
            print("Winner:", winner if winner else "Draw")
        return winner

    def start_self_play(self, player, temp: float = 1e-3):
        """
        自我博弈，使用 MCTS 带温度采样。
        返回: (winner, play_data)
          play_data = list of (state, mcts_prob, current_player)
        """
        self.board.reset()
        player.reset_player()
        states, mcts_probs, current_players = [], [], []

        while not self.board.game_over():
            # 前 TEMP_THRESHOLD 步高温探索
            t = 1.0 if self.board.move_count < config.TEMP_THRESHOLD else temp
            move, move_probs = player.get_action(self.board, temp=t, return_prob=True)
            states.append(self.board.get_current_state())
            mcts_probs.append(move_probs)
            current_players.append(self.board.current_player)
            self.board.do_move(move)

        winner = self.board.winner
        # 计算每步的 z 值（从当前玩家视角）
        winners_z = np.zeros(len(current_players), dtype=np.float32)
        if winner != 0:
            winners_z[np.array(current_players) == winner] =  1.0
            winners_z[np.array(current_players) != winner] = -1.0

        # 对称扩充 → 8× 数据量
        play_data = []
        for s, p, z in zip(states, mcts_probs, winners_z):
            play_data.extend(Board.augment_data(s, p, float(z)))
        return winner, play_data
