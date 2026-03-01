"""
web/game_session.py
单局游戏会话管理。每个 HTTP 会话（用 session_id 标识）持有一个独立的棋盘 + AI 玩家。
"""

import uuid
import time
from typing import Dict, Optional
from gomoku.game import Board
from gomoku.config import config


# 难度 → MCTS 模拟次数
DIFFICULTY_PLAYOUT = {
    "easy":   200,
    "medium": 400,
    "hard":   800,
}


class GameSession:
    """
    一局游戏的完整状态。
    human_player: 1=人类执黑, 2=人类执白
    ai_player:    3 - human_player
    """

    def __init__(self, session_id: str, human_player: int = 1,
                 difficulty: str = "medium"):
        self.session_id  = session_id
        self.human_player = human_player
        self.ai_player    = 3 - human_player
        self.difficulty   = difficulty
        self.n_playout    = DIFFICULTY_PLAYOUT.get(difficulty, 400)
        self.board        = Board(config.BOARD_SIZE, config.N_IN_ROW)
        self.created_at   = time.time()
        self.last_active  = time.time()
        self._mcts_player = None  # 延迟初始化

    def get_mcts_player(self, policy_value_fn):
        """惰性创建 MCTSPlayer（每次请求复用，树复用减少计算量）。"""
        if self._mcts_player is None:
            from gomoku.mcts import MCTSPlayer
            self._mcts_player = MCTSPlayer(
                policy_value_fn,
                c_puct=config.C_PUCT,
                n_playout=self.n_playout,
                is_selfplay=False,
            )
            self._mcts_player.set_player_ind(self.ai_player)
        return self._mcts_player

    def touch(self):
        self.last_active = time.time()

    def to_dict(self) -> dict:
        """序列化给前端的状态。"""
        self.touch()
        return {
            "session_id":   self.session_id,
            "board":        self.board.board.tolist(),
            "board_size":   self.board.size,
            "n_in_row":     self.board.n_in_row,
            "current_player": int(self.board.current_player),
            "human_player": self.human_player,
            "ai_player":    self.ai_player,
            "last_move":    self.board.last_move,
            "move_count":   self.board.move_count,
            "game_over":    self.board.game_over(),
            "winner":       self.board.winner,
            "difficulty":   self.difficulty,
        }


class SessionManager:
    """线程安全的会话管理器（TTL 30 分钟自动清理）。"""

    TTL = 1800  # 秒

    def __init__(self):
        self._sessions: Dict[str, GameSession] = {}

    def create(self, human_player: int = 1, difficulty: str = "medium") -> GameSession:
        sid = str(uuid.uuid4())
        session = GameSession(sid, human_player, difficulty)
        self._sessions[sid] = session
        self._cleanup()
        return session

    def get(self, session_id: str) -> Optional[GameSession]:
        session = self._sessions.get(session_id)
        if session:
            session.touch()
        return session

    def remove(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def _cleanup(self) -> None:
        now  = time.time()
        dead = [sid for sid, s in self._sessions.items()
                if now - s.last_active > self.TTL]
        for sid in dead:
            del self._sessions[sid]
