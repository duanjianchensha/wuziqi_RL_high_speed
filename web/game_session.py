"""
web/game_session.py
单局游戏会话管理。每个 HTTP 会话（用 session_id 标识）持有一个独立的棋盘 + AI 玩家。
"""

import uuid
import time
import os
import json
from typing import Dict, Optional
from gomoku.game import Board
from gomoku.config import config


def _round_to_10(v: float) -> int:
    return int(round(v / 10.0) * 10)


def get_difficulty_playout() -> dict:
    """
    根据训练配置和Web难度配置生成各难度对应的MCTS模拟次数。
    支持倍数或绝对值配置。
    """
    total_games = max(1, int(config.N_SELFPLAY_GAMES))
    train_playout = int(config.N_PLAYOUT_TRAIN)

    # 若存在最近训练配置，则优先使用（支持 train.py 命令行覆盖后的自动同步）
    profile_path = os.path.join(config.MODEL_DIR, "latest_train_profile.json")
    if os.path.exists(profile_path):
        try:
            with open(profile_path, "r", encoding="utf-8") as f:
                profile = json.load(f)
            total_games = max(1, int(profile.get("n_selfplay_games", total_games)))
            train_playout = int(profile.get("n_playout_train", train_playout))
        except Exception:
            pass

    # 以 500 局为基准，限制缩放范围，避免极端值
    scale = max(0.6, min(1.6, total_games / 500.0))
    base = int(train_playout * scale)

    result = {}
    for difficulty in ["easy", "medium", "hard"]:
        # 优先检查绝对值配置
        absolute_playout = config.WEB_DIFFICULTY_PLAYOUTS.get(difficulty)
        if absolute_playout is not None:
            result[difficulty] = max(10, int(absolute_playout))  # 最小10次模拟
        else:
            # 使用倍数配置
            multiplier = config.WEB_DIFFICULTY_MULTIPLIERS.get(difficulty, 1.0)
            playout = max(80, _round_to_10(base * multiplier))
            result[difficulty] = playout

    # 确保难度递增：easy <= medium <= hard
    easy = result["easy"]
    medium = max(easy, result["medium"])
    hard = max(medium, result["hard"])

    return {
        "easy": easy,
        "medium": medium,
        "hard": hard,
    }


class GameSession:
    """
    一局游戏的完整状态。
    human_player: 1=人类执黑, 2=人类执白
    ai_player:    3 - human_player
    """

    def __init__(
        self, session_id: str, human_player: int = 1, difficulty: str = "medium"
    ):
        difficulty_playout = get_difficulty_playout()
        self.session_id = session_id
        self.human_player = human_player
        self.ai_player = 3 - human_player
        self.difficulty = difficulty
        self.n_playout = difficulty_playout.get(
            difficulty, difficulty_playout["medium"]
        )
        self.board = Board(config.BOARD_SIZE, config.N_IN_ROW)
        self.created_at = time.time()
        self.last_active = time.time()
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
            "session_id": self.session_id,
            "board": self.board.board.tolist(),
            "board_size": self.board.size,
            "n_in_row": self.board.n_in_row,
            "current_player": int(self.board.current_player),
            "human_player": self.human_player,
            "ai_player": self.ai_player,
            "last_move": self.board.last_move,
            "move_count": self.board.move_count,
            "game_over": self.board.game_over(),
            "winner": self.board.winner,
            "difficulty": self.difficulty,
            "n_playout": self.n_playout,
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
        now = time.time()
        dead = [
            sid for sid, s in self._sessions.items() if now - s.last_active > self.TTL
        ]
        for sid in dead:
            del self._sessions[sid]
