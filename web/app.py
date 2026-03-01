"""
web/app.py
FastAPI 后端 — 五子棋人机对战 API

接口:
  POST /api/new_game        创建新游戏
  GET  /api/state/{sid}     获取当前棋盘状态
  POST /api/human_move/{sid} 人类落子
  POST /api/ai_move/{sid}   AI 走子
  POST /api/resign/{sid}    认输
"""

import os
import sys

# 将项目根目录加入 sys.path（从 web/ 目录运行时）
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional

from web.game_session import SessionManager
from gomoku.config import config
from gomoku.neural_net import PolicyValueFunction

# ──────────────────────────────────────────────
# 全局单例：模型 + 会话管理器
# ──────────────────────────────────────────────
app = FastAPI(title="Gomoku AlphaZero API", version="1.0")

# 挂载静态文件
_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

# 全局会话管理
session_manager = SessionManager()

# 模型（启动时加载一次）
_model: Optional[PolicyValueFunction] = None


def get_model() -> PolicyValueFunction:
    global _model
    if _model is None:
        best = config.BEST_POLICY
        _model = PolicyValueFunction(
            board_size=config.BOARD_SIZE,
            model_path=best if os.path.exists(best) else None,
        )
        if not os.path.exists(best):
            print("[Web] 警告：未找到 best_policy.pth，使用随机初始化网络（先训练再使用）")
    return _model


# ──────────────────────────────────────────────
# Pydantic 请求模型
# ──────────────────────────────────────────────
class NewGameRequest(BaseModel):
    human_player: int   = Field(1, description="人类棋子颜色: 1=黑子(先手), 2=白子")
    difficulty:   str   = Field("medium", description="难度: easy / medium / hard")


class MoveRequest(BaseModel):
    action: int = Field(..., description="落子动作: row*board_size + col")


# ──────────────────────────────────────────────
# 路由
# ──────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def index():
    return FileResponse(os.path.join(_STATIC_DIR, "index.html"))


@app.post("/api/new_game")
async def new_game(req: NewGameRequest):
    """创建新游戏，返回初始状态。"""
    if req.human_player not in (1, 2):
        raise HTTPException(400, "human_player 必须为 1 或 2")
    if req.difficulty not in ("easy", "medium", "hard"):
        raise HTTPException(400, "difficulty 必须为 easy/medium/hard")
    session = session_manager.create(req.human_player, req.difficulty)
    return {"ok": True, **session.to_dict()}


@app.get("/api/state/{session_id}")
async def get_state(session_id: str):
    """获取当前棋盘状态。"""
    s = session_manager.get(session_id)
    if not s:
        raise HTTPException(404, "会话不存在（可能已超时）")
    return {"ok": True, **s.to_dict()}


@app.post("/api/human_move/{session_id}")
async def human_move(session_id: str, req: MoveRequest):
    """
    人类落子。
    返回更新后的棋盘状态，若游戏结束则 game_over=True。
    """
    s = session_manager.get(session_id)
    if not s:
        raise HTTPException(404, "会话不存在")
    if s.board.game_over():
        raise HTTPException(400, "游戏已结束")
    if s.board.current_player != s.human_player:
        raise HTTPException(400, "现在不是人类玩家的回合")
    if req.action not in s.board.availables:
        raise HTTPException(400, f"非法落子: {req.action}")

    s.board.do_move(req.action)
    # 落子后通知 MCTS 树更新（树复用）
    model = get_model()
    player = s.get_mcts_player(model.policy_value_fn)
    player.mcts.update_with_move(req.action)

    return {"ok": True, **s.to_dict()}


@app.post("/api/ai_move/{session_id}")
async def ai_move(session_id: str):
    """
    AI 落子（MCTS 计算）。
    返回更新后的棋盘状态及 AI 选择的 action。
    """
    s = session_manager.get(session_id)
    if not s:
        raise HTTPException(404, "会话不存在")
    if s.board.game_over():
        raise HTTPException(400, "游戏已结束")
    if s.board.current_player != s.ai_player:
        raise HTTPException(400, "现在不是 AI 的回合")

    model  = get_model()
    player = s.get_mcts_player(model.policy_value_fn)
    # temp=1e-3 → 近似贪心（对弈时不需要探索）
    action = player.get_action(s.board, temp=1e-3)
    s.board.do_move(action)

    resp = s.to_dict()
    resp["ai_action"] = int(action)
    resp["ok"] = True
    return resp


@app.post("/api/resign/{session_id}")
async def resign(session_id: str):
    """人类认输，标记游戏结束。"""
    s = session_manager.get(session_id)
    if not s:
        raise HTTPException(404, "会话不存在")
    # 强制设置胜者为 AI
    s.board.winner = s.ai_player
    return {"ok": True, **s.to_dict()}


@app.get("/api/hint/{session_id}")
async def get_hint(session_id: str):
    """
    返回 AI 推荐落子位置的概率热图（前端可显示提示）。
    """
    s = session_manager.get(session_id)
    if not s:
        raise HTTPException(404, "会话不存在")
    if s.board.game_over():
        raise HTTPException(400, "游戏已结束")

    import numpy as np
    model = get_model()
    action_probs, value = model.policy_value_fn(s.board)
    n = s.board.size
    heatmap = np.zeros(n * n, dtype=np.float32)
    for a, p in action_probs:
        heatmap[a] = p
    return {
        "ok":      True,
        "heatmap": heatmap.tolist(),
        "value":   round(float(value), 4),
    }
