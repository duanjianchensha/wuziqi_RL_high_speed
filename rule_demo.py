"""
rule_demo.py — 规则玩家对战 Demo（零额外依赖，内置 HTTP 服务）

用法：
  python rule_demo.py            # 默认 http://127.0.0.1:7070
  python rule_demo.py --port 8080

功能：
  • 浏览器内和规则策略对弈
  • 显示规则程序的分数热图
  • 对局结束后保存落子记录到 rule_demo_records/record_<时间>.json
"""

import argparse
import json
import os
import sys
import time
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gomoku.config import config
from gomoku.game import Board
from gomoku.rule_player import get_action_scores, scores_to_probs

BOARD_SIZE = config.BOARD_SIZE
N_IN_ROW = config.N_IN_ROW
RECORD_DIR = os.path.join(os.path.dirname(__file__), "rule_demo_records")
os.makedirs(RECORD_DIR, exist_ok=True)

# ── 全局对局状态（单用户 demo，线程锁保护）──────────
_lock = threading.Lock()
_state: dict = {}


def _new_game(human_player: int = 1) -> dict:
    board = Board(BOARD_SIZE, N_IN_ROW)
    return {
        "board": board,
        "human_player": human_player,   # 1=黑(先手) 2=白(后手)
        "moves": [],                     # [(player, row, col), ...]
        "over": False,
        "winner": None,
        "start_ts": time.time(),
    }


def _board_to_list(board: Board) -> list:
    return board.board.tolist()


def _get_scores_map(board: Board) -> list:
    """返回 size×size 的分数矩阵（已规一化到 [0,1]），非法位 -1。"""
    scores = get_action_scores(board)
    n = BOARD_SIZE
    result = [[-1.0] * n for _ in range(n)]
    avail_set = set(board.availables)
    finite = [scores[a] for a in avail_set if scores[a] > -1e9]
    if not finite:
        return result
    mn, mx = min(finite), max(finite)
    span = mx - mn if mx > mn else 1.0
    for a in avail_set:
        r, c = divmod(a, n)
        v = scores[a]
        result[r][c] = float((v - mn) / span) if v > -1e9 else 0.0
    return result


def _rule_move(board: Board) -> int:
    """规则程序选择落子（确定性，取最高分）。"""
    scores = get_action_scores(board)
    avail = board.availables
    best = max(avail, key=lambda a: scores[a])
    return best


def _save_record(state: dict) -> str:
    record = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "human_player": state["human_player"],
        "winner": state["winner"],
        "winner_label": (
            "平局" if state["winner"] == 0
            else ("人类" if state["winner"] == state["human_player"] else "规则AI")
            if state["winner"] is not None else "未结束"
        ),
        "total_moves": len(state["moves"]),
        "duration_sec": round(time.time() - state["start_ts"], 1),
        "moves": [
            {"move": i + 1, "player": pl, "row": r, "col": c,
             "side": "人类" if pl == state["human_player"] else "规则AI"}
            for i, (pl, r, c) in enumerate(state["moves"])
        ],
    }
    fname = f"record_{int(time.time())}.json"
    path = os.path.join(RECORD_DIR, fname)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    return path


# ── HTTP 处理 ─────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # 静默日志
        pass

    def _send_json(self, data: dict, status: int = 200):
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html: str):
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/index.html":
            self._send_html(HTML_PAGE)
        elif path == "/api/state":
            with _lock:
                if not _state:
                    self._send_json({"error": "no game"}, 400)
                    return
                s = _state
                self._send_json({
                    "board":        _board_to_list(s["board"]),
                    "current":      s["board"].current_player,
                    "human_player": s["human_player"],
                    "over":         s["over"],
                    "winner":       s["winner"],
                    "move_count":   s["board"].move_count,
                    "moves":        s["moves"],
                })
        elif path == "/api/scores":
            with _lock:
                if not _state or _state["over"]:
                    self._send_json({"scores": []})
                    return
                self._send_json({"scores": _get_scores_map(_state["board"])})
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}

        if path == "/api/new_game":
            human = int(body.get("human_player", 1))
            with _lock:
                _state.clear()
                _state.update(_new_game(human))
                # 若规则先手（人类执白），立刻走一步
                if _state["board"].current_player != human:
                    act = _rule_move(_state["board"])
                    r, c = divmod(act, BOARD_SIZE)
                    _state["moves"].append((_state["board"].current_player, r, c))
                    _state["board"].do_move(act)
            self._send_json({"ok": True})

        elif path == "/api/human_move":
            row = int(body.get("row", -1))
            col = int(body.get("col", -1))
            with _lock:
                if not _state or _state["over"]:
                    self._send_json({"error": "no active game"}, 400)
                    return
                s = _state
                board = s["board"]
                if board.current_player != s["human_player"]:
                    self._send_json({"error": "not your turn"}, 400)
                    return
                action = row * BOARD_SIZE + col
                if action not in board._avail_set:
                    self._send_json({"error": "illegal move"}, 400)
                    return
                # 人类落子
                s["moves"].append((board.current_player, row, col))
                board.do_move(action)

                if board.game_over():
                    s["over"] = True
                    s["winner"] = board.winner
                    _save_record(s)
                    self._send_json({"over": True, "winner": board.winner})
                    return

                # 规则程序响应
                act = _rule_move(board)
                ar, ac = divmod(act, BOARD_SIZE)
                s["moves"].append((board.current_player, ar, ac))
                board.do_move(act)

                if board.game_over():
                    s["over"] = True
                    s["winner"] = board.winner
                    _save_record(s)

                self._send_json({
                    "over":      s["over"],
                    "winner":    s["winner"],
                    "ai_move":   {"row": ar, "col": ac},
                })

        elif path == "/api/resign":
            with _lock:
                if _state and not _state["over"]:
                    _state["over"] = True
                    _state["winner"] = 3 - _state["human_player"]
                    _save_record(_state)
            self._send_json({"ok": True})

        else:
            self.send_response(404)
            self.end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()


# ── HTML 页面 ─────────────────────────────────────
HTML_PAGE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>规则AI对战 Demo</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#1a1a2e;color:#e0e0e0;font-family:'Segoe UI',sans-serif;
     display:flex;flex-direction:column;align-items:center;min-height:100vh;padding:20px}
h1{font-size:1.4rem;margin-bottom:10px;color:#a8d8ea}
#controls{display:flex;gap:10px;margin-bottom:12px;flex-wrap:wrap;justify-content:center}
button{padding:8px 16px;border:none;border-radius:6px;cursor:pointer;
       font-size:.9rem;font-weight:600;transition:.2s}
.btn-primary{background:#4a90d9;color:#fff}
.btn-primary:hover{background:#357ab8}
.btn-danger{background:#c0392b;color:#fff}
.btn-danger:hover{background:#a93226}
.btn-toggle{background:#2d4a7a;color:#ccc}
.btn-toggle.on{background:#27ae60;color:#fff}
select{padding:7px 10px;border-radius:6px;background:#2d3561;color:#e0e0e0;border:1px solid #4a5b8c}
#status{font-size:1rem;margin-bottom:10px;min-height:1.5em;color:#f0c040;text-align:center}
#wrap{position:relative;display:inline-block}
canvas{display:block;border-radius:8px;cursor:crosshair}
#info{margin-top:10px;font-size:.82rem;color:#888;text-align:center}
#movelog{margin-top:12px;max-height:180px;overflow-y:auto;width:100%;
         max-width:460px;background:#111827;border-radius:8px;padding:8px;font-size:.78rem;color:#9ca3af}
#movelog span{display:inline-block;margin:1px 3px;padding:1px 5px;
              border-radius:3px;background:#1f2937}
#movelog span.human{color:#60a5fa}
#movelog span.ai{color:#f87171}
</style>
</head>
<body>
<h1>规则AI对战 Demo</h1>
<div id="controls">
  <select id="side">
    <option value="1">我执黑（先手）</option>
    <option value="2">我执白（后手）</option>
  </select>
  <button class="btn-primary" onclick="newGame()">新游戏</button>
  <button class="btn-toggle" id="heatBtn" onclick="toggleHeat()">热图 OFF</button>
  <button class="btn-danger" onclick="resign()">认输</button>
</div>
<div id="status">点击「新游戏」开始</div>
<div id="wrap"><canvas id="c"></canvas></div>
<div id="info">落子记录自动保存到 rule_demo_records/</div>
<div id="movelog"></div>

<script>
const BOARD=15,CELL=38,PAD=30;
const W=PAD*2+(BOARD-1)*CELL,H=W;
const canvas=document.getElementById('c');
const ctx=canvas.getContext('2d');
canvas.width=W;canvas.height=H;

let board=Array.from({length:BOARD},()=>Array(BOARD).fill(0));
let over=false,humanPlayer=1,currentPlayer=1,showHeat=false;
let heatData=null,lastAI=null,moveCount=0;
const log=document.getElementById('movelog');

function px(i){return PAD+i*CELL}

function drawBoard(){
  ctx.fillStyle='#3d2b1f';
  ctx.fillRect(0,0,W,H);
  // 星位
  const stars=BOARD===15?[[3,3],[3,11],[7,7],[11,3],[11,11]]:[];
  ctx.strokeStyle='#6b4c2a';ctx.lineWidth=1;
  for(let i=0;i<BOARD;i++){
    ctx.beginPath();ctx.moveTo(PAD,px(i));ctx.lineTo(PAD+(BOARD-1)*CELL,px(i));ctx.stroke();
    ctx.beginPath();ctx.moveTo(px(i),PAD);ctx.lineTo(px(i),PAD+(BOARD-1)*CELL);ctx.stroke();
  }
  stars.forEach(([r,c])=>{
    ctx.beginPath();ctx.arc(px(c),px(r),3,0,Math.PI*2);
    ctx.fillStyle='#6b4c2a';ctx.fill();
  });
}

function drawHeat(){
  if(!showHeat||!heatData)return;
  for(let r=0;r<BOARD;r++)for(let c=0;c<BOARD;c++){
    const v=heatData[r][c];
    if(v<0||board[r][c]!==0)continue;
    const alpha=0.12+v*0.55;
    ctx.fillStyle=`rgba(255,${Math.round(200*(1-v))},0,${alpha})`;
    ctx.fillRect(px(c)-CELL/2+1,px(r)-CELL/2+1,CELL-2,CELL-2);
  }
}

function drawStones(){
  for(let r=0;r<BOARD;r++)for(let c=0;c<BOARD;c++){
    if(!board[r][c])continue;
    const x=px(c),y=px(r),rad=CELL*0.44;
    const g=ctx.createRadialGradient(x-rad*.3,y-rad*.3,rad*.1,x,y,rad);
    if(board[r][c]===1){g.addColorStop(0,'#666');g.addColorStop(1,'#111');}
    else{g.addColorStop(0,'#fffde7');g.addColorStop(1,'#ccc');}
    ctx.beginPath();ctx.arc(x,y,rad,0,Math.PI*2);
    ctx.fillStyle=g;ctx.fill();
    // 最后一步AI落点标记
    if(lastAI&&lastAI[0]===r&&lastAI[1]===c){
      ctx.beginPath();ctx.arc(x,y,rad*.38,0,Math.PI*2);
      ctx.fillStyle='rgba(255,60,60,.85)';ctx.fill();
    }
  }
}

function redraw(){drawBoard();drawHeat();drawStones();}

function addLog(player,r,c){
  const sp=document.createElement('span');
  const isHuman=(player===humanPlayer);
  sp.className=isHuman?'human':'ai';
  const col='ABCDEFGHIJKLMNOPQRSTUVWXYZ'[c];
  sp.textContent=`${isHuman?'我':'AI'}${col}${r+1}`;
  log.appendChild(sp);
  log.scrollTop=log.scrollHeight;
}

function setStatus(msg){document.getElementById('status').textContent=msg;}

async function fetchState(){
  const r=await fetch('/api/state');
  if(!r.ok)return;
  const d=await r.json();
  board=d.board;over=d.over;currentPlayer=d.current;
  humanPlayer=d.human_player;moveCount=d.move_count;
  redraw();
  if(over)setStatus(d.winner===0?'平局！':d.winner===humanPlayer?'🥳 你赢了！':'😅 规则AI获胜！');
  else setStatus(`当前：${currentPlayer===humanPlayer?'你（'+( humanPlayer===1?'黑':'白')+'棋）':'规则AI思考中...'}`);
}

async function fetchHeat(){
  if(!showHeat)return;
  const r=await fetch('/api/scores');
  const d=await r.json();
  heatData=d.scores||null;
  redraw();
}

async function newGame(){
  over=false;lastAI=null;heatData=null;log.innerHTML='';
  humanPlayer=parseInt(document.getElementById('side').value);
  await fetch('/api/new_game',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({human_player:humanPlayer})});
  const r=await fetch('/api/state');const d=await r.json();
  board=d.board;currentPlayer=d.current;over=d.over;moveCount=d.move_count;
  // 如果AI已经走了第一步（人类执白），记录它
  if(d.moves&&d.moves.length>0){
    const [pl,row,col]=d.moves[d.moves.length-1];
    lastAI=[row,col];addLog(pl,row,col);
  }
  redraw();
  await fetchHeat();
  setStatus(`当前：${currentPlayer===humanPlayer?'你（'+(humanPlayer===1?'黑':'白')+'棋）':'规则AI思考中...'}`);
}

canvas.addEventListener('click',async e=>{
  if(over||currentPlayer!==humanPlayer)return;
  const rect=canvas.getBoundingClientRect();
  const sx=canvas.width/rect.width,sy=canvas.height/rect.height;
  const mx=(e.clientX-rect.left)*sx,my=(e.clientY-rect.top)*sy;
  let best=null,bd=Infinity;
  for(let r=0;r<BOARD;r++)for(let c=0;c<BOARD;c++){
    const d=Math.hypot(mx-px(c),my-px(r));
    if(d<bd){bd=d;best=[r,c];}
  }
  if(!best||bd>CELL*.6)return;
  const [row,col]=best;
  if(board[row][col]!==0)return;

  const resp=await fetch('/api/human_move',{method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({row,col})});
  const d=await resp.json();
  if(d.error){setStatus('⚠ '+d.error);return;}

  board[row][col]=humanPlayer;
  addLog(humanPlayer,row,col);
  lastAI=null;

  if(d.ai_move){
    const {row:ar,col:ac}=d.ai_move;
    board[ar][ac]=3-humanPlayer;
    lastAI=[ar,ac];
    addLog(3-humanPlayer,ar,ac);
  }
  over=d.over;
  redraw();
  await fetchHeat();
  if(over)setStatus(d.winner===0?'平局！':d.winner===humanPlayer?'🥳 你赢了！':'😅 规则AI获胜！');
  else setStatus('你（'+(humanPlayer===1?'黑':'白')+'棋）的回合');
});

async function resign(){
  if(over)return;
  await fetch('/api/resign',{method:'POST'});
  over=true;setStatus('你认输了。');redraw();
}

function toggleHeat(){
  showHeat=!showHeat;
  const b=document.getElementById('heatBtn');
  b.textContent=showHeat?'热图 ON':'热图 OFF';
  b.className=showHeat?'btn-toggle on':'btn-toggle';
  if(showHeat)fetchHeat();else{heatData=null;redraw();}
}

// 初始绘制空棋盘
drawBoard();
</script>
</body>
</html>
"""


# ── 启动服务器 ─────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="规则AI对战 Demo")
    parser.add_argument("--port", type=int, default=7070)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    args = parser.parse_args()

    server = HTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}"
    print(f"规则AI对战 Demo 已启动：{url}")
    print(f"对局记录保存到：{RECORD_DIR}")
    print("Ctrl+C 停止\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止。")


if __name__ == "__main__":
    main()
