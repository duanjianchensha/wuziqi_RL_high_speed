/**
 * board.js — Canvas 棋盘渲染器
 *
 * 负责：
 *  - 棋盘线条、坐标标注绘制
 *  - 落子动画（缓入放大）
 *  - 最后一步标记
 *  - AI 推荐热图叠加
 *  - 胜利五连高亮
 *  - 鼠标/触摸悬停预览
 */

class BoardRenderer {
  /**
   * @param {HTMLCanvasElement} canvas
   * @param {number} boardSize  棋盘格数（如 8 → 8×8）
   */
  constructor(canvas, boardSize = 8) {
    this.canvas    = canvas;
    this.ctx       = canvas.getContext("2d");
    this.boardSize = boardSize;

    // 布局参数
    this.PADDING   = 32;   // 棋盘四周留白（像素）
    this.LINE_W    = 1.5;  // 棋盘线宽

    this._initSize();
    this._bindResize();

    // 状态
    this.stones    = [];     // boardSize×boardSize 的二维数组 0/1/2
    this.lastMove  = null;   // 上一步动作 index
    this.heatmap   = null;   // Float32 数组（N*N）或 null
    this.hoverCell = null;   // {r, c} 或 null
    this.winLine   = null;   // 五连格子 [{r,c}, ...]
    this.humanPlayer = 1;
  }

  // ── 初始化 / Resize ─────────────────────────
  _initSize() {
    // 根据窗口宽度计算棋盘边长
    const maxW = Math.min(window.innerWidth - 48, 560);
    const size  = Math.max(280, maxW);
    this.canvas.width  = size;
    this.canvas.height = size;
    this.cellSize = (size - this.PADDING * 2) / (this.boardSize - 1);
    this.stoneR   = this.cellSize * 0.44;
  }

  _bindResize() {
    window.addEventListener("resize", () => {
      this._initSize();
      this.render();
    });
  }

  // ── 坐标转换 ────────────────────────────────
  cellToXY(r, c) {
    return {
      x: this.PADDING + c * this.cellSize,
      y: this.PADDING + r * this.cellSize,
    };
  }

  xyToCell(x, y) {
    const c = Math.round((x - this.PADDING) / this.cellSize);
    const r = Math.round((y - this.PADDING) / this.cellSize);
    if (r < 0 || r >= this.boardSize || c < 0 || c >= this.boardSize) return null;
    return { r, c };
  }

  pixelToAction(x, y) {
    const cell = this.xyToCell(x, y);
    if (!cell) return null;
    return cell.r * this.boardSize + cell.c;
  }

  // ── 主渲染入口 ──────────────────────────────
  render() {
    const ctx  = this.ctx;
    const W    = this.canvas.width;
    const H    = this.canvas.height;
    ctx.clearRect(0, 0, W, H);

    this._drawBackground();
    this._drawGrid();
    this._drawCoords();
    if (this.heatmap) this._drawHeatmap();
    this._drawStones();
    if (this.hoverCell) this._drawHoverPreview();
  }

  // ── 背景 ────────────────────────────────────
  _drawBackground() {
    const ctx = this.ctx;
    ctx.fillStyle = "#d4a96a";
    ctx.beginPath();
    ctx.roundRect(0, 0, this.canvas.width, this.canvas.height, 8);
    ctx.fill();
    // 木纹纹理（简单渐变）
    const grad = ctx.createLinearGradient(0, 0, this.canvas.width, this.canvas.height);
    grad.addColorStop(0, "rgba(255,220,160,.15)");
    grad.addColorStop(1, "rgba(120,70,20,.1)");
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
  }

  // ── 棋盘线 ──────────────────────────────────
  _drawGrid() {
    const ctx = this.ctx;
    ctx.strokeStyle = "#7a4f2a";
    ctx.lineWidth   = this.LINE_W;

    for (let i = 0; i < this.boardSize; i++) {
      const { x: x0, y: y0 } = this.cellToXY(0, i);
      const { x: x1, y: y1 } = this.cellToXY(this.boardSize - 1, i);
      ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();

      const { x: x2, y: y2 } = this.cellToXY(i, 0);
      const { x: x3, y: y3 } = this.cellToXY(i, this.boardSize - 1);
      ctx.beginPath(); ctx.moveTo(x2, y2); ctx.lineTo(x3, y3); ctx.stroke();
    }

    // 天元 / 星位（棋盘中心）
    if (this.boardSize >= 7) {
      const center = Math.floor(this.boardSize / 2);
      this._drawDot(center, center, 4, "#7a4f2a");
    }
  }

  _drawDot(r, c, radius, color) {
    const { x, y } = this.cellToXY(r, c);
    this.ctx.fillStyle = color;
    this.ctx.beginPath();
    this.ctx.arc(x, y, radius, 0, Math.PI * 2);
    this.ctx.fill();
  }

  // ── 坐标标注 ────────────────────────────────
  _drawCoords() {
    const ctx    = this.ctx;
    const cols   = "ABCDEFGHJKLMNOP".split("").slice(0, this.boardSize);
    ctx.fillStyle = "#5a3812";
    ctx.font      = `${Math.round(this.cellSize * 0.3)}px "Microsoft YaHei", sans-serif`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";

    const pad = this.PADDING * 0.6;
    for (let i = 0; i < this.boardSize; i++) {
      const { x, y } = this.cellToXY(i, 0);
      ctx.fillText(String(this.boardSize - i), this.PADDING / 2, y);
      const { x: cx } = this.cellToXY(0, i);
      ctx.fillText(cols[i], cx, this.PADDING / 2);
    }
  }

  // ── 热图（AI 提示） ──────────────────────────
  _drawHeatmap() {
    const ctx = this.ctx;
    const hm  = this.heatmap;
    const max_ = Math.max(...hm);
    if (max_ <= 0) return;

    for (let r = 0; r < this.boardSize; r++) {
      for (let c = 0; c < this.boardSize; c++) {
        const idx = r * this.boardSize + c;
        const v   = hm[idx] / max_;
        if (v < 0.01) continue;
        const { x, y } = this.cellToXY(r, c);
        ctx.fillStyle = `rgba(233,69,96,${(v * 0.65).toFixed(2)})`;
        ctx.beginPath();
        ctx.arc(x, y, this.stoneR * v * 0.8 + 2, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }

  // ── 棋子 ────────────────────────────────────
  _drawStones() {
    if (!this.stones || !this.stones.length) return;
    for (let r = 0; r < this.boardSize; r++) {
      for (let c = 0; c < this.boardSize; c++) {
        const v = Array.isArray(this.stones[r]) ? this.stones[r][c] : 0;
        if (v === 0) continue;
        const isLast = this.lastMove === r * this.boardSize + c;
        const isWin  = this.winLine &&
          this.winLine.some(wc => wc.r === r && wc.c === c);
        this._drawStone(r, c, v, isLast, isWin);
      }
    }
  }

  _drawStone(r, c, player, isLast, isWin) {
    const ctx = this.ctx;
    const { x, y } = this.cellToXY(r, c);
    const radius     = this.stoneR;

    ctx.save();

    // 胜利五连：放大 + 发光
    if (isWin) {
      ctx.shadowColor = player === 1 ? "rgba(80,80,80,.8)" : "rgba(255,255,200,.9)";
      ctx.shadowBlur  = 14;
      ctx.scale(1.08, 1.08);
    }

    // 棋子渐变（3D 效果）
    const grad = ctx.createRadialGradient(x - radius * .3, y - radius * .3, radius * 0.05,
                                           x, y, radius);
    if (player === 1) {
      grad.addColorStop(0, "#606060");
      grad.addColorStop(1, "#0d0d0d");
    } else {
      grad.addColorStop(0, "#ffffff");
      grad.addColorStop(1, "#c0c0c0");
    }
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fill();

    // 边缘描边
    ctx.strokeStyle = player === 1 ? "rgba(255,255,255,.05)" : "rgba(0,0,0,.25)";
    ctx.lineWidth   = 1;
    ctx.stroke();

    // 最后一步标记（红点）
    if (isLast) {
      ctx.fillStyle = "#e94560";
      ctx.beginPath();
      ctx.arc(x, y, radius * 0.2, 0, Math.PI * 2);
      ctx.fill();
    }

    ctx.restore();
  }

  // ── 悬停预览 ─────────────────────────────────
  _drawHoverPreview() {
    const cell = this.hoverCell;
    if (!cell) return;
    const { r, c } = cell;
    if (this.stones[r] && this.stones[r][c] !== 0) return;  // 已有棋子

    const { x, y } = this.cellToXY(r, c);
    const ctx = this.ctx;
    ctx.save();
    ctx.globalAlpha = 0.4;
    const isBlack = this.humanPlayer === 1;
    ctx.fillStyle = isBlack ? "#333" : "#eee";
    ctx.beginPath();
    ctx.arc(x, y, this.stoneR, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  }

  // ── 动画落子（scale in）────────────────────
  animateStone(r, c, player, onDone) {
    const { x, y } = this.cellToXY(r, c);
    const ctx  = this.ctx;
    const maxR = this.stoneR;
    let scale  = 0;
    const step = () => {
      scale = Math.min(scale + 0.12, 1);
      this.render();
      // 在当前帧额外绘制放大中的棋子
      ctx.save();
      ctx.translate(x, y);
      ctx.scale(scale, scale);
      ctx.translate(-x, -y);
      this._drawStone(r, c, player, false, false);
      ctx.restore();
      if (scale < 1) requestAnimationFrame(step);
      else if (onDone) onDone();
    };
    requestAnimationFrame(step);
  }

  // ── 公共方法 ─────────────────────────────────
  setState(board2d, lastMove, humanPlayer) {
    this.stones      = board2d;
    this.lastMove    = lastMove;
    this.humanPlayer = humanPlayer;
    this.winLine     = null;
    this.heatmap     = null;
    this.render();
  }

  setHeatmap(hm) {
    this.heatmap = hm;
    this.render();
  }

  clearHeatmap() {
    this.heatmap = null;
    this.render();
  }

  setWinLine(cells) {
    this.winLine = cells;
    this.render();
  }

  setHover(r, c) {
    this.hoverCell = (r !== null) ? { r, c } : null;
    this.render();
  }
}
