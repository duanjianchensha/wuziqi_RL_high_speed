/**
 * game.js — 游戏控制器
 *
 * 职责：
 *  - 管理 UI 状态（设置面板 ↔ 游戏面板）
 *  - 与 FastAPI 后端通信
 *  - 协调 BoardRenderer 渲染
 *  - 处理用户交互（点击落子、提示、认输）
 */

(function () {
  "use strict";

  // ── DOM 引用 ────────────────────────────────
  const $ = id => document.getElementById(id);
  const setupPanel = $("setup-panel");
  const gamePanel = $("game-panel");
  const statusText = $("status-text");
  const playerIndicator = $("player-indicator");
  const thinkingBar = $("thinking-bar");
  const moveLog = $("move-log");
  const resultOverlay = $("result-overlay");
  const resultIcon = $("result-icon");
  const resultTitle = $("result-title");
  const resultDesc = $("result-desc");
  const canvas = $("board-canvas");

  const btnStart = $("btn-start");
  const btnHint = $("btn-hint");
  const btnResign = $("btn-resign");
  const btnNew = $("btn-new");
  const btnResultNew = $("btn-result-new");
  const difficultyEasyLabel = $("difficulty-easy");
  const difficultyMediumLabel = $("difficulty-medium");
  const difficultyHardLabel = $("difficulty-hard");

  const btnReloadModel = $("btn-reload-model");

  // ── 游戏状态 ─────────────────────────────────
  let renderer = null;
  let sessionId = null;
  let gameState = null;   // 最近服务器返回的状态
  let waiting = false;  // 是否正在等待 AI / 服务器
  let boardSize = 8;
  let canvasEventsBound = false;

  function getColLabel(col) {
    const alphabet = "ABCDEFGHJKLMNOPQRSTUVWXYZ";
    if (col < alphabet.length) return alphabet[col];
    let n = col;
    let out = "";
    while (n >= 0) {
      out = alphabet[n % alphabet.length] + out;
      n = Math.floor(n / alphabet.length) - 1;
    }
    return out;
  }

  // ── 工具 ─────────────────────────────────────
  function action2rc(action) {
    return { r: Math.floor(action / boardSize), c: action % boardSize };
  }
  function rc2action(r, c) { return r * boardSize + c; }

  async function api(method, path, body) {
    const opts = { method, headers: { "Content-Type": "application/json" } };
    if (body) opts.body = JSON.stringify(body);
    const res = await fetch(path, opts);
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || res.statusText);
    }
    return res.json();
  }

  async function loadDifficultyConfig() {
    try {
      const data = await api("GET", "/api/difficulty_config");
      const cfg = data.difficulty_playout || {};
      if (difficultyEasyLabel) {
        difficultyEasyLabel.innerHTML =
          '<input type="radio" name="difficulty" value="easy" /> 简单（' + (cfg.easy ?? "-") + '次模拟）';
      }
      if (difficultyMediumLabel) {
        difficultyMediumLabel.innerHTML =
          '<input type="radio" name="difficulty" value="medium" checked /> 中等（' + (cfg.medium ?? "-") + '次模拟）';
      }
      if (difficultyHardLabel) {
        difficultyHardLabel.innerHTML =
          '<input type="radio" name="difficulty" value="hard" /> 困难（' + (cfg.hard ?? "-") + '次模拟）';
      }
    } catch (e) {
      console.warn("加载动态难度配置失败，使用默认文案", e);
    }
  }

  function bindCanvasEventsOnce() {
    if (canvasEventsBound) return;

    // 鼠标悬停
    canvas.addEventListener("mousemove", e => {
      if (!renderer) return;
      if (!gameState || gameState.game_over || waiting) return;
      if (gameState.current_player !== gameState.human_player) return;
      const rect = canvas.getBoundingClientRect();
      const cell = renderer.xyToCell(e.clientX - rect.left, e.clientY - rect.top);
      if (cell) renderer.setHover(cell.r, cell.c);
      else renderer.setHover(null, null);
    });
    canvas.addEventListener("mouseleave", () => {
      if (!renderer) return;
      renderer.setHover(null, null);
    });

    // 点击落子
    canvas.addEventListener("click", e => {
      if (!renderer) return;
      if (!gameState || gameState.game_over || waiting) return;
      if (gameState.current_player !== gameState.human_player) return;
      const rect = canvas.getBoundingClientRect();
      const action = renderer.pixelToAction(e.clientX - rect.left, e.clientY - rect.top);
      if (action === null) return;
      doHumanMove(action);
    });

    // 触摸支持
    canvas.addEventListener("touchend", e => {
      if (!renderer) return;
      e.preventDefault();
      if (!gameState || gameState.game_over || waiting) return;
      if (gameState.current_player !== gameState.human_player) return;
      const touch = e.changedTouches[0];
      const rect = canvas.getBoundingClientRect();
      const action = renderer.pixelToAction(
        touch.clientX - rect.left, touch.clientY - rect.top);
      if (action !== null) doHumanMove(action);
    }, { passive: false });

    canvasEventsBound = true;
  }

  // ── 初始化渲染器 ─────────────────────────────
  function initRenderer(humanPlayer, size) {
    renderer = new BoardRenderer(canvas, size);
    renderer.humanPlayer = humanPlayer;
    bindCanvasEventsOnce();
  }

  // ── 更新棋盘显示 ─────────────────────────────
  function applyState(state) {
    gameState = state;
    boardSize = state.board_size || boardSize;
    renderer.setState(state.board, state.last_move, state.human_player);
    updateStatusBar(state);
    if (state.game_over) {
      showResult(state);
    }
  }

  function updateStatusBar(state) {
    const stoneSymbol = p => p === 1 ? "⚫" : "⚪";
    if (state.game_over) {
      statusText.textContent = state.winner === 0 ? "平局" :
        `${stoneSymbol(state.winner)} ${state.winner === state.human_player ? "你赢了" : "AI 赢了"}`;
      playerIndicator.textContent = "";
    } else {
      const cp = state.current_player;
      const who = cp === state.human_player ? "你" : "AI";
      statusText.textContent = `${stoneSymbol(cp)} ${who}的回合`;
      playerIndicator.textContent = `第 ${state.move_count + 1} 手`;
    }
  }

  // ── 落子记录 ─────────────────────────────────
  function addMoveLog(action, player, extra) {
    const { r, c } = action2rc(action);
    const tag = document.createElement("span");
    tag.className = `move-tag ${player === 1 ? "black" : "white"}`;
    tag.textContent = `${getColLabel(c)}${boardSize - r}${extra || ""}`;
    moveLog.appendChild(tag);
    moveLog.scrollLeft = 99999;
  }

  // ── 人类落子 ─────────────────────────────────
  async function doHumanMove(action) {
    if (waiting) return;
    renderer.clearHeatmap();
    setWaiting(true);
    try {
      const state = await api("POST", `/api/human_move/${sessionId}`, { action });
      const { r, c } = action2rc(action);
      renderer.animateStone(r, c, state.human_player, () => {
        applyState(state);
        addMoveLog(action, state.human_player);
        if (!state.game_over) scheduleAiMove();
        else setWaiting(false);
      });
    } catch (err) {
      alert("落子出错：" + err.message);
      setWaiting(false);
    }
  }

  // ── AI 落子 ──────────────────────────────────
  async function scheduleAiMove() {
    setWaiting(true);
    showThinking(true);
    try {
      const state = await api("POST", `/api/ai_move/${sessionId}`);
      const action = state.ai_action;
      const { r, c } = action2rc(action);
      showThinking(false);
      renderer.animateStone(r, c, state.ai_player, () => {
        applyState(state);
        addMoveLog(action, state.ai_player, " AI");
        setWaiting(false);
      });
    } catch (err) {
      showThinking(false);
      alert("AI 出错：" + err.message);
      setWaiting(false);
    }
  }

  // ── 提示按钮 ─────────────────────────────────
  btnHint.addEventListener("click", async () => {
    if (!sessionId || waiting || gameState?.game_over) return;
    if (gameState.current_player !== gameState.human_player) return;
    try {
      const data = await api("GET", `/api/hint/${sessionId}`);
      renderer.setHeatmap(data.heatmap);
      // 3 秒后自动清除热图
      setTimeout(() => renderer.clearHeatmap(), 3000);
    } catch (e) {
      console.warn("提示失败", e);
    }
  });

  // ── 认输 ─────────────────────────────────────
  btnResign.addEventListener("click", async () => {
    if (!sessionId || gameState?.game_over) return;
    if (!confirm("确认认输？")) return;
    try {
      const state = await api("POST", `/api/resign/${sessionId}`);
      applyState(state);
    } catch (e) {
      console.warn("认输请求失败", e);
    }
  });

  // ── 新游戏 ────────────────────────────────────
  function resetToSetup() {
    resultOverlay.classList.add("hidden");
    gamePanel.classList.add("hidden");
    setupPanel.classList.remove("hidden");
    sessionId = null;
    gameState = null;
    moveLog.innerHTML = "";
  }
  btnNew.addEventListener("click", resetToSetup);
  btnResultNew.addEventListener("click", resetToSetup);

  // ── 开始游戏 ──────────────────────────────────
  btnStart.addEventListener("click", async () => {
    const humanPlayer = parseInt(
      document.querySelector('input[name="color"]:checked').value);
    const difficulty = document.querySelector('input[name="difficulty"]:checked').value;

    btnStart.disabled = true;
    btnStart.textContent = "正在连接…";
    try {
      const state = await api("POST", "/api/new_game", {
        human_player: humanPlayer,
        difficulty,
      });
      sessionId = state.session_id;

      setupPanel.classList.add("hidden");
      gamePanel.classList.remove("hidden");
      moveLog.innerHTML = "";

      boardSize = state.board_size || boardSize;
      initRenderer(humanPlayer, boardSize);
      applyState(state);

      // 若人类执白（AI 先手），立即触发 AI 落子
      if (humanPlayer === 2) {
        await scheduleAiMove();
      }
    } catch (err) {
      alert("创建游戏失败：" + err.message);
    } finally {
      btnStart.disabled = false;
      btnStart.textContent = "开始游戏";
    }
  });

  // ── 胜负对话框 ────────────────────────────────
  function showResult(state) {
    let icon, title, desc;
    if (state.winner === 0) {
      icon = "🤝"; title = "平局"; desc = "势均力敌，再战一局！";
    } else if (state.winner === state.human_player) {
      icon = "🎉"; title = "你赢了！"; desc = "恭喜，战胜了 AlphaZero AI！";
    } else {
      icon = "🤖"; title = "AI 赢了"; desc = "不要气馁，再试一次！";
    }
    resultIcon.textContent = icon;
    resultTitle.textContent = title;
    resultDesc.textContent = desc;
    // 延迟显示，等动画完毕
    setTimeout(() => resultOverlay.classList.remove("hidden"), 600);
  }

  // ── 辅助 ─────────────────────────────────────
  function setWaiting(flag) {
    waiting = flag;
    canvas.style.cursor = flag ? "wait" : "pointer";
    btnHint.disabled = flag;
    btnResign.disabled = flag;
  }

  function showThinking(flag) {
    thinkingBar.classList.toggle("hidden", !flag);
  }

  btnReloadModel?.addEventListener("click", async () => {
    try {
      await api("POST", "/api/reload_model");
      alert("已从磁盘热重载模型权重。");
    } catch (e) {
      alert(e.message);
    }
  });

  // ── 模型选择 ─────────────────────────────────
  const selPlayModel = $("sel-play-model");
  const btnApplyModel = $("btn-apply-model");

  async function loadModelSelect() {
    if (!selPlayModel) return;
    const presets = ["models/best_policy.pth", "models/current_policy.pth"];
    let prefPath = "";
    try {
      const pr = await api("GET", "/api/model/prefs");
      if (pr.prefs && pr.prefs.play_model_path) prefPath = String(pr.prefs.play_model_path);
    } catch (_e) { }
    selPlayModel.innerHTML = "";
    presets.forEach(p => {
      const o = document.createElement("option");
      o.value = p;
      o.textContent = p;
      selPlayModel.appendChild(o);
    });
    try {
      const ck = await api("GET", "/api/model/checkpoints?limit=40");
      (ck.checkpoints || []).forEach(c => {
        const o = document.createElement("option");
        o.value = c.path;
        o.textContent = c.name;
        selPlayModel.appendChild(o);
      });
    } catch (_e) { }
    if (prefPath) {
      let found = false;
      for (let i = 0; i < selPlayModel.options.length; i++) {
        if (selPlayModel.options[i].value === prefPath) {
          selPlayModel.selectedIndex = i;
          found = true;
          break;
        }
      }
      if (!found) {
        const o = document.createElement("option");
        o.value = prefPath;
        o.textContent = prefPath + " (当前)";
        selPlayModel.insertBefore(o, selPlayModel.firstChild);
        selPlayModel.selectedIndex = 0;
      }
    }
  }

  btnApplyModel?.addEventListener("click", async () => {
    if (!selPlayModel || !selPlayModel.value) return;
    try {
      await api("POST", "/api/model/set_play_path", {
        path: selPlayModel.value,
        reload: true,
      });
      alert("已保存对弈权重并热重载。");
    } catch (e) {
      alert(e.message);
    }
  });

  loadDifficultyConfig();
  loadModelSelect();
})();
