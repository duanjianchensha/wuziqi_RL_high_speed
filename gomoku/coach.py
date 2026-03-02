"""
gomoku/coach.py
AlphaZero 自弈训练协调器（Phase 1: 纯 Python + multiprocessing）

并行策略：
  - 主进程持有神经网络（训练/推理）
  - N 个 worker 进程接收模型权重 → 进行本地自弈 → 返回训练数据
  - Windows 兼容（spawn 启动方式）：worker 函数必须定义在顶层

TODO (Phase 2): 替换为 C++ MCTS worker，通过 pybind11 调用
"""

import os
import time
import pickle
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple

import numpy as np

from gomoku.config import config
from gomoku.game import Board, Game
from gomoku.mcts import MCTSPlayer, PureMCTSPlayer
from gomoku.neural_net import PolicyValueFunction
from gomoku.replay_buffer import ReplayBuffer


# ────────────────────────────────────────────────
# Worker 函数（必须为顶层函数，Windows spawn 兼容）
# ────────────────────────────────────────────────
def _selfplay_worker(args: dict) -> List[Tuple]:
    """
    在子进程中执行 n_games 局自弈，返回训练数据列表。
    args 包含: model_weights(bytes), n_games, n_playout, board_size, n_in_row
    """
    import io, torch, sys, os
    from gomoku.neural_net import PolicyValueNet
    from gomoku.game import Board, Game
    from gomoku.config import config

    # 优先尝试导入 C++ 模块以实现 Phase 2 加速
    USE_CPP = False
    try:
        release_path = os.path.join(os.getcwd(), "Release")
        if release_path not in sys.path:
            sys.path.append(release_path)
        import gomoku_cpp  # type: ignore[import-not-found]

        USE_CPP = True
    except ImportError:
        from gomoku.mcts import MCTSPlayer

    board_size = args["board_size"]
    n_in_row = args["n_in_row"]
    n_playout = args["n_playout"]
    n_games = args["n_games"]
    weights = args["model_weights"]

    # 构建本地网络（CPU 即可，工作进程不需要 GPU）
    net = PolicyValueNet(board_size)
    buf = io.BytesIO(weights)
    net.load_state_dict(torch.load(buf, map_location="cpu", weights_only=True))
    net.eval()

    import torch.nn.functional as F

    def policy_value_fn(board_or_state):
        if USE_CPP:
            # gomoku_cpp.MCTSPlayer 传入的是 board.get_features() 后的 numpy 数组
            state = board_or_state
        else:
            state = board_or_state.get_current_state()

        t = torch.from_numpy(state).unsqueeze(0)
        with torch.no_grad():
            logp, v = net(t)

        probs = F.softmax(logp, dim=1).squeeze(0).numpy()
        if USE_CPP:
            return probs, float(v.item())
        else:
            avail = board_or_state.availables
            return list(zip(avail, probs[avail])), float(v.item())

    if USE_CPP:
        player = gomoku_cpp.MCTSPlayer(
            policy_value_fn, c_puct=config.C_PUCT, n_playout=n_playout
        )
    else:
        player = MCTSPlayer(
            policy_value_fn, c_puct=config.C_PUCT, n_playout=n_playout, is_selfplay=True
        )

    if USE_CPP:
        board = gomoku_cpp.Board()
    else:
        board = Board(board_size, n_in_row)

    game = Game(board)

    import time as _time

    worker_id = args.get("worker_id", "?")
    all_data = []
    engine_str = "C++" if USE_CPP else "Python"
    print(
        f"  [Worker-{worker_id}] 启动 ({engine_str} 引擎)，执行 {n_games} 局自弈"
        f"（MCTS模拟={n_playout}次/步，棋盘={board_size}×{board_size}）",
        flush=True,
    )
    for g in range(n_games):
        t_g = _time.time()
        if USE_CPP:
            # C++ Board 由 game.start_self_play 内部处理 init/reset
            winner, play_data = game.start_self_play(player, is_cpp=True)
        else:
            board.reset()
            winner, play_data = game.start_self_play(player)

        all_data.extend(play_data)
        elapsed_g = _time.time() - t_g
        move_count = game.board.move_cnt if USE_CPP else game.board.move_count
        w_str = "黑赢" if winner == 1 else ("白赢" if winner == 2 else "平局")
        print(
            f"  [Worker-{worker_id}] 第{g+1}/{n_games}局 "
            f"{w_str} 共{move_count}步 {engine_str}样本+{len(play_data)} 耗时{elapsed_g:.1f}s",
            flush=True,
        )
    print(
        f"  [Worker-{worker_id}] 完成，共生成 {len(all_data)} 条样本",
        flush=True,
    )
    return all_data


# ────────────────────────────────────────────────
# Coach 主类
# ────────────────────────────────────────────────
class Coach:
    def __init__(self):
        # 初始化 / 加载模型
        best_path = config.BEST_POLICY
        if os.path.exists(best_path):
            print(f"[Coach] 发现已有模型，加载中: {best_path}", flush=True)
        else:
            print(f"[Coach] 未找到已有模型，将使用随机初始化权重", flush=True)
        self.policy = PolicyValueFunction(
            board_size=config.BOARD_SIZE,
            model_path=best_path if os.path.exists(best_path) else None,
        )
        print(
            f"[Coach] 模型初始化完成 | 设备={config.DEVICE} | "
            f"残差块×{config.N_RES_BLOCKS} | 滤波器={config.N_FILTERS}",
            flush=True,
        )
        self.replay_buf = ReplayBuffer(config.BUFFER_SIZE)
        print(f"[Coach] ReplayBuffer 就绪（容量={config.BUFFER_SIZE:,}）", flush=True)
        self.lr_mult = config.LR_MULTIPLIER  # 自适应学习率倍率
        self.n_games_played = 0
        self.win_ratio_history: List[float] = []

    # ── 并行自弈（ProcessPoolExecutor）────────────
    def _collect_selfplay_data(
        self, n_workers: int, games_per_worker: int, n_playout: int
    ) -> int:
        """
        启动 n_workers 个进程，每个完成 games_per_worker 局自弈。
        返回本轮采集的样本总数。
        """
        weights = self.policy.get_weights()
        worker_args = {
            "model_weights": weights,
            "n_games": games_per_worker,
            "n_playout": n_playout,
            "board_size": config.BOARD_SIZE,
            "n_in_row": config.N_IN_ROW,
        }

        total_samples = 0
        t_collect = time.time()
        print(
            f"[Coach] 启动 {n_workers} 个 Worker 进程，"
            f"每个执行 {games_per_worker} 局（MCTS模拟={n_playout}次/步）...",
            flush=True,
        )
        # ProcessPoolExecutor 在 Windows spawn 下安全
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(
                    _selfplay_worker,
                    {**worker_args, "worker_id": i + 1},
                ): i
                + 1
                for i in range(n_workers)
            }
            completed = 0
            for fut in as_completed(futures):
                wid = futures[fut]
                completed += 1
                try:
                    data = fut.result()
                    self.replay_buf.push(data)
                    total_samples += len(data)
                    self.n_games_played += games_per_worker
                    print(
                        f"[Coach] Worker-{wid} 完成 ({completed}/{n_workers}) "
                        f"贡献 {len(data)} 样本 | "
                        f"缓冲池={len(self.replay_buf):,}/{config.BUFFER_SIZE:,}",
                        flush=True,
                    )
                except Exception as e:
                    print(f"[Coach] Worker-{wid} 异常: {e}", flush=True)
        elapsed_collect = time.time() - t_collect
        print(
            f"[Coach] 本轮采集完成: {total_samples} 样本 耗时={elapsed_collect:.1f}s",
            flush=True,
        )
        return total_samples

    # ── 训练步骤 ──────────────────────────────────
    def _train(self) -> dict:
        """
        从 replay buffer 采样 EPOCHS_PER_UPDATE 次，返回最后一次的 loss 信息。
        使用自适应学习率（KL 散度监控）。
        """
        lr = config.LR * self.lr_mult
        last_info = {}
        t_train = time.time()
        print(
            f"[Train] 开始训练 {config.EPOCHS_PER_UPDATE} 个 epoch "
            f"(lr={lr:.2e}, batch={config.BATCH_SIZE}, "
            f"缓冲池={len(self.replay_buf):,})",
            flush=True,
        )

        for epoch in range(config.EPOCHS_PER_UPDATE):
            states, probs, winners = self.replay_buf.sample(config.BATCH_SIZE)
            info = self.policy.train_step(states, probs, winners, lr=lr)
            last_info = info

            # 自适应学习率调整
            kl = info["kl"]
            if kl > config.KL_TARGET * 2 and self.lr_mult > 0.1:
                self.lr_mult /= 1.5
                lr_tag = "lr↓"
            elif kl < config.KL_TARGET / 2 and self.lr_mult < 10:
                self.lr_mult *= 1.5
                lr_tag = "lr↑"
            else:
                lr_tag = "lr─"
            print(
                f"  [Epoch {epoch+1}/{config.EPOCHS_PER_UPDATE}] "
                f"loss={info['loss']:.4f} "
                f"(p={info['loss_p']:.4f} v={info['loss_v']:.4f}) "
                f"kl={kl:.4f} {lr_tag}",
                flush=True,
            )

        elapsed_train = time.time() - t_train
        print(
            f"[Train] 训练完成 耗时={elapsed_train:.1f}s "
            f"lr_mult={self.lr_mult:.3f}",
            flush=True,
        )
        last_info["lr"] = lr
        last_info["lr_mult"] = self.lr_mult
        return last_info

    # ── 评估：新模型 vs 纯 MCTS ───────────────────
    def _evaluate(self, n_games: int = config.EVAL_GAMES) -> float:
        """
        新模型 vs 纯 MCTS（n_playout=config.N_PLAYOUT_EVAL），
        交换先后手各一半局数，返回新模型胜率。
        """

        def policy_fn(board):
            return self.policy.policy_value_fn(board)

        new_player = MCTSPlayer(
            policy_fn, c_puct=config.C_PUCT, n_playout=config.N_PLAYOUT_TRAIN
        )
        pure_player = PureMCTSPlayer(n_playout=config.N_PLAYOUT_EVAL)
        board = Board(config.BOARD_SIZE, config.N_IN_ROW)
        game = Game(board)

        wins, loses, draws = 0, 0, 0
        t_eval = time.time()
        print(
            f"[Eval] 开始评估 {n_games} 局"
            f"（新模型 vs 纯MCTS-{config.N_PLAYOUT_EVAL}次模拟）",
            flush=True,
        )
        for i in range(n_games):
            t_g = time.time()
            if i % 2 == 0:
                winner = game.start_play(new_player, pure_player, start_player=1)
                new_ind = 1
                role = "新模型执黑"
            else:
                winner = game.start_play(pure_player, new_player, start_player=1)
                new_ind = 2
                role = "新模型执白"
            elapsed_g = time.time() - t_g
            if winner == new_ind:
                wins += 1
                result = "赢"
            elif winner == 0:
                draws += 1
                result = "平"
            else:
                loses += 1
                result = "负"
            print(
                f"  [Eval {i+1}/{n_games}] {role}: {result} "
                f"| 累计 {wins}W/{draws}D/{loses}L "
                f"| 耗时{elapsed_g:.1f}s",
                flush=True,
            )

        win_ratio = (wins + 0.5 * draws) / n_games
        elapsed_eval = time.time() - t_eval
        print(
            f"[Eval] 最终: {wins}赢/{draws}平/{loses}负 "
            f"胜率={win_ratio:.3f} 阈值={config.WIN_RATIO_THRESHOLD} "
            f"总耗时={elapsed_eval:.1f}s",
            flush=True,
        )
        return win_ratio

    # ── 主训练循环 ────────────────────────────────
    def run(self, total_games: int = config.N_SELFPLAY_GAMES):
        """
        主训练循环。
        total_games: 总自弈局数目标
        """
        print(f"[Coach] 开始训练，目标 {total_games} 局自弈")
        print(
            f"[Coach] Workers={config.N_WORKERS}, "
            f"MCTS模拟={config.N_PLAYOUT_TRAIN}, "
            f"BatchSize={config.BATCH_SIZE}, Device={config.DEVICE}"
        )

        iteration = 0
        best_ratio = 0.0

        os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(config.MODEL_DIR, exist_ok=True)

        while self.n_games_played < total_games:
            iteration += 1
            t0 = time.time()
            progress_pct = self.n_games_played / total_games * 100
            bar_len = 30
            filled = int(bar_len * self.n_games_played / total_games)
            bar = "█" * filled + "░" * (bar_len - filled)
            print(f"\n{'─'*60}", flush=True)
            print(
                f"[Iter {iteration}] [{bar}] {progress_pct:.1f}% "
                f"({self.n_games_played}/{total_games}局) "
                f"| 最佳胜率={best_ratio:.3f} "
                f"| 缓冲池={len(self.replay_buf):,}",
                flush=True,
            )
            print(f"{'─'*60}", flush=True)

            # 1. 并行自弈采集
            n_w = config.N_WORKERS
            n_gw = config.GAMES_PER_WORKER
            samples = self._collect_selfplay_data(n_w, n_gw, config.N_PLAYOUT_TRAIN)
            elapsed = time.time() - t0

            print(
                f"[Iter {iteration}] 自弈采集完毕: {samples} 样本 "
                f"(总局数={self.n_games_played}/{total_games}, 耗时={elapsed:.1f}s)",
                flush=True,
            )

            # 2. 训练（缓冲池满足最小批次后才开始）
            if self.replay_buf.ready(config.BATCH_SIZE):
                info = self._train()
                print(
                    f"[Train] loss={info['loss']:.4f}  "
                    f"loss_v={info['loss_v']:.4f}  "
                    f"loss_p={info['loss_p']:.4f}  "
                    f"kl={info['kl']:.4f}  "
                    f"lr={info['lr']:.2e}  "
                    f"lr_mult={info['lr_mult']:.2f}"
                )
            else:
                print(
                    f"[Train] 数据不足，跳过（当前={len(self.replay_buf):,}, "
                    f"需要={config.BATCH_SIZE}）"
                )
                continue

            # 3. 保存当前模型
            cur_path = config.CURRENT_POLICY
            self.policy.save(cur_path)

            # 4. 定期评估
            if iteration % config.CHECK_FREQ == 0:
                print(f"\n[Eval] 第 {iteration} 轮评估...")
                win_ratio = self._evaluate(config.EVAL_GAMES)
                self.win_ratio_history.append(win_ratio)

                ckpt_path = os.path.join(
                    config.CHECKPOINT_DIR,
                    f"policy_iter{iteration}_wr{win_ratio:.3f}.pth",
                )
                self.policy.save(ckpt_path)

                if win_ratio >= config.WIN_RATIO_THRESHOLD:
                    print(
                        f"[Coach] 新最佳模型！胜率={win_ratio:.3f} → 保存 {config.BEST_POLICY}"
                    )
                    self.policy.save(config.BEST_POLICY)
                    best_ratio = win_ratio
                else:
                    if win_ratio < 0.05:
                        # 训练发散，回滚
                        print("[Coach] 警告：胜率过低，加载最佳模型...")
                        if os.path.exists(config.BEST_POLICY):
                            self.policy.load(config.BEST_POLICY)

        print(f"\n[Coach] 训练完成！最佳胜率={best_ratio:.3f}")
        self.policy.save(config.BEST_POLICY)
