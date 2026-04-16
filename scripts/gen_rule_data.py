"""
scripts/gen_rule_data.py
使用规则策略自弈，批量生成监督预训练数据（.npz 格式）。

流程：
  1. 规则玩家 vs 规则玩家（含噪声保证多样性）
  2. 记录每步的棋局状态 + 软概率标签 + 终局结果
  3. 保存为标准 npz 文件（与 pretrain.py 兼容）

典型用法：
  python scripts/gen_rule_data.py --games 10000
  python scripts/gen_rule_data.py --games 5000 --workers 4
  python scripts/gen_rule_data.py --games 10000 --out-dir models/human_games/rule_data

完成后执行预训练：
  python pretrain.py --data-dir models/human_games/rule_data --epochs 60
  python train.py --resume models/current_policy.pth
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
import uuid

import numpy as np

# 确保可以 import 项目包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ────────────────────────────────────────────────
# Worker 函数（顶层，spawn 兼容）
# ────────────────────────────────────────────────

def _play_one_game(args: dict) -> dict:
    """
    执行一局规则 vs 规则对弈，返回该局的训练样本字典。

    返回字段：
      states   : (T, 4, N, N) float32
      probs    : (T, N²)      float32  软概率标签
      zs       : (T,)         float32  终局结果（行棋方视角）
      mc       : (T,)         int32    落子前已走步数
      winner   : int          终局赢家 (0/1/2)
    """
    # 每个进程独立设随机种子
    seed = args.get("seed", None)
    if seed is not None:
        np.random.seed(seed)

    from gomoku.config import config
    from gomoku.game import Board
    from gomoku.rule_player import get_action_scores, scores_to_probs

    board_size = args.get("board_size", config.BOARD_SIZE)
    n_in_row   = args.get("n_in_row",   config.N_IN_ROW)
    score_temp = args.get("score_temp", 0.5)
    noise_eps  = args.get("noise_eps",  0.15)
    noise_alpha = args.get("noise_alpha", 0.3)
    # 前 random_open_moves 步完全随机落子，增加开局多样性
    random_open_moves = args.get("random_open_moves", 2)

    board = Board(board_size, n_in_row)
    # 随机决定哪方先手，避免黑方数据过多
    if np.random.random() < 0.5:
        board.current_player = 2

    nn = board_size * board_size
    states_list = []
    probs_list  = []
    players_list = []
    mc_list     = []

    while not board.game_over():
        avail = board.availables
        if not avail:
            break

        state = board.get_current_state()
        mc    = board.move_count

        # 开局前几步随机落子（增加多样性），但仍记录规则评分作为标签
        if mc < random_open_moves:
            scores = get_action_scores(board)
            probs  = scores_to_probs(scores, avail, score_temp)
            # 随机选动作（忽略规则偏好）
            action = int(np.random.choice(avail))
        else:
            scores = get_action_scores(board)
            probs  = scores_to_probs(scores, avail, score_temp)

            # 混入 Dirichlet 噪声（仅在采样动作时，不影响标签 probs）
            if noise_eps > 0 and len(avail) > 1:
                noise_vec = np.zeros(nn, dtype=np.float32)
                nd = np.random.dirichlet(np.ones(len(avail)) * noise_alpha)
                for i, a in enumerate(avail):
                    noise_vec[a] = nd[i]
                noisy = (1.0 - noise_eps) * probs + noise_eps * noise_vec
                s = noisy.sum()
                if s > 0:
                    noisy /= s
                action = int(np.random.choice(nn, p=noisy))
            else:
                action = int(np.argmax(probs))

        states_list.append(state)
        probs_list.append(probs.copy())
        players_list.append(board.current_player)
        mc_list.append(mc)

        board.do_move(action)

    winner = int(board.winner) if board.winner is not None else 0

    if not states_list:
        return {
            "states":  np.zeros((0, 4, board_size, board_size), np.float32),
            "probs":   np.zeros((0, nn), np.float32),
            "zs":      np.zeros((0,), np.float32),
            "mc":      np.zeros((0,), np.int32),
            "winner":  winner,
            "n_steps": 0,
        }

    # 回填终局结果（行棋方视角）
    zs = []
    for pl in players_list:
        if winner == 0:
            zs.append(0.0)
        elif pl == winner:
            zs.append(1.0)
        else:
            zs.append(-1.0)

    return {
        "states":  np.stack(states_list).astype(np.float32),
        "probs":   np.stack(probs_list).astype(np.float32),
        "zs":      np.array(zs, dtype=np.float32),
        "mc":      np.array(mc_list, dtype=np.int32),
        "winner":  winner,
        "n_steps": len(states_list),
    }


# ────────────────────────────────────────────────
# 文件保存
# ────────────────────────────────────────────────

def _save_batch(
    batch: list,
    out_dir: str,
    board_size: int,
) -> str:
    """将一批对局数据合并保存为单个 npz 文件。"""
    all_states = np.concatenate([b["states"] for b in batch], axis=0)
    all_probs  = np.concatenate([b["probs"]  for b in batch], axis=0)
    all_zs     = np.concatenate([b["zs"]     for b in batch], axis=0)
    all_mc     = np.concatenate([b["mc"]     for b in batch], axis=0)
    last_winner = batch[-1]["winner"]

    name = f"game_{int(time.time())}_{uuid.uuid4().hex[:8]}.npz"
    path = os.path.join(out_dir, name)
    np.savez_compressed(
        path,
        states=all_states,
        mcts_probs=all_probs,
        winners=all_zs,
        move_count_before=all_mc,
        winner_end=np.int32(last_winner),
        board_size=np.int32(board_size),
        meta_json=np.array('{"source":"rule_player","version":"1.0"}'),
    )
    return path


# ────────────────────────────────────────────────
# 主函数
# ────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="规则策略自弈数据生成（用于神经网络监督预训练）"
    )
    parser.add_argument("--games",   type=int,   default=5000,  help="生成对局数（建议 5000~20000）")
    parser.add_argument("--out-dir", type=str,   default=None,  help="npz 输出目录（默认 models/human_games/rule_data）")
    parser.add_argument("--workers", type=int,   default=1,     help="并行 worker 进程数（1=单进程，建议 ≤CPU核数-1）")
    parser.add_argument("--games-per-file", type=int, default=200,
                        help="每个 npz 文件包含的对局数（影响文件大小，不影响训练效果）")
    parser.add_argument("--score-temp",  type=float, default=0.5,
                        help="规则评分 → 概率的 softmax 温度（越低越贪心，建议 0.3~0.7）")
    parser.add_argument("--noise-eps",   type=float, default=0.15,
                        help="Dirichlet 探索噪声混合比例（建议 0.10~0.20）")
    parser.add_argument("--noise-alpha", type=float, default=0.3,
                        help="Dirichlet 浓度参数（建议 0.2~0.5）")
    parser.add_argument("--random-open", type=int,  default=2,
                        help="开局随机落子步数，增加开局多样性（建议 2~4）")
    args = parser.parse_args()

    from gomoku.config import config

    out_dir = args.out_dir or config.RULE_DATA_DIR
    os.makedirs(out_dir, exist_ok=True)

    board_size = config.BOARD_SIZE
    n_in_row   = config.N_IN_ROW

    print("=" * 60)
    print(f"  规则策略自弈数据生成")
    print(f"  棋盘: {board_size}×{board_size}  胜利条件: {n_in_row}连")
    print(f"  对局数: {args.games}  并行: {args.workers} worker(s)")
    print(f"  输出目录: {out_dir}")
    print(f"  标签温度: {args.score_temp}  探索噪声: ε={args.noise_eps} α={args.noise_alpha}")
    print("=" * 60)

    worker_args_list = [
        {
            "seed":             i * 31337 + 42,  # 可复现但多样
            "board_size":       board_size,
            "n_in_row":         n_in_row,
            "score_temp":       args.score_temp,
            "noise_eps":        args.noise_eps,
            "noise_alpha":      args.noise_alpha,
            "random_open_moves": args.random_open,
        }
        for i in range(args.games)
    ]

    t0 = time.time()
    total_steps = 0
    total_games = 0
    files_saved = 0
    batch: list = []

    def flush_batch():
        nonlocal files_saved, batch
        if not batch:
            return
        _save_batch(batch, out_dir, board_size)
        files_saved += 1
        batch = []

    if args.workers > 1:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=args.workers) as pool:
            for result in pool.imap_unordered(
                _play_one_game, worker_args_list, chunksize=4
            ):
                total_steps += result["n_steps"]
                total_games += 1
                batch.append(result)

                if len(batch) >= args.games_per_file:
                    flush_batch()

                if total_games % 500 == 0 or total_games == args.games:
                    elapsed = time.time() - t0
                    spd = total_games / max(elapsed, 0.001)
                    eta = (args.games - total_games) / max(spd, 0.001)
                    print(
                        f"  [{total_games:>6}/{args.games}] "
                        f"样本 {total_steps:>8,}  "
                        f"{spd:>5.1f} 局/秒  "
                        f"剩余 {eta:>5.0f}s"
                    )
    else:
        for i, wa in enumerate(worker_args_list):
            result = _play_one_game(wa)
            total_steps += result["n_steps"]
            total_games += 1
            batch.append(result)

            if len(batch) >= args.games_per_file:
                flush_batch()

            if total_games % 200 == 0 or total_games == args.games:
                elapsed = time.time() - t0
                spd = total_games / max(elapsed, 0.001)
                eta = (args.games - total_games) / max(spd, 0.001)
                print(
                    f"  [{total_games:>6}/{args.games}] "
                    f"样本 {total_steps:>8,}  "
                    f"{spd:>5.1f} 局/秒  "
                    f"剩余 {eta:>5.0f}s"
                )

    flush_batch()

    elapsed = time.time() - t0
    avg_len = total_steps / max(total_games, 1)
    print()
    print("=" * 60)
    print(f"  生成完成！")
    print(f"  总对局: {total_games:,}  总样本: {total_steps:,}")
    print(f"  平均局长: {avg_len:.1f} 步  npz 文件数: {files_saved}")
    print(f"  总耗时: {elapsed:.1f}s  平均: {elapsed/max(total_games,1)*1000:.0f}ms/局")
    print()
    print("  下一步：执行监督预训练")
    print(f"    python pretrain.py --data-dir \"{out_dir}\" --epochs 60")
    print()
    print("  预训练完成后继续自弈强化：")
    print(f"    python train.py --resume models/current_policy.pth")
    print("=" * 60)


if __name__ == "__main__":
    # Windows spawn 兼容
    mp.freeze_support()
    main()
