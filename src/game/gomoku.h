// src/game/gomoku.h
// C++ 五子棋棋盘（位棋盘表示），供 C++ MCTS 引擎调用
// Phase 2 加速模块

#pragma once
#include <array>
#include <vector>
#include <cstdint>
#include <optional>
#include <cassert>

namespace gomoku {

constexpr int BOARD_SIZE = 8;
constexpr int N_SQUARES  = BOARD_SIZE * BOARD_SIZE;
constexpr int N_IN_ROW   = 5;

// 玩家编号
enum class Player : int8_t { None = 0, Black = 1, White = 2 };

inline Player opponent(Player p) {
    return p == Player::Black ? Player::White : Player::Black;
}

// ──────────────────────────────────────────────
// 棋盘状态（值语义，可廉价拷贝）
// ──────────────────────────────────────────────
struct Board {
    // 使用 uint64_t 位棋盘：两个 bitboard 分别记录黑/白棋
    uint64_t black_bb  = 0;   // bit i = 1 表示 (i/SIZE, i%SIZE) 有黑棋
    uint64_t white_bb  = 0;
    Player   current   = Player::Black;
    int      last_move = -1;  // -1 = 无
    int      move_cnt  = 0;
    std::optional<int> winner; // 0=平局, 1=黑, 2=白 (std::nullopt=未结束)

    // ── 查询 ────────────────────────────────
    bool occupied(int idx) const {
        uint64_t mask = 1ULL << idx;
        return (black_bb | white_bb) & mask;
    }

    Player at(int idx) const {
        uint64_t mask = 1ULL << idx;
        if (black_bb & mask) return Player::Black;
        if (white_bb & mask) return Player::White;
        return Player::None;
    }

    bool game_over() const { return winner.has_value(); }

    std::vector<int> availables() const {
        std::vector<int> avail;
        avail.reserve(N_SQUARES - move_cnt);
        for (int i = 0; i < N_SQUARES; ++i)
            if (!occupied(i)) avail.push_back(i);
        return avail;
    }

    // ── 落子 ────────────────────────────────
    void do_move(int action) {
        assert(!occupied(action));
        uint64_t mask = 1ULL << action;
        if (current == Player::Black) black_bb |= mask;
        else                          white_bb |= mask;
        last_move = action;
        ++move_cnt;

        int r = action / BOARD_SIZE, c = action % BOARD_SIZE;
        if (_check_winner(r, c)) {
            winner = static_cast<int>(current);
        } else if (move_cnt == N_SQUARES) {
            winner = 0; // 平局
        }
        current = opponent(current);
    }

    // ── 胜负判断 ─────────────────────────────
    bool _check_winner(int r, int c) const {
        Player p = (current == Player::Black) ? Player::Black : Player::White;
        uint64_t bb = (p == Player::Black) ? black_bb : white_bb;

        const int dirs[4][2] = {{1,0},{0,1},{1,1},{1,-1}};
        for (auto& d : dirs) {
            int cnt = 1;
            for (int sign : {1, -1}) {
                int nr = r + sign * d[0], nc = c + sign * d[1];
                while (nr >= 0 && nr < BOARD_SIZE && nc >= 0 && nc < BOARD_SIZE) {
                    int idx = nr * BOARD_SIZE + nc;
                    if (!((bb >> idx) & 1)) break;
                    ++cnt;
                    nr += sign * d[0];
                    nc += sign * d[1];
                }
            }
            if (cnt >= N_IN_ROW) return true;
        }
        return false;
    }

    // ── 神经网络特征（4 通道，展平为 float 数组）───
    // 调用方分配 4 * N_SQUARES 的 float 缓冲区
    void get_features(float* out) const {
        uint64_t cur_bb, opp_bb;
        if (current == Player::Black) {
            cur_bb = black_bb; opp_bb = white_bb;
        } else {
            cur_bb = white_bb; opp_bb = black_bb;
        }
        float* ch0 = out;
        float* ch1 = out + N_SQUARES;
        float* ch2 = out + 2 * N_SQUARES;
        float* ch3 = out + 3 * N_SQUARES;
        for (int i = 0; i < N_SQUARES; ++i) {
            ch0[i] = (cur_bb >> i) & 1 ? 1.f : 0.f;
            ch1[i] = (opp_bb >> i) & 1 ? 1.f : 0.f;
            ch2[i] = (last_move == i) ? 1.f : 0.f;
            ch3[i] = (current == Player::Black) ? 1.f : 0.f;
        }
    }
};

} // namespace gomoku
