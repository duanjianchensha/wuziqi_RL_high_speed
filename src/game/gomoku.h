// src/game/gomoku.h
// C++ 五子棋棋盘（位棋盘表示），供 C++ MCTS 引擎调用
// Phase 2 加速模块

#pragma once
#include <vector>
#include <cstdint>
#include <optional>
#include <cassert>
#include <stdexcept>
#include <unordered_set>

namespace gomoku
{

    // 玩家编号
    enum class Player : int8_t
    {
        None = 0,
        Black = 1,
        White = 2
    };

    inline Player opponent(Player p)
    {
        return p == Player::Black ? Player::White : Player::Black;
    }

    // ──────────────────────────────────────────────
    // 棋盘状态（值语义，可廉价拷贝）
    // ──────────────────────────────────────────────
    struct Board
    {
        int board_size = 8;
        int n_in_row = 5;
        std::vector<int8_t> cells;          // 0=空, 1=黑, 2=白
        std::unordered_set<int> avail_set_; // 增量合法动作集合，O(1) 删除/查询

        Player current = Player::Black;
        int last_move = -1; // -1 = 无
        int move_cnt = 0;
        std::optional<int> winner; // 0=平局, 1=黑, 2=白 (std::nullopt=未结束)

        Board(int size = 8, int nrow = 5)
            : board_size(size), n_in_row(nrow), cells(size * size, 0)
        {
            if (board_size <= 0)
            {
                throw std::invalid_argument("board_size must be positive");
            }
            if (n_in_row <= 1 || n_in_row > board_size)
            {
                throw std::invalid_argument("n_in_row must be in [2, board_size]");
            }
            // 初始化合法动作集合
            avail_set_.reserve(size * size);
            for (int i = 0; i < size * size; ++i)
                avail_set_.insert(i);
        }

        int size() const { return board_size; }
        int n_squares() const { return board_size * board_size; }

        void reset()
        {
            std::fill(cells.begin(), cells.end(), 0);
            current = Player::Black;
            last_move = -1;
            move_cnt = 0;
            winner.reset();
            avail_set_.clear();
            for (int i = 0; i < n_squares(); ++i)
                avail_set_.insert(i);
        }

        // ── 查询 ────────────────────────────────
        bool occupied(int idx) const
        {
            return cells[idx] != 0;
        }

        Player at(int idx) const
        {
            if (cells[idx] == 1)
                return Player::Black;
            if (cells[idx] == 2)
                return Player::White;
            return Player::None;
        }

        bool game_over() const { return winner.has_value(); }

        std::vector<int> availables() const
        {
            // O(k) 返回剩余合法动作，由增量集合维护而非全盘扫描
            return std::vector<int>(avail_set_.begin(), avail_set_.end());
        }

        // ── 落子 ────────────────────────────────
        void do_move(int action)
        {
            assert(action >= 0 && action < n_squares());
            assert(!occupied(action));
            cells[action] = static_cast<int8_t>(current);
            avail_set_.erase(action); // O(1) 增量删除
            last_move = action;
            ++move_cnt;

            int r = action / board_size, c = action % board_size;
            if (_check_winner(r, c))
            {
                winner = static_cast<int>(current);
            }
            else if (avail_set_.empty()) // O(1) 空检查
            {
                winner = 0; // 平局
            }
            current = opponent(current);
        }

        // ── 胜负判断 ─────────────────────────────
        bool _check_winner(int r, int c) const
        {
            int player = static_cast<int>(current);

            const int dirs[4][2] = {{1, 0}, {0, 1}, {1, 1}, {1, -1}};
            for (auto &d : dirs)
            {
                int cnt = 1;
                for (int sign : {1, -1})
                {
                    int nr = r + sign * d[0], nc = c + sign * d[1];
                    while (nr >= 0 && nr < board_size && nc >= 0 && nc < board_size)
                    {
                        int idx = nr * board_size + nc;
                        if (cells[idx] != player)
                            break;
                        ++cnt;
                        nr += sign * d[0];
                        nc += sign * d[1];
                    }
                }
                if (cnt >= n_in_row)
                    return true;
            }
            return false;
        }

        // ── 神经网络特征（4 通道，展平为 float 数组）───
        // 调用方分配 4 * n_squares() 的 float 缓冲区
        void get_features(float *out) const
        {
            int ns = n_squares();
            int cur = static_cast<int>(current);
            int opp = (cur == 1) ? 2 : 1;

            float *ch0 = out;
            float *ch1 = out + ns;
            float *ch2 = out + 2 * ns;
            float *ch3 = out + 3 * ns;
            for (int i = 0; i < ns; ++i)
            {
                ch0[i] = (cells[i] == cur) ? 1.f : 0.f;
                ch1[i] = (cells[i] == opp) ? 1.f : 0.f;
                ch2[i] = (last_move == i) ? 1.f : 0.f;
                ch3[i] = (current == Player::Black) ? 1.f : 0.f;
            }
        }
    };

} // namespace gomoku
