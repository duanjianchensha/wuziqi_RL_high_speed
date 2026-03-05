// src/mcts/mcts.h
// C++ MCTS 搜索引擎（PUCT + Virtual Loss 并行）
// Phase 2 加速模块

#pragma once
#include "node.h"
#include "../game/gomoku.h"
#include "thread_pool.h"

#include <functional>
#include <vector>
#include <thread>
#include <future>
#include <queue>
#include <condition_variable>
#include <numeric>
#include <cmath>
#include <memory>

namespace mcts
{

    // NN 推理回调类型（由 Python 侧提供）
    // 输入: 特征数组 (4 * board.n_squares() floats)
    // 输出: (policy probs board.n_squares() floats, value float)
    using InferenceFn = std::function<void(
        const float *state_in, // (4, N, N) 展平
        float *probs_out,      // (N*N,)
        float &value_out)>;

    // ──────────────────────────────────────────────
    // MCTS 搜索器
    // ──────────────────────────────────────────────
    class MCTS
    {
    public:
        MCTS(InferenceFn infer_fn, float c_puct = 5.f, int n_playout = 400,
             int n_threads = 1)
            : infer_fn_(std::move(infer_fn)), c_puct_(c_puct),
              n_playout_(n_playout), arena_(3000000),
              pool_(n_threads > 1 ? std::make_unique<ThreadPool>(n_threads) : nullptr)
        {
            root_ = arena_.alloc(nullptr, 1.f);
        }

        // 执行 n_playout 次模拟，返回根节点处各动作访问次数分布
        // actions_out, visits_out: 调用方分配（至少 board.n_squares() 大小）
        int search(const gomoku::Board &board,
                   int *actions_out, float *probs_out,
                   float temp = 1e-3f)
        {
            if (pool_)
            {
                // 并行 playout：spawn n_playout 个任务到线程池
                // virtual loss 防止多线程选择同一路径
                std::vector<std::future<void>> futures;
                futures.reserve(n_playout_);
                for (int i = 0; i < n_playout_; ++i)
                {
                    futures.push_back(pool_->submit([this, board]()
                                                    {
                        gomoku::Board b = board;
                        _playout(b, root_); }));
                }
                for (auto &f : futures)
                    f.get(); // 等待所有 playout 完成
            }
            else
            {
                // 单线程路径（默认）
                for (int i = 0; i < n_playout_; ++i)
                {
                    gomoku::Board b = board;
                    _playout(b, root_);
                }
            }

            // 收集子节点访问次数
            auto &ch = root_->children;
            int n = static_cast<int>(ch.size());
            std::vector<float> visits(n);
            for (int i = 0; i < n; ++i)
            {
                actions_out[i] = ch[i].first;
                visits[i] = static_cast<float>(ch[i].second->N.load());
            }

            // 温度采样（数值稳定）
            if (temp <= 1e-2f)
            {
                int best = static_cast<int>(
                    std::max_element(visits.begin(), visits.end()) - visits.begin());
                std::fill(probs_out, probs_out + n, 0.f);
                probs_out[best] = 1.f;
            }
            else
            {
                std::vector<float> logits(n, 0.f);
                float max_logit = -1e30f;
                for (int i = 0; i < n; ++i)
                {
                    // log(v^(1/t)) = log(v)/t, 对 v=0 使用极小值避免 -inf 传播
                    float v = visits[i];
                    float lv = (v > 0.f) ? std::log(v) : -1e30f;
                    logits[i] = lv / temp;
                    if (logits[i] > max_logit)
                        max_logit = logits[i];
                }

                float sum = 0.f;
                for (int i = 0; i < n; ++i)
                {
                    probs_out[i] = std::exp(logits[i] - max_logit);
                    sum += probs_out[i];
                }

                if (sum <= 0.f || !std::isfinite(sum))
                {
                    float uni = 1.f / static_cast<float>(n);
                    for (int i = 0; i < n; ++i)
                        probs_out[i] = uni;
                }
                else
                {
                    for (int i = 0; i < n; ++i)
                        probs_out[i] /= sum;
                }
            }
            return n;
        }

        // 树复用：移动根节点到 last_move 子节点（惰性复用）
        // 找到匹配子节点时不重置 arena——旧节点被遗弃但无需释放（惰性策略）。
        // 仅当 arena 接近满载时才强制重置，避免 overflow_error。
        void update_with_move(int last_move)
        {
            if (last_move >= 0)
            {
                for (auto &[a, child] : root_->children)
                {
                    if (a == last_move)
                    {
                        child->parent = nullptr;
                        root_ = child;
                        // 若 arena 已用量超过 60%，主动重置以防后续溢出
                        if (arena_.size() > arena_.capacity() * 6 / 10)
                        {
                            arena_.reset();
                            root_ = arena_.alloc(nullptr, 1.f);
                        }
                        return; // 成功复用，直接返回
                    }
                }
            }
            // move 未找到或 last_move == -1：完全重置
            arena_.reset();
            root_ = arena_.alloc(nullptr, 1.f);
        }

        int n_playout() const { return n_playout_; }

    private:
        InferenceFn infer_fn_;
        float c_puct_;
        int n_playout_;
        NodeArena arena_;
        MCTSNode *root_ = nullptr;
        std::unique_ptr<ThreadPool> pool_; // nullptr = 单线程模式

        void _playout(gomoku::Board &board, MCTSNode *node)
        {
            // 1. 选择
            while (!node->is_leaf())
            {
                if (board.game_over())
                    break;
                // PUCT 选择最优子节点
                MCTSNode *best_child = nullptr;
                int best_action = -1;
                float best_val = -1e9f;
                for (auto &[a, child] : node->children)
                {
                    float v = child->get_value(c_puct_);
                    if (v > best_val)
                    {
                        best_val = v;
                        best_action = a;
                        best_child = child;
                    }
                }
                if (!best_child)
                    break;
                best_child->add_virtual_loss();
                board.do_move(best_action);
                node = best_child;
            }

            // 2. 扩展 & 估值
            float leaf_value = 0.f;
            if (!board.game_over())
            {
                std::unique_lock<std::mutex> lk(node->expand_mutex);
                if (!node->expanded)
                {
                    // 调用 Python NN 推理
                    int ns = board.n_squares();
                    std::vector<float> features(4 * ns);
                    board.get_features(features.data());
                    std::vector<float> probs(ns, 0.f);
                    infer_fn_(features.data(), probs.data(), leaf_value);

                    // 扩展子节点
                    auto avail = board.availables();
                    for (int a : avail)
                    {
                        node->children.emplace_back(a, arena_.alloc(node, probs[a]));
                    }
                    node->expanded = true;
                }
            }
            else
            {
                auto w = board.winner;
                if (*w == 0)
                    leaf_value = 0.f;
                else
                    leaf_value = (*w == static_cast<int>(board.current)) ? -1.f : 1.f;
            }

            // 3. 撤销 virtual loss + 反向传播
            MCTSNode *cur = node;
            while (cur)
            {
                cur->revert_virtual_loss();
                cur->update(leaf_value);
                leaf_value = -leaf_value;
                cur = cur->parent;
            }
        }
    };

} // namespace mcts
