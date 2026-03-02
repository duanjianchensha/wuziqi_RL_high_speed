// src/mcts/node.h
// MCTS 树节点（Arena 分配 + 原子访问）
// Phase 2 加速模块

#pragma once
#include <atomic>
#include <vector>
#include <memory>
#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <mutex>
#include <stdexcept>

namespace mcts
{

    // ──────────────────────────────────────────────
    // MCTSNode
    // ──────────────────────────────────────────────
    struct MCTSNode
    {
        MCTSNode *parent = nullptr;
        float P = 0.f; // 策略先验概率

        std::atomic<int> N{0};     // 访问次数
        std::atomic<float> W{0.f}; // 累计价值
        float Q = 0.f;             // 均值价值（受 N 保护）

        // Virtual loss（并行 MCTS 时临时减值以避免多线程走同一路径）
        std::atomic<int> virtual_loss{0};

        // 子节点（action → node）
        std::vector<std::pair<int, MCTSNode *>> children;
        std::mutex expand_mutex; // 扩展时上锁
        bool expanded = false;

        MCTSNode() = default;
        MCTSNode(MCTSNode *par, float prior) : parent(par), P(prior) {}

        // std::atomic / std::mutex 不可拷贝；提供显式移动构造供 vector 内部使用
        // （move 后旧对象不再使用，mutex 保持默认构造状态即可）
        MCTSNode(MCTSNode &&o) noexcept
            : parent(o.parent), P(o.P), N(o.N.load(std::memory_order_relaxed)), W(o.W.load(std::memory_order_relaxed)), Q(o.Q), virtual_loss(o.virtual_loss.load(std::memory_order_relaxed)), children(std::move(o.children)), expanded(o.expanded)
        {
        }
        MCTSNode &operator=(MCTSNode &&) = delete;
        MCTSNode(const MCTSNode &) = delete;
        MCTSNode &operator=(const MCTSNode &) = delete;

        // ── PUCT 值 ───────────────────────────
        float get_value(float c_puct) const
        {
            int n = N.load(std::memory_order_acquire);
            int par_n = parent ? parent->N.load(std::memory_order_acquire) : 1;
            float vl = static_cast<float>(virtual_loss.load(std::memory_order_relaxed));
            float q_eff = (n > 0) ? (W.load(std::memory_order_acquire) / n - vl * 1.0f) : -1.f;
            float u = c_puct * P * std::sqrt(static_cast<float>(par_n)) / (1.f + n);
            return q_eff + u;
        }

        // ── 加/减 Virtual Loss ─────────────────
        void add_virtual_loss() { virtual_loss.fetch_add(1, std::memory_order_relaxed); }
        void revert_virtual_loss() { virtual_loss.fetch_sub(1, std::memory_order_relaxed); }

        // ── 反向传播（原子更新）────────────────
        void update(float v)
        {
            N.fetch_add(1, std::memory_order_release);
            float old_w, new_w;
            do
            {
                old_w = W.load(std::memory_order_acquire);
                new_w = old_w + v;
            } while (!W.compare_exchange_weak(old_w, new_w,
                                              std::memory_order_release, std::memory_order_acquire));
            Q = W.load(std::memory_order_relaxed) / N.load(std::memory_order_relaxed);
        }

        void update_recursive(float v)
        {
            if (parent)
                parent->update_recursive(-v);
            update(v);
        }

        bool is_leaf() const { return !expanded; }
        bool is_root() const { return parent == nullptr; }
    };

    // ──────────────────────────────────────────────
    // 简单 Arena 分配器（避免 new/delete 碎片化）
    // 注意：容量一旦耗尽不自动扩容，因为扩容会导致 vector 重分配，
    //       令所有已发出的 MCTSNode* 指针悬空。
    // ──────────────────────────────────────────────
    class NodeArena
    {
    public:
        explicit NodeArena(size_t capacity = 500000)
            : pool_(capacity), used_(0) {}

        MCTSNode *alloc(MCTSNode *parent, float prior)
        {
            if (used_ >= pool_.size())
            {
                throw std::overflow_error(
                    "NodeArena capacity exceeded; increase initial capacity");
            }
            MCTSNode *node = &pool_[used_++];
            new (node) MCTSNode(parent, prior);
            return node;
        }

        void reset()
        {
            // 析构所有已使用节点
            for (size_t i = 0; i < used_; ++i)
                pool_[i].~MCTSNode();
            used_ = 0;
        }

        size_t size() const { return used_; }

    private:
        std::vector<MCTSNode> pool_;
        size_t used_;
    };

} // namespace mcts
