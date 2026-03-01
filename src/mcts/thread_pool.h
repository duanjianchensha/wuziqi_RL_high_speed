// src/mcts/thread_pool.h
// 通用 C++17 线程池（供并行自弈使用）
// Phase 2 加速模块

#pragma once
#include <functional>
#include <thread>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <future>
#include <atomic>

class ThreadPool {
public:
    explicit ThreadPool(size_t n_threads) : stop_(false) {
        for (size_t i = 0; i < n_threads; ++i) {
            workers_.emplace_back([this] {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lk(mtx_);
                        cv_.wait(lk, [this] { return stop_ || !tasks_.empty(); });
                        if (stop_ && tasks_.empty()) return;
                        task = std::move(tasks_.front());
                        tasks_.pop();
                    }
                    task();
                }
            });
        }
    }

    template<typename F, typename... Args>
    auto submit(F&& f, Args&&... args)
        -> std::future<std::invoke_result_t<F, Args...>>
    {
        using return_t = std::invoke_result_t<F, Args...>;
        auto pkg = std::make_shared<std::packaged_task<return_t()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...));
        std::future<return_t> fut = pkg->get_future();
        {
            std::unique_lock<std::mutex> lk(mtx_);
            if (stop_) throw std::runtime_error("ThreadPool stopped");
            tasks_.emplace([pkg] { (*pkg)(); });
        }
        cv_.notify_one();
        return fut;
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lk(mtx_);
            stop_ = true;
        }
        cv_.notify_all();
        for (auto& w : workers_) w.join();
    }

private:
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex mtx_;
    std::condition_variable cv_;
    bool stop_;
};
