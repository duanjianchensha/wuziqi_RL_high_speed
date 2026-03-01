// src/bindings.cpp
// pybind11 模块入口 — 暴露 C++ 游戏引擎与 MCTS 给 Python
// Phase 2 加速模块
//
// 编译后在 Python 中使用：
//   import gomoku_cpp
//   board = gomoku_cpp.Board()
//   board.do_move(27)
//   player = gomoku_cpp.MCTSPlayer(policy_fn, c_puct=5.0, n_playout=400)
//   action, probs = player.get_move(board, temp=1e-3)

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include "game/gomoku.h"
#include "mcts/mcts.h"

namespace py = pybind11;
using namespace gomoku;
using namespace mcts;

// Python 可调用的 NN 推理函数签名：
//   fn(state: np.ndarray shape=(4,N,N)) -> (probs: np.ndarray(N*N,), value: float)
using PyInferenceFn = std::function<
    std::pair<py::array_t<float>, float>(py::array_t<float>)
>;

// 包装器：将 Python NN 函数适配为 C++ InferenceFn
InferenceFn wrap_infer(PyInferenceFn py_fn) {
    return [py_fn](const float* state_in, float* probs_out, float& value_out) {
        py::gil_scoped_acquire acquire;  // 调用 Python 需要 GIL
        // 构造 numpy 数组视图，零拷贝
        std::array<py::ssize_t, 3> shape = {4, BOARD_SIZE, BOARD_SIZE};
        py::array_t<float> state_arr(shape, state_in);
        auto [probs_arr, value] = py_fn(state_arr);
        value_out = value;
        auto buf = probs_arr.request();
        std::memcpy(probs_out, buf.ptr, N_SQUARES * sizeof(float));
    };
}

// ── C++ MCTSPlayer 包装 ─────────────────────────
class CppMCTSPlayer {
public:
    CppMCTSPlayer(PyInferenceFn py_fn, float c_puct, int n_playout)
        : mcts_(wrap_infer(py_fn), c_puct, n_playout) {}

    // 返回 (action, full_prob_vector: np.ndarray(N*N,))
    std::pair<int, py::array_t<float>>
    get_move(const Board& board, float temp = 1e-3f) {
        std::vector<int>   acts(N_SQUARES);
        std::vector<float> probs(N_SQUARES);
        int n = mcts_.search(board, acts.data(), probs.data(), temp);

        // 构建完整概率向量
        py::array_t<float> full_probs(N_SQUARES);
        auto buf = full_probs.mutable_unchecked<1>();
        for (int i = 0; i < N_SQUARES; ++i) buf(i) = 0.f;
        for (int i = 0; i < n; ++i)         buf(acts[i]) = probs[i];

        // 按概率采样动作
        float r = static_cast<float>(rand()) / RAND_MAX;
        float cum = 0.f;
        int chosen = acts[0];
        for (int i = 0; i < n; ++i) {
            cum += probs[i];
            if (r <= cum) { chosen = acts[i]; break; }
        }
        mcts_.update_with_move(chosen);
        return {chosen, full_probs};
    }

    void reset() { mcts_.update_with_move(-1); }

private:
    MCTS mcts_;
};

// ── 模块定义 ──────────────────────────────────
PYBIND11_MODULE(gomoku_cpp, m) {
    m.doc() = "五子棋 AlphaZero C++ 加速模块 (Phase 2)";

    // Board
    py::class_<Board>(m, "Board")
        .def(py::init<>())
        .def("do_move",    &Board::do_move)
        .def("availables", &Board::availables)
        .def("game_over",  &Board::game_over)
        .def_readonly("move_cnt",  &Board::move_cnt)
        .def_readonly("last_move", &Board::last_move)
        .def_property_readonly("winner", [](const Board& b) -> py::object {
            if (!b.winner) return py::none();
            return py::int_(*b.winner);
        })
        .def_property_readonly("current_player", [](const Board& b) {
            return static_cast<int>(b.current);
        })
        .def("get_features", [](const Board& b) {
            py::array_t<float> out({4, BOARD_SIZE, BOARD_SIZE});
            b.get_features(out.mutable_data());
            return out;
        })
        .def("copy", [](const Board& b) { return b; });  // 值拷贝

    // MCTSPlayer
    py::class_<CppMCTSPlayer>(m, "MCTSPlayer")
        .def(py::init<PyInferenceFn, float, int>(),
             py::arg("policy_value_fn"),
             py::arg("c_puct")    = 5.f,
             py::arg("n_playout") = 400)
        .def("get_move", &CppMCTSPlayer::get_move,
             py::arg("board"), py::arg("temp") = 1e-3f)
        .def("reset", &CppMCTSPlayer::reset);

    m.attr("BOARD_SIZE") = BOARD_SIZE;
    m.attr("N_SQUARES")  = N_SQUARES;
    m.attr("N_IN_ROW")   = N_IN_ROW;
}
