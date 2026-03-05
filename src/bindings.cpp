// src/bindings.cpp
// pybind11 module entry - exposing C++ Game engine and MCTS to Python
// Phase 2 acceleration module
//
// Usage in Python:
//   import gomoku_cpp
//   board = gomoku_cpp.Board()
//   board.do_move(27)
//   player = gomoku_cpp.MCTSPlayer(policy_fn, c_puct=5.0, n_playout=400)
//   action, probs = player.get_move(board, temp=1e-3)

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <random>

#include "game/gomoku.h"
#include "mcts/mcts.h"

namespace py = pybind11;
using namespace gomoku;
using namespace mcts;

// Python NN inference function signature:
//   fn(state: np.ndarray shape=(4,N,N)) -> (probs: np.ndarray(N*N,), value: float)
using PyInferenceFn = std::function<
    std::pair<py::array_t<float>, float>(py::array_t<float>)>;

// Wrapper: adapt Python NN function as C++ InferenceFn
InferenceFn wrap_infer(PyInferenceFn py_fn, int board_size)
{
    return [py_fn, board_size](const float *state_in, float *probs_out, float &value_out)
    {
        py::gil_scoped_acquire acquire; // Calling Python requires GIL
        // Construct numpy array view, zero-copy
        std::array<py::ssize_t, 3> shape = {4, board_size, board_size};
        py::array_t<float> state_arr(shape, state_in);
        auto [probs_arr, value] = py_fn(state_arr);
        value_out = value;
        auto buf = probs_arr.request();
        int n_squares = board_size * board_size;
        std::memcpy(probs_out, buf.ptr, n_squares * sizeof(float));
    };
}

// -- C++ MCTSPlayer Wrapper -------------------------
class CppMCTSPlayer
{
public:
    CppMCTSPlayer(PyInferenceFn py_fn, float c_puct, int n_playout,
                  int board_size, int n_threads = 1)
        : board_size_(board_size),
          mcts_(wrap_infer(py_fn, board_size), c_puct, n_playout, n_threads),
          rng_(std::random_device{}()) {}

    // Returns (action, full_prob_vector: np.ndarray(N*N,))
    std::pair<int, py::array_t<float>>
    get_move(const Board &board, float temp = 1e-3f)
    {
        int n_squares = board.n_squares();
        std::vector<int> acts(n_squares);
        std::vector<float> probs(n_squares);
        int n = mcts_.search(board, acts.data(), probs.data(), temp);

        // Construct full probs vector
        py::array_t<float> full_probs(n_squares);
        auto buf = full_probs.mutable_unchecked<1>();
        for (int i = 0; i < n_squares; ++i)
            buf(i) = 0.f;
        for (int i = 0; i < n; ++i)
            buf(acts[i]) = probs[i];

        // 使用线程安全的 mt19937 采样（替代全局状态 rand()）
        std::uniform_real_distribution<float> dist(0.f, 1.f);
        float r = dist(rng_);
        float cum = 0.f;
        int chosen = acts[0];
        for (int i = 0; i < n; ++i)
        {
            cum += probs[i];
            if (r <= cum)
            {
                chosen = acts[i];
                break;
            }
        }
        mcts_.update_with_move(chosen);
        return {chosen, full_probs};
    }

    void reset() { mcts_.update_with_move(-1); }

private:
    int board_size_;
    MCTS mcts_;
    std::mt19937 rng_;
};

// -- Module definition ----------------------------------
PYBIND11_MODULE(gomoku_cpp, m)
{
    m.doc() = "Gomoku AlphaZero C++ acceleration module (Phase 2)";

    // Board
    py::class_<Board>(m, "Board")
        .def(py::init<int, int>(), py::arg("size") = 15, py::arg("n_in_row") = 5)
        .def("do_move", &Board::do_move)
        .def("availables", &Board::availables)
        .def("game_over", &Board::game_over)
        .def("reset", &Board::reset)
        .def_readonly("move_cnt", &Board::move_cnt)
        .def_readonly("last_move", &Board::last_move)
        .def_property_readonly("size", &Board::size)
        .def_readonly("n_in_row", &Board::n_in_row)
        .def_property_readonly("winner", [](const Board &b) -> py::object
                               {
            if (!b.winner) return py::none();
            return py::int_(*b.winner); })
        .def_property_readonly("current_player", [](const Board &b)
                               { return static_cast<int>(b.current); })
        .def("get_features", [](const Board &b)
             {
            py::array_t<float> out({4, b.size(), b.size()});
            b.get_features(out.mutable_data());
            return out; })
        .def("copy", [](const Board &b)
             { return b; }); // 值拷贝

    // MCTSPlayer
    py::class_<CppMCTSPlayer>(m, "MCTSPlayer")
        .def(py::init<PyInferenceFn, float, int, int, int>(),
             py::arg("policy_value_fn"),
             py::arg("c_puct") = 5.f,
             py::arg("n_playout") = 400,
             py::arg("board_size") = 15,
             py::arg("n_threads") = 1)
        .def("get_move", &CppMCTSPlayer::get_move,
             py::arg("board"), py::arg("temp") = 1e-3f)
        .def("reset", &CppMCTSPlayer::reset);

    m.attr("BOARD_SIZE") = 15;
    m.attr("N_SQUARES") = 225;
    m.attr("N_IN_ROW") = 5;
}
