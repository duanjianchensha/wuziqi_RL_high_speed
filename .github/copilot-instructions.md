# 五子棋 AlphaZero 项目指南

> 基于 AlphaZero 范式的五子棋强化学习 AI（Python/PyTorch + C++ MCTS + FastAPI Web）

详细文档见 [README.md](../README.md)

---

## 架构概览

| 组件 | 文件 | 状态 |
|------|------|------|
| 规则数据生成 | `scripts/gen_rule_data.py` | ✅ |
| 监督预训练入口 | `pretrain.py` | ✅ |
| 训练入口 | `train.py` | ✅ |
| Web 服务入口 | `server.py` | ✅ |
| 配置中心 | `gomoku/config.py` | ✅ 所有超参数 |
| 棋盘环境 | `gomoku/game.py` | ✅ |
| 策略-价值网络 | `gomoku/neural_net.py` | ✅ ResNet(128/7) |
| MCTS 搜索 | `gomoku/mcts.py` | ✅ Python + C++ fallback |
| 训练主循环 | `gomoku/coach.py` | ✅ |
| 经验回放池 | `gomoku/replay_buffer.py` | ✅ |
| 规则策略玩家 | `gomoku/rule_player.py` | ✅ 威胁评估 |
| 数据加载工具 | `gomoku/data_utils.py` | ✅ load_npz_files / list_checkpoints |
| Web API | `web/app.py` | ✅ FastAPI |
| 会话管理 | `web/game_session.py` | ✅ 30min TTL |
| C++ 引擎 | `src/` + `CMakeLists.txt` | ⚠️ Phase 2 约 30% |

**三阶段流水线**：
1. `gen_rule_data.py` → 规则对弈数据 (`models/human_games/rule_data/`)
2. `pretrain.py` → 监督预训练 (`models/current_policy.pth`)
3. `train.py` → AlphaZero 自弈强化 (`models/best_policy.pth`)

**两阶段策略**：Phase 1 = 纯 Python（当前可用）；Phase 2 = C++ pybind11 加速 MCTS，自动降级：`import gomoku_cpp` 失败时回退 Python 实现。

---

## 构建与运行

```bash
# 安装依赖
pip install -r requirements.txt

# 完整流水线（Windows）
.\full_train_pipeline.ps1

# 分步执行
python scripts/gen_rule_data.py --games 5000 --workers 4
python pretrain.py --epochs 80
python train.py --fresh

# 从断点继续自弈
python train.py

# 小规模快速验证
python train.py --games 100 --playout 200

# 启动 Web 对战服务（默认 http://127.0.0.1:8000）
python server.py

# 编译 C++ 加速模块（需要 VS2019+ / CMake 3.18+）
scripts\build.bat
```

---

## 核心约定

### 动作编码
- 棋盘动作编码为平铺下标：`action = row * BOARD_SIZE + col`
- 逆变换：`row, col = divmod(action, BOARD_SIZE)`

### 神经网络输入（4 通道）
```
ch0: 当前玩家的棋子位置
ch1: 对方棋子位置
ch2: 上一步落子位置（全 1 at 最后一步）
ch3: 当前玩家（黑子全 1，白子全 0）
```

### 自弈数据格式
- 标准 AlphaZero 三元组：`(state: ndarray(4,N,N), mcts_policy: ndarray(N²), z: float)`
- 训练时在线做 8 折对称增强（4 旋转 × 2 翻转），不存储

### NPZ 文件格式
- 键：`states`, `mcts_probs`, `winners`, `board_size`
- 由 `gen_rule_data.py` 生成，`gomoku/data_utils.load_npz_files()` 读取

### 混合回放
- `MIX_EXPERT_REPLAY_RATIO=0.15`：每 batch 约 15% 采样自规则预训练数据
- 数据来自 `config.RULE_DATA_DIR`（`models/human_games/rule_data/`）
- `coach._reload_expert_mix_data()` 在 Coach 构造时一次性加载

### 多进程（Windows 兼容）
- `torch.multiprocessing.set_start_method('spawn')` 已在 `train.py` 设置
- Worker 函数 `_selfplay_worker` 必须是**顶层函数**（非类方法、非 lambda）
- 模型权重通过 `pickle bytes` 传递，不用 `torch.multiprocessing.Queue`

### 设备管理
- 训练主进程：`config.DEVICE`（优先 CUDA）
- Worker 子进程：`config.WORKER_DEVICE`（通常 CPU，避免 VRAM 碎片）
- AMP 混合精度：仅 CUDA 开启，CPU 自动禁用

---

## 常用修改位置

| 任务 | 文件 | 关键配置 |
|------|------|---------|
| 改棋盘大小 | `gomoku/config.py` | `BOARD_SIZE` |
| 调整 MCTS 强度 | `gomoku/config.py` | `N_PLAYOUT_TRAIN`, `C_PUCT` |
| 修改学习率/批次 | `gomoku/config.py` | `LR`, `BATCH_SIZE`, `EPOCHS_PER_UPDATE` |
| 更改 Web 难度系数 | `gomoku/config.py` | `WEB_DIFFICULTY_MULTIPLIERS` |
| 新增 API 接口 | `web/app.py` | `@app.get/post(...)` |
| 修改自弈逻辑 | `gomoku/coach.py` | `Coach.learn()` 主循环 |
| 修改损失函数 | `gomoku/neural_net.py` | `PolicyValueNet.train_step()` |
| 规则玩家强度 | `gomoku/rule_player.py` | 威胁评分权重 |
| C++ 绑定 | `src/bindings.cpp` | pybind11 模块定义 |

---

## Web API 速查

| 端点 | 方法 | 作用 |
|------|------|------|
| `/` | GET | 主页面 |
| `/api/new_game` | POST | 新建对局 |
| `/api/state/{sid}` | GET | 获取当前棋盘 |
| `/api/human_move/{sid}` | POST | 人类落子 |
| `/api/ai_move/{sid}` | POST | AI 落子（MCTS 计算）|
| `/api/hint/{sid}` | GET | AI 落子概率热图 |
| `/api/resign/{sid}` | POST | 认输 |
| `/api/difficulty_config` | GET | 当前难度配置 |
| `/api/model/checkpoints` | GET | 可切换权重列表 |
| `/api/model/set_play_path` | POST | 热切换对战模型 |
| `/api/reload_model` | POST | 热重载模型 |
| `/api/train/start` | POST | 后台启动训练 |
| `/api/train/status` | GET | 查询训练状态 |
| `/api/train/log` | GET | 获取训练日志 |
| `/api/train/stop` | POST | 停止训练 |

---

## 注意事项

**避免踩坑**：
- ❌ 不要在 Worker 进程内调用 `torch.cuda`（spawn 模式下不继承 CUDA 上下文）
- ❌ 不要把 `config.py` 的 `BUFFER_SIZE` 改得太大（会稀释新数据）
- ❌ C++ build 调试前确认 pybind11 已通过 pip 安装或作为 git submodule
- ✅ 测试参数变更时先用 `--games 50 --playout 100` 快速验证流程
- ✅ 加载 npz 数据统一使用 `gomoku.data_utils.load_npz_files()`，不要直接调用旧模块

