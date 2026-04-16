# 五子棋 AlphaZero AI — 项目文档

> 基于 AlphaZero 范式的五子棋强化学习 AI。  
> 流水线：**规则数据生成 → 监督预训练 → AlphaZero 自弈强化**，  
> Python/PyTorch 实现训练，FastAPI + Canvas 提供 Web 人机对战界面。

---

## 目录

1. [项目概览](#1-项目概览)
2. [环境搭建](#2-环境搭建)
3. [快速开始](#3-快速开始)
4. [强化学习算法详解](#4-强化学习算法详解)
5. [游戏环境详解](#5-游戏环境详解)
6. [神经网络架构](#6-神经网络架构)
7. [MCTS 搜索引擎](#7-mcts-搜索引擎)
8. [训练流水线](#8-训练流水线)
9. [Web 对战界面](#9-web-对战界面)
10. [C++ 加速模块（Phase 2）](#10-c-加速模块phase-2)
11. [配置参数说明](#11-配置参数说明)
12. [性能基准](#12-性能基准)
13. [常见问题](#13-常见问题)
14. [参考资料](#14-参考资料)

---

## 1. 项目概览

### 1.1 三阶段训练流水线

```
scripts/gen_rule_data.py          # 规则策略自弈 → models/human_games/rule_data/*.npz
         ↓
pretrain.py                       # 监督预训练 → models/current_policy.pth
         ↓
train.py                          # AlphaZero 自弈强化 → models/best_policy.pth
```

**规则预训练的必要性**：纯随机初始化网络在 15×15 棋盘上需要极长时间才能探索出有效落子策略。用规则策略（威胁评估 + 搜索）生成的 5000～20000 局数据先做监督预训练，可将自弈起点的棋力提高约 20～30 代自弈效果，大幅缩短总训练时间。

### 1.2 技术栈

| 层次 | 技术 |
|------|------|
| 神经网络 | Python 3.10+, PyTorch 2.x |
| 并行自弈 | Python multiprocessing / C++17 线程池（Phase 2）|
| MCTS 引擎 | Python（Phase 1）/ C++ + pybind11（Phase 2）|
| Web 后端 | FastAPI, Uvicorn |
| Web 前端 | HTML5 Canvas, Vanilla JS |
| 构建工具 | CMake 3.18+, MSVC / GCC-11+ |

### 1.3 文件结构

```
/
├── train.py                   ← 自弈强化训练入口
├── pretrain.py                ← 规则数据监督预训练入口
├── server.py                  ← Web 服务入口
├── full_train_pipeline.ps1    ← 一键三阶段流水线（Windows）
├── requirements.txt
├── CMakeLists.txt             ← C++ 构建配置
│
├── gomoku/                    ← Python 核心包
│   ├── config.py              ← 所有超参数
│   ├── game.py                ← 棋盘环境
│   ├── neural_net.py          ← ResNet 策略-价值网络
│   ├── mcts.py                ← MCTS 搜索（PUCT + Dirichlet 噪声）
│   ├── replay_buffer.py       ← 经验回放池
│   ├── coach.py               ← 自弈训练主循环
│   ├── rule_player.py         ← 规则策略玩家（数据生成用）
│   ├── data_utils.py          ← npz 数据加载工具
│   └── web_ui_prefs.py        ← Web 界面偏好持久化
│
├── scripts/
│   ├── gen_rule_data.py       ← 批量生成规则数据
│   ├── web_train_pipeline.py  ← Web 一键训练子进程包装
│   ├── recommend_train_params.py ← 硬件自动推荐超参数
│   └── build.bat              ← C++ 模块编译脚本
│
├── web/
│   ├── app.py                 ← FastAPI 路由
│   ├── game_session.py        ← 会话管理（30 分钟 TTL）
│   ├── train_job.py           ← 后台训练任务管理
│   └── static/
│       ├── index.html
│       └── js/game.js
│
├── models/                    ← 权重与训练状态
│   ├── best_policy.pth
│   ├── current_policy.pth
│   ├── checkpoints/           ← 历史快照
│   └── human_games/rule_data/ ← 规则预训练数据 (*.npz)
│
└── src/                       ← C++ 加速模块（Phase 2）
    ├── game/gomoku.h
    ├── mcts/
    └── bindings.cpp
```


---

## 2. 环境搭建

### 2.1 系统要求

- **操作系统**：Windows 10/11, Linux, macOS
- **Python**：3.10 或以上
- **内存**：建议 8 GB 以上
- **处理器**：4 核以上（自弈并行加速）
- **GPU（可选）**：CUDA 11.8+ 兼容显卡（无 GPU 时自动降级至 CPU 模式）

### 2.2 安装 Python 依赖

```bash
# 克隆项目
git clone <repo_url>
cd gomoku_alphazero

# 创建虚拟环境（推荐）
python -m venv venv

# Windows 激活
venv\Scripts\activate

# Linux/macOS 激活
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

**requirements.txt 主要依赖：**

```text
torch>=2.0.0          # 神经网络训练与推理
fastapi>=0.104.0      # Web API 框架
uvicorn[standard]     # ASGI 服务器
numpy>=1.24.0         # 数值计算
pybind11>=2.11.0      # C++ 绑定（Phase 2）
pyyaml>=6.0           # 配置文件解析
```

### 2.3 编译 C++ 加速模块（Phase 2，可选）

**Windows（Visual Studio）：**

```bat
:: 需要 Visual Studio 2019 或以上，CMake 3.18+
scripts\build.bat
```

**Linux / macOS：**

```bash
bash scripts/build.sh
```

编译成功后，`gomoku/mcts.py` 会自动检测并切换为 C++ 引擎：

```python
# gomoku/mcts.py 中的自动降级逻辑
try:
    import gomoku_cpp  # C++ 编译产物
    USE_CPP_ENGINE = True
except ImportError:
    USE_CPP_ENGINE = False  # 降级到纯 Python 实现
```

---

## 3. 快速开始

### 3.1 完整训练流水线（推荐）

```bash
# 一键运行三阶段流水线（Windows PowerShell）
.\full_train_pipeline.ps1

# 或分步执行：

# Step 1：生成规则数据（约 5000 局，~15分钟）
python scripts/gen_rule_data.py --games 5000 --workers 4

# Step 2：监督预训练（约 80 个 epoch）
python pretrain.py --epochs 80

# Step 3：AlphaZero 自弈强化
python train.py --fresh
```

### 3.2 仅自弈训练（已有预训练模型时）

```bash
# 从 current_policy.pth 出发继续自弈
python train.py

# 指定参数
python train.py --games 5000 --playout 400 --workers 4

# 快速验证流程（小规模）
python train.py --games 50 --playout 100
```

### 3.3 启动 Web 对战服务

```bash
python server.py
```

启动后访问 `http://127.0.0.1:8000`，选择难度与执黑/执白后即可对弈。

### 3.4 Web 界面操作

1. **选择执黑/执白**：黑子先手，白子后手
2. **选择难度**：简单 / 中等 / 困难（影响 AI MCTS 模拟次数）
3. **点击棋盘**：在交叉点落子
4. **AI 热图**：勾选"显示 AI 分析"可查看 AI 评估热图
5. **重新开始**：点击"新游戏"重置棋盘

---

## 4. 强化学习算法详解

### 4.1 算法选型：为什么是 AlphaZero？

五子棋是**完美信息、确定性、零和、双人**博弈，这类问题具有以下特点：

| 特点 | 说明 |
|------|------|
| 稀疏奖励 | 只有终局才有 +1/-1 奖励，中间步骤无即时奖励 |
| 状态空间巨大 | 15×15 棋盘约 $3^{225}$ 种状态，无法枚举 |
| 完美信息 | 双方均可看到完整棋盘，无随机隐信息 |

相比其他 RL 算法：

| 算法 | 适用场景 | 五子棋适用性 |
|------|----------|------------|
| DQN | 单智能体、离散动作 | ❌ 不适合双人博弈 |
| PPO/A3C | 连续控制、稀疏奖励 | ⚠️ 可用但效率低 |
| **AlphaZero** | 完美信息双人博弈 | ✅ 专为此类问题设计 |

### 4.2 AlphaZero 核心思想

AlphaZero 的核心是将**深度神经网络**与**蒙特卡洛树搜索（MCTS）**结合成一个闭环系统：

```
┌─────────────────────────────────────────────────────┐
│                   AlphaZero 循环                      │
│                                                       │
│  ┌──────────┐    自弈数据     ┌──────────────────┐   │
│  │ 神经网络  │─────────────→  │   MCTS 增强搜索   │   │
│  │ f(s;θ)   │                │ π = MCTS(s, f)    │   │
│  │  ↓  ↓   │ ←─────────────  │ 落子选择 a~π      │   │
│  │ p    v   │   更新参数 θ    └──────────────────┘   │
│  └──────────┘                                        │
│       ↑                                              │
│   最小化损失: L = L_policy + L_value + L_reg          │
└─────────────────────────────────────────────────────┘
```


- **$f(s; \theta) = (p, v)$**：神经网络接收棋盘状态 $s$，输出：
  - $p$：落子概率向量（策略头），形状 `[board_size²]`
  - $v$：当前局面对当前玩家的价值估计 $v \in [-1, 1]$（价值头）

- **MCTS**：使用 $p$ 引导搜索树扩展，搜索完毕后输出改进的策略 $\pi$

- **训练目标**：使神经网络的输出逼近 MCTS 搜索给出的更优策略：
  $$\mathcal{L}(\theta) = \underbrace{-\sum_a \pi(a) \log p(a)}_{\text{策略损失（交叉熵）}} + \underbrace{(v - z)^2}_{\text{价值损失（MSE）}} + \underbrace{\lambda \|\theta\|^2}_{\text{L2 正则化}}$$

  其中 $z \in \{-1, 0, +1\}$ 为该局对弈的最终结果。

### 4.3 自我博弈数据生成

```
自弈一局的完整流程：

初始空棋盘 s₀
    │
    ▼
for t = 0, 1, 2, ... :
    │
    ├─ 运行 MCTS（n_playout 次模拟）
    │   得到访问计数 N(s,a)
    │
    ├─ 计算改进策略 πₜ = N(s,aᵢ)^(1/τ) / Σ N(s,aⱼ)^(1/τ)
    │   τ: 温度参数（早期=1.0，后期→0 强化最优选择）
    │
    ├─ 按 πₜ 采样落子动作 aₜ
    │
    └─ 保存 (sₜ, πₜ, ?) 到临时缓冲区
           ↓ 游戏结束时填入最终结果 z
           ↓ 玩家1赢 → z=+1，玩家2赢 → z=-1，平局 → z=0

一局结束：填充 z，将所有 (sₜ, πₜ, zₜ) 送入 ReplayBuffer
```

### 4.4 温度参数 τ 的作用

温度参数控制**探索与利用的平衡**：

$$\pi(a) \propto N(s, a)^{1/\tau}$$

| 温度 τ | 行为 |
|--------|------|
| τ = 1.0 | 按访问计数比例采样，高探索性 |
| τ → 0 | 确定性选择访问次数最多的动作，高利用性 |
| τ → ∞ | 均匀随机选择，完全探索 |

**本项目策略**：前 `temp_threshold`（默认前 15 步）使用 τ=1.0 促进多样性，之后 τ→0 选择最优着法。

### 4.5 Dirichlet 噪声增强探索

自弈时，在根节点的先验概率上叠加 Dirichlet 噪声，防止 AI 陷入固定套路：

$$p(a) = (1 - \epsilon) \cdot p_{\text{net}}(a) + \epsilon \cdot \eta_a, \quad \eta \sim \text{Dir}(\alpha)$$

- $\epsilon = 0.25$（噪声比例）
- $\alpha = 0.3$（Dirichlet 浓度，棋盘越大用越小的值）

---

## 5. 游戏环境详解

### 5.1 棋盘状态表示

棋盘使用 **4 通道特征平面** 表示，形状为 `[4, board_size, board_size]`：

| 通道 | 内容 | 取值 |
|------|------|------|
| 0 | 当前玩家的棋子位置 | 0 或 1 |
| 1 | 对手的棋子位置 | 0 或 1 |
| 2 | 上一步落子位置 | 0 或 1（仅该位为 1）|
| 3 | 当前玩家指示符 | 全 1（黑）或 全 0（白）|

**示例（8×8 棋盘，黑子在 (3,3)，白子在 (3,4)，上一步为白子落在 (3,4)）：**

```
通道 0（黑子）：    通道 1（白子）：    通道 3（当前玩家=白）：
0 0 0 0 0 0 0 0   0 0 0 0 0 0 0 0   0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0   0 0 0 0 0 0 0 0   0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0   0 0 0 0 0 0 0 0   0 0 0 0 0 0 0 0
0 0 0 1 0 0 0 0   0 0 0 0 1 0 0 0   0 0 0 0 0 0 0 0
...               ...               ...
```

### 5.2 状态转移与奖励

```python
# game.py 核心接口

class GomokuGame:
    def reset(self) -> np.ndarray:
        """重置棋盘，返回初始状态（4通道特征平面）"""

    def step(self, action: int) -> tuple[np.ndarray, float, bool]:
        """
        执行落子动作
        action: 落子位置索引，范围 [0, board_size²-1]
               对应棋盘坐标 (row, col) = (action // board_size, action % board_size)

        Returns:
            next_state: 新的 4 通道特征平面
            reward: 0.0（未结束）, +1.0（当前玩家赢）, -1.0（当前玩家输）
            done: 游戏是否结束
        """

    def get_legal_moves(self) -> np.ndarray:
        """返回当前所有合法落子位置的布尔数组，形状 [board_size²]"""

    def get_canonical_state(self) -> np.ndarray:
        """
        返回从当前玩家视角的规范化状态。
        无论黑白方，始终以"当前玩家"为第一人称，
        保证神经网络输入的一致性。
        """
```

### 5.3 胜负判断逻辑

五子棋胜负判断通过扫描四个方向实现（水平、垂直、左斜、右斜）：

```
检查最新落子点 (r, c) 的四个方向：
  → 水平：(0,  1) 方向
  ↓ 垂直：(1,  0) 方向
  ↘ 右斜：(1,  1) 方向
  ↗ 左斜：(1, -1) 方向

对每个方向，向两侧延伸，统计连续同色棋子数：
  count = 1（当前点）
  向正方向延伸：count += 连续同色子数
  向负方向延伸：count += 连续同色子数
  if count >= 5: 判定胜利
```

### 5.4 8 种对称数据扩充

棋盘具有 **二面体群 D₄** 对称性（4 次旋转 × 2 次翻转 = 8 种变换），每条训练样本可扩充为 8 条：

```
原始棋盘       旋转 90°      旋转 180°     旋转 270°
┌─────┐       ┌─────┐       ┌─────┐       ┌─────┐
│ ·   │  →    │     │  →    │   · │  →    │     │
│  ●  │       │  ●  │       │  ●  │       │  ●  │
│   · │       │ ·   │       │ ·   │       │   · │
└─────┘       └─────┘       └─────┘       └─────┘
    ↓ 水平翻转（各自再生成4种）
┌─────┐       ...
│   · │
│  ●  │
│ ·   │
└─────┘
```

```python
# replay_buffer.py 中的对称扩充实现
def augment(state, policy, value):
    """对单条 (s, π, z) 样本做 8 种对称变换，返回 8 条样本"""
    samples = []
    board_size = int(np.sqrt(len(policy)))
    policy_2d = policy.reshape(board_size, board_size)

    for k in range(4):                    # 旋转 0°, 90°, 180°, 270°
        rot_s = np.rot90(state, k, axes=(1, 2))
        rot_p = np.rot90(policy_2d, k).flatten()
        samples.append((rot_s, rot_p, value))

        flip_s = np.flip(rot_s, axis=2)   # 水平翻转
        flip_p = np.fliplr(rot_p.reshape(board_size, board_size)).flatten()
        samples.append((flip_s, flip_p, value))

    return samples  # 返回 8 条样本
```

---

## 6. 神经网络架构

### 6.1 整体架构

```
输入: [batch, 4, 8, 8]
    │
    ▼
┌─────────────────────────────────┐
│ 初始卷积块                        │
│  Conv2d(4→64, 3×3, pad=1)       │
│  BatchNorm2d(64)                 │
│  ReLU                            │
└─────────────┬───────────────────┘
              │
              ▼ × 5
┌─────────────────────────────────┐
│ 残差块 (ResBlock)                 │
│  Conv2d(64→64, 3×3, pad=1)      │
│  BatchNorm2d(64) → ReLU          │
│  Conv2d(64→64, 3×3, pad=1)      │
│  BatchNorm2d(64)                 │
│  + 跳跃连接 → ReLU               │
└─────────────┬───────────────────┘
              │
       ┌──────┴──────┐
       ▼             ▼
┌────────────┐  ┌────────────────────┐
│  策略头     │  │  价值头             │
│ Conv(64→2) │  │ Conv(64→1)         │
│ BN → ReLU  │  │ BN → ReLU          │
│ Flatten    │  │ Flatten            │
│ Linear(    │  │ Linear(64→64)      │
│  128→64)   │  │ ReLU               │
│ Linear(    │  │ Linear(64→1)       │
│  64→64)    │  │ tanh               │
│ log_softmax│  └─────────┬──────────┘
└─────┬──────┘            │
      │                   │
      ▼                   ▼
  p: [batch, 64]      v: [batch, 1]
  落子概率对数          局面价值 ∈ [-1,1]
```

### 6.2 残差块设计

残差连接（Residual Connection）解决深度网络中的梯度消失问题：

$$\text{ResBlock}(x) = \text{ReLU}(F(x) + x)$$

$$F(x) = \text{BN}(\text{Conv}(\text{ReLU}(\text{BN}(\text{Conv}(x)))))$$

```python
# neural_net.py
class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)   # 跳跃连接
```

### 6.3 CPU 与 GPU 配置对比

| 参数 | CPU 模式 | GPU 模式 |
|------|----------|----------|
| 卷积滤波器数 | 64 | 128 |
| 残差块数 | 5 | 10 |
| 批次大小 | 256 | 512 |
| 网络参数量 | ~1.2M | ~8.5M |
| 单次推理耗时（8×8）| ~2ms | ~0.3ms |

---

## 7. MCTS 搜索引擎

### 7.1 MCTS 完整流程

每次模拟（Simulation）包含四个阶段：

```
           根节点 s₀
               │
     ┌─────────▼──────────┐
     │    1. 选择 SELECT   │
     │  按 PUCT 公式选择    │
     │  最大得分的子节点    │
     └─────────┬──────────┘
               │ （重复直到叶节点）
     ┌─────────▼──────────┐
     │    2. 扩展 EXPAND   │
     │  叶节点调用神经网络  │
     │  获取 (p, v)        │
     │  为所有合法动作创建  │
     │  子节点             │
     └─────────┬──────────┘
               │
     ┌─────────▼──────────┐
     │   3. 评估 EVALUATE  │
     │  使用神经网络输出 v  │
     │  作为局面价值        │
     └─────────┬──────────┘
               │
     ┌─────────▼──────────┐
     │  4. 反传 BACKPROP   │
     │  将 v 沿路径传回     │
     │  更新 N(s,a), W(s,a)│
     └─────────────────────┘
```

### 7.2 PUCT 选择公式

PUCT（Predictor + Upper Confidence bound for Trees）公式平衡**利用**（高胜率）与**探索**（少访问的节点）：

$$a^* = \arg\max_a \left[ Q(s, a) + c_{\text{puct}} \cdot P(s, a) \cdot \frac{\sqrt{N(s)}}{1 + N(s, a)} \right]$$

| 符号 | 含义 |
|------|------|
| $Q(s, a)$ | 动作 $a$ 的平均价值：$W(s,a) / N(s,a)$ |
| $P(s, a)$ | 神经网络给出的先验概率 |
| $N(s)$ | 父节点的总访问次数 |
| $N(s, a)$ | 动作 $a$ 的访问次数 |
| $c_{\text{puct}}$ | 探索常数（默认 5.0），控制探索力度 |

**直觉解释：**
- $Q(s,a)$ 项：**利用**，优先选择历史上胜率高的动作
- 第二项：**探索**，分子 $\sqrt{N(s)}$ 随总访问增长，$1+N(s,a)$ 使访问少的节点得分更高；$P(s,a)$ 用神经网络先验引导探索方向

### 7.3 节点数据结构

```python
# mcts.py
class MCTSNode:
    def __init__(self, parent, prior_p: float):
        self.parent   = parent
        self.children = {}       # dict[action → MCTSNode]
        self.n_visits = 0        # N(s,a)：访问次数
        self.q_value  = 0.0      # Q(s,a)：平均价值
        self.u_value  = 0.0      # U(s,a)：置信上界
        self.prior_p  = prior_p  # P(s,a)：先验概率

    def get_puct_score(self, c_puct: float, parent_n: int) -> float:
        """计算 PUCT 得分 = Q + c_puct * P * sqrt(N_parent) / (1 + N)"""
        self.u_value = (c_puct * self.prior_p *
                        np.sqrt(parent_n) / (1 + self.n_visits))
        return self.q_value + self.u_value

    def update(self, value: float):
        """反向传播更新：N += 1, Q = W/N"""
        self.n_visits += 1
        self.q_value  += (value - self.q_value) / self.n_visits
```

### 7.4 Virtual Loss（C++ 并行 MCTS，Phase 2）

多线程并行 MCTS 时，不同线程可能选择相同路径（因为它们读到的是相同的 N 值），导致重复模拟，降低搜索效率。**Virtual Loss** 通过临时增加节点的访问计数来引导不同线程探索不同路径：

```
线程1 到达节点 A：N(A) += 3（Virtual Loss=3），Q(A) 临时降低
      → PUCT 得分下降，线程2 被引导到其他分支

模拟完成后：N(A) -= 3（撤销 Virtual Loss）
             N(A) += 1（真实更新）
             W(A) += 真实 value
```

---

## 8. 训练流水线

### 8.1 三阶段流水线

#### Phase 1 — 规则数据生成

```bash
python scripts/gen_rule_data.py --games 5000 --workers 4
# 输出: models/human_games/rule_data/game_*.npz
```

规则玩家基于威胁评估（五连、活四、冲四、活三等）进行对弈，并加入少量噪声保证多样性。每局保存 `(状态, 访问次数归一化概率, 终局结果)` 三元组。

#### Phase 2 — 监督预训练

```bash
python pretrain.py --epochs 80
# 读取 config.RULE_DATA_DIR，写出 models/current_policy.pth
```

对规则数据做普通交叉熵（策略头）+ MSE（价值头）监督学习。预训练后网络已能识别基本棋型，MCTS 探索起点大幅提高。

#### Phase 3 — AlphaZero 自弈强化

```bash
python train.py --fresh
```

标准 AlphaZero 循环：多进程并行自弈 → 数据写入回放池 → mini-batch 梯度更新 → 定期评估保存最优权重。

**混合回放**：每个 batch 中约 15%（`MIX_EXPERT_REPLAY_RATIO=0.15`）采样自规则预训练数据，防止网络忘掉基础棋型知识。

### 8.2 Coach 训练主循环

```
for iteration = 1 to N_SELFPLAY_GAMES / GAMES_PER_WORKER:
    │
    ├─ 1. 多进程自弈
    │      Worker×N → 各自弈 GAMES_PER_WORKER 局
    │      收集 (s, π, z) + 8 折对称增强 → 写入 ReplayBuffer
    │
    ├─ 2. 网络训练（EPOCHS_PER_UPDATE 轮）
    │      batch = 自弈样本(85%) + 规则样本(15%)
    │      L = L_ce(policy) + 0.5×L_mse(value) + L2
    │
    └─ 3. 评估（每 CHECK_FREQ 轮）
           新模型 vs 纯 MCTS(N_PLAYOUT_EVAL)
           胜率创历史新高 → 保存 best_policy.pth
```

### 8.3 多进程自弈架构

```
主进程（Coach）
    │
    ├─ 序列化模型权重（pickle bytes）分发给 Worker
    │
    ├─ Worker 进程 1 ──→ 自弈 GAMES_PER_WORKER 局 ──→ 返回样本列表
    ├─ Worker 进程 2 ──→ 自弈 GAMES_PER_WORKER 局 ──→ 返回样本列表
    └─ ...
           │
    主进程汇总后写入 ReplayBuffer
```

> **Windows 注意**：spawn 多进程要求 worker 函数（`_selfplay_worker`）为顶层函数，`train.py` 已正确设置 `set_start_method('spawn')`。

### 8.4 ReplayBuffer 设计

- 固定容量 FIFO 队列（`deque(maxlen=BUFFER_SIZE)`）
- 每条原始样本在加入时做 8 折对称增强（4 旋转 × 2 翻转）
- 支持 `RECENCY_SAMPLE_ALPHA > 0` 的几何偏重采样（优先新样本）

---

## 9. Web 对战界面

### 9.1 API 接口说明

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/api/new_game` | 创建新游戏会话 |
| `GET` | `/api/state/{sid}` | 获取当前棋盘状态 |
| `POST` | `/api/human_move/{sid}` | 人类玩家落子 |
| `POST` | `/api/ai_move/{sid}` | AI 落子（MCTS 计算）|
| `GET` | `/api/hint/{sid}` | AI 落子概率热图 |
| `POST` | `/api/resign/{sid}` | 认输 |
| `GET` | `/api/difficulty_config` | 当前难度配置 |
| `GET` | `/api/model/checkpoints` | 可切换的权重列表 |
| `POST` | `/api/model/set_play_path` | 热切换对战模型 |
| `POST` | `/api/reload_model` | 热重载模型权重 |
| `POST` | `/api/train/start` | 后台启动训练 |
| `GET` | `/api/train/status` | 查询训练状态 |
| `GET` | `/api/train/log` | 获取训练日志 |
| `POST` | `/api/train/stop` | 停止训练 |

**创建新游戏请求体：**

```json
{
  "board_size": 8,
  "human_color": "black",       // "black" 或 "white"
  "difficulty": "medium"         // "easy"(200) | "medium"(400) | "hard"(800)
}
```

**落子请求体：**

```json
{
  "row": 3,
  "col": 4
}
```

**AI 落子响应体：**

```json
{
  "action": 28,                  // 落子索引
  "row": 3,
  "col": 4,
  "think_time_ms": 1240,         // AI 思考时间（毫秒）
  "win_rate": 0.68,              // AI 对当前局面的胜率估计
  "game_over": false,
  "winner": null                 // "black" | "white" | "draw" | null
}
```

### 9.2 前端架构

```
index.html
├── board.js（Canvas 渲染层）
│   ├── drawBoard()         - 绘制棋盘格线和星位
│   ├── drawStones()        - 绘制棋子（含落子动画）
│   ├── drawLastMove()      - 高亮最后一手
│   ├── drawHeatmap()       - AI 置信度热图（透明度叠加）
│   └── handleClick(event)  - 坐标转换 → 触发落子
│
└── game.js（交互逻辑层）
    ├── newGame()           - 创建会话，初始化界面
    ├── humanMove(row,col)  - 发送落子请求，更新棋盘
    ├── requestAiMove()     - 轮询 AI 落子（显示"思考中..."动画）
    ├── updateStatus()      - 更新状态栏（当前玩家、胜率）
    └── handleGameOver()    - 显示胜负结果弹窗
```

---

## 10. C++ 加速模块（Phase 2）

### 10.1 性能对比

| 操作 | Python 实现 | C++ 实现 | 加速比 |
|------|-------------|----------|--------|
| 单次 MCTS 模拟（400次）| ~800ms | ~80ms | **10×** |
| 胜负判断 | ~0.05ms | ~0.002ms | 25× |
| 自弈 100 局 | ~45min | ~5min | 9× |

### 10.2 C++ 模块结构

**位棋盘（Bitboard）表示：**

```cpp
// src/game/gomoku.h
// 8×8 棋盘可用 64 位整数表示，位运算极速判断五连
class GomokuGame {
    uint64_t black_board_;  // 黑子位置（每bit对应一个格子）
    uint64_t white_board_;  // 白子位置
    int      current_player_;

    // 五连检测：水平/垂直/斜线 4个方向的位移掩码预计算
    static constexpr uint64_t H_MASK = 0x7F7F7F7F7F7F7F7FULL;  // 水平方向
};
```

**并行 MCTS 核心：**

```cpp
// src/mcts/mcts.h
class ParallelMCTS {
    std::vector<std::thread> workers_;   // 工作线程池
    std::atomic<int>         sim_count_; // 已完成模拟数
    PyInferenceCallback      nn_eval_;   // 回调 Python NN 推理

    // Virtual Loss 实现
    void apply_virtual_loss(MCTSNode* node, int vl = 3);
    void revert_virtual_loss(MCTSNode* node, int vl = 3);
};
```

### 10.3 pybind11 接口

```cpp
// src/bindings.cpp
PYBIND11_MODULE(gomoku_cpp, m) {
    py::class_<GomokuGame>(m, "GomokuGame")
        .def(py::init<int>())
        .def("reset",           &GomokuGame::reset)
        .def("step",            &GomokuGame::step)
        .def("get_state",       &GomokuGame::get_state_tensor)  // 返回 numpy array
        .def("get_legal_moves", &GomokuGame::get_legal_moves);

    py::class_<ParallelMCTS>(m, "ParallelMCTS")
        .def(py::init<int, int, float>())  // board_size, n_threads, c_puct
        .def("set_nn_callback",  &ParallelMCTS::set_nn_callback)
        .def("get_action_probs", &ParallelMCTS::get_action_probs);
}
```

---

## 11. 配置参数说明

所有超参数集中在 `gomoku/config.py`：

```python
# gomoku/config.py

class Config:
    # ── 棋盘设置 ────────────────────────
    BOARD_SIZE    : int   = 8      # 棋盘大小（8×8）
    N_IN_ROW      : int   = 5      # 几子连珠获胜

    # ── 神经网络（CPU 模式自动降级）──────
    NUM_RES_BLOCKS: int   = 5      # 残差块数量
    NUM_FILTERS   : int   = 64     # 卷积滤波器数（GPU 模式 128）
    BATCH_SIZE    : int   = 256    # 训练批次大小（GPU 模式 512）
    LEARNING_RATE : float = 1e-3   # Adam 初始学习率
    L2_REG        : float = 1e-4   # L2 正则化系数
    TARGET_KL     : float = 0.02   # KL 散度目标（自适应学习率）

    # ── MCTS 搜索 ────────────────────────
    N_PLAYOUT     : int   = 400    # 每步模拟次数（简单=200，困难=800）
    C_PUCT        : float = 5.0    # PUCT 探索常数
    TEMP_THRESHOLD: int   = 15     # 前几步使用高温采样
    DIRICHLET_EPS : float = 0.25   # Dirichlet 噪声比例
    DIRICHLET_ALPHA: float = 0.3   # Dirichlet 浓度参数

    # ── 训练 ────────────────────────────
    MAX_ITERATIONS: int   = 1500   # 总训练轮数
    SELFPLAY_GAMES: int   = 15     # 每轮自弈局数
    N_WORKERS     : int   = 8      # 并行自弈进程数
    BUFFER_SIZE   : int   = 10000  # 经验回放池容量
    UPDATE_EPOCHS : int   = 5      # 每轮训练 epoch 数

    # ── 评估 ────────────────────────────
    EVAL_FREQ     : int   = 50     # 每隔多少轮评估一次
    EVAL_GAMES    : int   = 10     # 评估对局场数
    WIN_RATE_THRESH: float = 0.55  # 保存新模型的胜率阈值
    PURE_MCTS_SIM : int   = 500    # 纯 MCTS 基准的模拟次数
```

---

## 12. 性能基准

测试环境：Intel Core i7-12700（12 核），无 GPU，Python 3.11，PyTorch 2.1（CPU）

| 指标 | 数值 |
|------|------|
| 单局自弈时间（400 次模拟）| ~3 分钟 |
| 15 进程并行自弈吞吐 | ~5 局/分钟 |
| 每轮训练时间（含自弈+训练）| ~15 分钟 |
| 达到"可玩强度"（vs 新手）所需时间 | ~4 小时（~200 轮） |
| 开局数据库覆盖（200 轮后）| ~800 局 |
| Web AI 响应时间（中等难度）| ~2-3 秒 |

---

## 13. 常见问题

### Q1：训练时 ValueError: Not enough samples in replay buffer

**原因**：ReplayBuffer 中样本数不足一个 batch，通常发生在训练初期。  
**解决**：系统会自动等待 buffer 积累到 `BATCH_SIZE` 以上再开始训练，无需手动干预。正常情况下经过 3-5 轮自弈即可开始训练。

---

### Q2：Web 界面 AI 一直显示"思考中..."

**原因**：服务器端 MCTS 搜索耗时过长（模型文件不存在或模型被 CPU 满载）。  
**解决**：
1. 确认 `models/best_policy.pth` 存在（至少训练一次后才生成）
2. 降低难度为"简单"（200 次模拟）
3. 检查终端是否有报错信息

---

### Q3：Windows 下训练时出现 RuntimeError: An attempt has been made to start a new process

**原因**：Windows 多进程 `spawn` 模式要求保护主入口。  
**解决**：确保 `train.py` 中有 `if __name__ == '__main__':` 保护，当前版本已包含此处理。

---

### Q4：如何迁移到 GPU 训练？

无需修改代码，确保安装了 CUDA 版 PyTorch 后，系统会自动检测并使用 GPU：

```bash
# 卸载 CPU 版 PyTorch
pip uninstall torch

# 安装 CUDA 12.1 版本（根据实际 CUDA 版本选择）
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 运行训练，会自动切换到 GPU 模式（滤波器数量自动升至 128）
python train.py
```

---

### Q5：如何调整棋盘大小到 15×15？

修改 `gomoku/config.py`：

```python
BOARD_SIZE = 15
N_IN_ROW   = 5
```

> ⚠️ **注意**：15×15 棋盘的状态空间远大于 8×8，建议同时将 `NUM_FILTERS` 提升至 128，`N_PLAYOUT` 提升至 800，并使用 GPU 训练，否则训练时间可能超过数天。

---

## 14. 参考资料

| 论文/资源 | 说明 |
|-----------|------|
| [Silver et al., 2017 - Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](https://arxiv.org/abs/1712.01815) | AlphaZero 原论文 |
| [Silver et al., 2016 - Mastering the game of Go with deep neural networks and tree search](https://www.nature.com/articles/nature16961) | AlphaGo Nature 论文 |
| [Rosin, 2011 - Multi-armed bandits with episode context](https://link.springer.com/article/10.1007/s10472-011-9258-6) | PUCT 公式来源 |
| [junxiaosong/AlphaZero_Gomoku](https://github.com/junxiaosong/AlphaZero_Gomoku) | 五子棋 AlphaZero 参考实现 |
| [PyTorch 官方文档](https://pytorch.org/docs/) | PyTorch API 参考 |
| [pybind11 官方文档](https://pybind11.readthedocs.io/) | Python/C++ 绑定参考 |

---

## 版本历史

| 版本 | 日期 | 主要变化 |
|------|------|---------|
| v1.0 | 2026-02-28 | 初始版本：纯 Python MCTS + PyTorch ResNet + FastAPI Web |
| v1.1 | 计划中 | Phase 2：C++ MCTS 引擎 + Virtual Loss 并行自弈 |
| v1.2 | 计划中 | 15×15 完整棋盘支持 + 开局数据库 |