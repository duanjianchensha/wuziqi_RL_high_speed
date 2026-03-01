"""
gomoku/neural_net.py
AlphaZero 残差策略-价值网络（PyTorch 实现）

架构：
  输入: (B, 4, N, N)  4通道特征平面
  ├─ 主干: 1 个卷积层 + N 个残差块 (N_FILTERS 通道)
  ├─ 策略头: 小卷积 → Flatten → Linear → Softmax → (B, N*N)
  └─ 价值头: 小卷积 → Flatten → Linear → Linear → Tanh  → (B, 1)
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler
from typing import Tuple, List

from gomoku.config import config


# ────────────────────────────────────────────────
# 辅助模块
# ────────────────────────────────────────────────
class ResBlock(nn.Module):
    """标准残差块：Conv-BN-ReLU-Conv-BN + shortcut。"""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


# ────────────────────────────────────────────────
# 策略-价值网络
# ────────────────────────────────────────────────
class PolicyValueNet(nn.Module):
    """
    共享主干的双头网络。
    输入 shape: (B, 4, size, size)
    策略输出:   (B, size*size)  log-softmax 概率
    价值输出:   (B,)            tanh 标量
    """

    def __init__(
        self,
        board_size: int = config.BOARD_SIZE,
        n_filters: int = config.N_FILTERS,
        n_res_blocks: int = config.N_RES_BLOCKS,
    ):
        super().__init__()
        self.board_size = board_size
        n_squares = board_size * board_size

        # 主干：入口卷积 + 残差堆叠
        self.stem = nn.Sequential(
            nn.Conv2d(4, n_filters, 3, padding=1, bias=False),
            nn.BatchNorm2d(n_filters),
            nn.ReLU(inplace=True),
        )
        self.res_layers = nn.Sequential(
            *[ResBlock(n_filters) for _ in range(n_res_blocks)]
        )

        # 策略头
        self.policy_head = nn.Sequential(
            nn.Conv2d(n_filters, 2, 1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(2 * n_squares, n_squares),
        )

        # 价值头
        self.value_head = nn.Sequential(
            nn.Conv2d(n_filters, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(n_squares, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.res_layers(self.stem(x))
        logp = F.log_softmax(self.policy_head(feat), dim=1)  # (B, N*N)
        v = self.value_head(feat).squeeze(1)  # (B,)
        return logp, v


# ────────────────────────────────────────────────
# 网络包装器（含推理/训练接口）
# ────────────────────────────────────────────────
class PolicyValueFunction:
    """
    封装 PolicyValueNet，提供：
      • policy_value_fn(board) → (action_probs, value)  供 MCTS 使用
      • train_step(batch) → loss dict               供 Coach 使用
    """

    def __init__(self, board_size: int = config.BOARD_SIZE, model_path: str = None):
        self.device = torch.device(config.DEVICE)
        self.model = PolicyValueNet(board_size).to(self.device)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.LR,
            weight_decay=config.L2_CONST,
        )
        # GPU 混合精度；CPU 上不使用
        self._use_amp = config.DEVICE == "cuda"
        self.scaler = GradScaler(enabled=self._use_amp)

        os.makedirs(config.MODEL_DIR, exist_ok=True)
        os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

        if model_path and os.path.exists(model_path):
            self.load(model_path)
            print(f"[PolicyValueFunction] 加载模型: {model_path}")
        else:
            print(f"[PolicyValueFunction] 从随机初始化开始，设备={config.DEVICE}")

    # ── 推理接口 ──────────────────────────────────
    def policy_value_fn(self, board) -> Tuple[List[Tuple[int, float]], float]:
        """
        供 MCTS 调用。
        返回: ([(action, prob), ...], value)
        """
        self.model.eval()
        state = board.get_current_state()  # (4, N, N)
        state_tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)  # (1,4,N,N)
        with torch.no_grad():
            with autocast(device_type=config.DEVICE, enabled=self._use_amp):
                logp, v = self.model(state_tensor)
        probs = torch.exp(logp).squeeze(0).cpu().numpy()  # (N*N,)
        value = float(v.item())
        avail = board.availables
        action_probs = list(zip(avail, probs[avail]))
        return action_probs, value

    def policy_value_batch(self, states: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        批量推理，供并行 self-play 使用。
        states: (B, 4, N, N) float32
        返回: probs (B, N*N), values (B,)
        """
        self.model.eval()
        t = torch.from_numpy(states).to(self.device)
        with torch.no_grad():
            with autocast(device_type=config.DEVICE, enabled=self._use_amp):
                logp, v = self.model(t)
        return torch.exp(logp).cpu().numpy(), v.cpu().numpy()

    # ── 训练接口 ──────────────────────────────────
    def train_step(
        self,
        states: np.ndarray,  # (B, 4, N, N)
        mcts_probs: np.ndarray,  # (B, N*N)
        winners: np.ndarray,  # (B,)
        lr: float = None,
    ) -> dict:
        """
        执行一步 mini-batch 梯度更新。
        损失 = 价值 MSE + 策略交叉熵 − 熵正则化
        返回损失字典。
        """
        if lr is not None:
            for g in self.optimizer.param_groups:
                g["lr"] = lr

        self.model.train()
        s = torch.from_numpy(states).to(self.device)
        p = torch.from_numpy(mcts_probs).to(self.device)
        z = torch.from_numpy(winners).to(self.device)

        self.optimizer.zero_grad()
        with autocast(device_type=config.DEVICE, enabled=self._use_amp):
            logp, v = self.model(s)
            # 价值损失
            loss_v = F.mse_loss(v, z)
            # 策略损失（交叉熵）
            loss_p = -torch.mean(torch.sum(p * logp, dim=1))
            loss = loss_v + loss_p

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # 计算 MCTS prob 与旧策略 KL（用于自适应学习率）
        with torch.no_grad():
            self.model.eval()
            with autocast(device_type=config.DEVICE, enabled=self._use_amp):
                logp_new, _ = self.model(s)
            kl = torch.mean(
                torch.sum(p * (torch.log(p + 1e-10) - logp_new), dim=1)
            ).item()

        return {
            "loss": loss.item(),
            "loss_v": loss_v.item(),
            "loss_p": loss_p.item(),
            "kl": kl,
        }

    # ── 持久化 ────────────────────────────────────
    def save(self, path: str) -> None:
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            try:
                self.optimizer.load_state_dict(ckpt["optimizer"])
            except Exception:
                pass  # 超参变化时跳过 optimizer 状态

    def get_weights(self) -> bytes:
        """序列化模型权重为 bytes，用于跨进程传递。"""
        import io

        buf = io.BytesIO()
        torch.save(self.model.state_dict(), buf)
        return buf.getvalue()

    def set_weights(self, weights: bytes) -> None:
        """从 bytes 恢复模型权重。"""
        import io

        buf = io.BytesIO(weights)
        state = torch.load(buf, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
