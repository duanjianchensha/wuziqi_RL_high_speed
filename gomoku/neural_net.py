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
        policy_logits = self.policy_head(feat)  # (B, N*N)

        # 提取合法落子掩码：x[:, 0] 是己方，x[:, 1] 是对方
        occupied = (x[:, 0] + x[:, 1]).view(x.size(0), -1)  # (B, N*N)
        # 非法落子位置的 logit 设为较小的负数（不要使用 -1e8 避免在 FP16 下溢出报错 RuntimeError）
        policy_logits = policy_logits.masked_fill(occupied > 0.5, -1e4)

        logp = F.log_softmax(policy_logits, dim=1)  # (B, N*N)
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
        # AdamW 正确解耦 L2 正则化方向与梯度自适应方向（比 Adam+weight_decay 更规范）
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.LR,
            weight_decay=config.L2_CONST,
        )
        # GPU 混合精度；CPU 上不使用
        self._use_amp = config.DEVICE == "cuda"
        self.scaler = GradScaler(enabled=self._use_amp)

        if self._use_amp:
            torch.backends.cudnn.benchmark = True  # 开启 CuDNN 底层算法自动寻优

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
        # 注意：不要在每次推理都调用 self.model.eval()，避免高频函数调用开销
        if self.model.training:
            self.model.eval()

        state = board.get_current_state()  # (4, N, N)
        state_tensor = (
            torch.from_numpy(state).unsqueeze(0).to(self.device, non_blocking=True)
        )  # (1,4,N,N)

        with torch.no_grad():
            # bs=1 时取消 autocast，因为 FP16 在此处 Python API 的转换开销远大于性能收益（实测提速 2 倍）
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
        p = torch.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
        p = torch.clamp(p, min=0.0)
        row_sum = p.sum(dim=1, keepdim=True)
        p = torch.where(row_sum > 0, p / row_sum, torch.full_like(p, 1.0 / p.shape[1]))
        z = torch.from_numpy(winners).to(self.device)

        self.optimizer.zero_grad()
        with autocast(device_type=config.DEVICE, enabled=self._use_amp):
            logp, v = self.model(s)
            # 价值损失（权重 0.5：与 AlphaZero 论文一致，避免 value loss 在大棋盘上压制策略学习）
            loss_v = F.mse_loss(v, z)
            # 策略损失（交叉熵）
            loss_p = -torch.mean(torch.sum(p * logp, dim=1))
            loss = 0.5 * loss_v + loss_p

        # 训练前向完成后立即保存 logp（梯度更新前的策略输出），
        # 避免 backward+step 后再做一次完整 eval 前向（节省约 50% 前向计算）
        logp_before = logp.detach()

        self.scaler.scale(loss).backward()
        # 梯度裁剪：防止训练早期或崩溃后梯度爆炸导致 NaN loss。
        # 必须在 unscale_ 之后、step() 之前执行。
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=config.GRAD_CLIP
        )
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # 用保存的训练前向输出计算 KL，无需额外 eval 前向传播
        with torch.no_grad():
            kl = torch.mean(
                torch.sum(p * (torch.log(p + 1e-10) - logp_before), dim=1)
            ).item()

        return {
            "loss": loss.item(),
            "loss_v": loss_v.item(),
            "loss_p": loss_p.item(),
            "kl": kl,
        }

    # ── 持久化 ────────────────────────────────────
    def save(self, path: str) -> None:
        import time

        dirpath = os.path.dirname(path) or "."
        os.makedirs(dirpath, exist_ok=True)

        if os.path.isdir(path):
            raise RuntimeError(f"保存失败：目标路径是目录而非文件: {path}")

        payload = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

        last_err = None
        for _ in range(3):
            tmp_path = f"{path}.tmp"
            try:
                torch.save(payload, tmp_path)
                os.replace(tmp_path, path)  # 原子替换
                return
            except Exception as e:
                last_err = e
                try:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                except Exception:
                    pass
                time.sleep(0.2)

        raise RuntimeError(f"模型保存失败: {path} | 原因: {last_err}")

    def reset_optimizer(self) -> None:
        """
        丢弃 Adam/GradScaler 状态，重新创建优化器。
        人机预训练若从自弈保存的权重 warm-start，checkpoint 里的 optimizer 二阶矩往往极大，
        导致有效学习率接近 0、loss 多 epoch 几乎不变。
        """
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.LR,
            weight_decay=config.L2_CONST,
        )
        self.scaler = GradScaler(enabled=self._use_amp)

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
