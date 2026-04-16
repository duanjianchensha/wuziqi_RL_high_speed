"""
gomoku/data_utils.py
通用 npz 数据加载工具（用于规则预训练数据与混合回放）。
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from gomoku.config import config


def iter_npz_files(root: str) -> List[str]:
    """递归收集 root 目录下全部 game_*.npz，按路径排序。"""
    out: List[str] = []
    if not os.path.isdir(root):
        return out
    for dirpath, _dirs, files in os.walk(root):
        for fn in files:
            if fn.startswith("game_") and fn.endswith(".npz"):
                out.append(os.path.join(dirpath, fn))
    return sorted(out)


def load_npz_file(path: str) -> Optional[Dict]:
    """加载单个 npz 文件，返回 dict；失败时返回 None。"""
    try:
        d = np.load(path, allow_pickle=False)
    except Exception:
        return None
    return {
        "states":       d["states"],
        "mcts_probs":   d["mcts_probs"],
        "winners":      d["winners"],
        "board_size":   int(d["board_size"]),
    }


def load_npz_files(data_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    递归加载 data_dir 下所有 game_*.npz，拼接后返回 (states, probs, zs)。
    任何一个通道为空时返回形状正确的空数组。
    """
    bs = config.BOARD_SIZE
    empty = (
        np.zeros((0, 4, bs, bs), np.float32),
        np.zeros((0, bs * bs),   np.float32),
        np.zeros((0,),            np.float32),
    )
    files = iter_npz_files(data_dir)
    if not files:
        return empty
    all_s, all_p, all_z = [], [], []
    for path in files:
        item = load_npz_file(path)
        if item is None:
            continue
        all_s.append(item["states"])
        all_p.append(item["mcts_probs"])
        all_z.append(item["winners"])
    if not all_s:
        return empty
    return (
        np.concatenate(all_s, axis=0).astype(np.float32),
        np.concatenate(all_p, axis=0).astype(np.float32),
        np.concatenate(all_z, axis=0).astype(np.float32),
    )


def list_checkpoints(limit: int = 50) -> List[Dict]:
    """
    扫描 models/checkpoints 目录，返回按修改时间倒序的 .pth 列表。
    每项: {"name": str, "path": str, "mtime": float}
    """
    d = config.CHECKPOINT_DIR
    if not os.path.isdir(d):
        return []
    out = []
    for fn in os.listdir(d):
        if not fn.endswith(".pth"):
            continue
        p = os.path.join(d, fn)
        try:
            st = os.stat(p)
            out.append({
                "name":  fn,
                "path":  os.path.join("models", "checkpoints", fn).replace("\\", "/"),
                "mtime": st.st_mtime,
            })
        except OSError:
            pass
    out.sort(key=lambda x: x["mtime"], reverse=True)
    return out[:limit]
