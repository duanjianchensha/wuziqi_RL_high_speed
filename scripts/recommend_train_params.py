#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据本机 CPU / GPU 显存探测，给出与 gomoku/config.py 字段对齐的训练参数建议。

用法（项目根目录，conda 环境 ship）:
  python scripts/recommend_train_params.py
  python scripts/recommend_train_params.py --json models/recommended_train.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _bytes_to_gib(b: int) -> float:
    return b / (1024.0**3)


def probe_machine():
    import torch

    cpu = os.cpu_count() or 2
    out = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cpu_logical_cores": cpu,
        "gpu_name": None,
        "gpu_total_gib": None,
        "gpu_free_gib": None,
    }
    if not torch.cuda.is_available():
        return out
    try:
        props = torch.cuda.get_device_properties(0)
        out["gpu_name"] = props.name
        out["gpu_total_gib"] = round(props.total_memory / (1024**3), 2)
    except Exception as e:
        out["gpu_probe_error"] = str(e)
    try:
        free, total = torch.cuda.mem_get_info(0)
        out["gpu_free_gib"] = round(_bytes_to_gib(free), 2)
        if out["gpu_total_gib"] is None:
            out["gpu_total_gib"] = round(_bytes_to_gib(total), 2)
    except Exception:
        pass
    return out


def recommend(info: dict) -> dict:
    """启发式规则：偏保守，避免多 worker OOM。"""
    cpu = max(1, info["cpu_logical_cores"] - 1)
    n_workers_cap = min(8, cpu)

    if not info.get("cuda_available"):
        return {
            "DEVICE": "cpu",
            "WORKER_DEVICE": "cpu",
            "N_WORKERS": n_workers_cap,
            "MAX_CUDA_WORKERS": 3,
            "BATCH_SIZE": 256,
            "BUFFER_SIZE": 12000,
            "N_PLAYOUT_TRAIN": 320,
            "PLAYOUT_EARLY": 220,
            "PLAYOUT_MID": 320,
            "PLAYOUT_LATE": 400,
            "N_PLAYOUT_EVAL": 480,
            "LEAF_BATCH_SIZE": 8,
            "_note": "No CUDA: workers on CPU; lower batch/playout for throughput.",
        }

    total = info.get("gpu_total_gib") or 0.0
    free = info.get("gpu_free_gib")
    # 以总显存为主，空闲显存为辅（启动瞬间空闲可能偏大）
    mem_ref = total
    if free is not None and total:
        mem_ref = min(total, free + 2.0)

    # 注意：8.0 GiB 笔记本常见，用 <9 与 <12 分界，避免 8.0 误判进 12G 档
    if mem_ref < 4:
        max_cuda_w, batch, play_base = 1, 256, 300
        note = "Low VRAM: use 1 CUDA worker; keep playout moderate."
    elif mem_ref < 6:
        max_cuda_w, batch, play_base = 2, 320, 360
        note = "6GB class: 2 workers; on OOM use 1 or WORKER_DEVICE=cpu."
    elif mem_ref < 9:
        max_cuda_w, batch, play_base = 3, 384, 420
        note = "8GB class (laptop/desktop): 3 workers, moderate batch/playout."
    elif mem_ref < 12:
        max_cuda_w, batch, play_base = 3, 512, 440
        note = "10-12GB: higher batch/playout; keep MAX_CUDA_WORKERS <= 4."
    else:
        max_cuda_w, batch, play_base = 4, 512, 480
        note = "Large VRAM: more parallel + quality; reduce workers if OOM."

    n_workers = min(n_workers_cap, max_cuda_w)

    return {
        "DEVICE": "cuda",
        "WORKER_DEVICE": "cuda",
        "N_WORKERS": n_workers,
        "MAX_CUDA_WORKERS": max_cuda_w,
        "BATCH_SIZE": batch,
        "BUFFER_SIZE": min(22000, max(12000, batch * 36)),
        "N_PLAYOUT_TRAIN": play_base,
        "PLAYOUT_EARLY": max(200, int(play_base * 0.65)),
        "PLAYOUT_MID": play_base,
        "PLAYOUT_LATE": min(600, int(play_base * 1.2)),
        "N_PLAYOUT_EVAL": min(800, int(play_base * 1.35)),
        "LEAF_BATCH_SIZE": 16 if batch >= 384 else 12,
        "_note": note,
    }


def _safe_print(s: str) -> None:
    """Windows 下 conda run 管道常用 GBK，避免中文乱码/编码错误。"""
    enc = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        sys.stdout.write(s + "\n")
    except UnicodeEncodeError:
        sys.stdout.buffer.write((s + "\n").encode(enc, errors="replace"))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--json",
        default=None,
        help="将探测结果与建议写入该 JSON 路径",
    )
    args = p.parse_args()

    info = probe_machine()
    rec = recommend(info)
    merged = {"machine": info, "recommended": rec}

    _safe_print("======== Machine probe ========")
    _safe_print(f"  PyTorch        : {info['torch_version']}")
    _safe_print(f"  CPU cores      : {info['cpu_logical_cores']}")
    _safe_print(f"  CUDA           : {info['cuda_available']}")
    if info.get("gpu_name"):
        _safe_print(f"  GPU            : {info['gpu_name']}")
        _safe_print(f"  GPU total VRAM : {info.get('gpu_total_gib')} GiB")
    if info.get("gpu_free_gib") is not None:
        _safe_print(f"  GPU free VRAM  : {info['gpu_free_gib']} GiB")
    _safe_print("")
    _safe_print("======== Suggested Config overrides (paste into gomoku/config.py) ========")
    skip = {"_note"}
    for k, v in rec.items():
        if k in skip:
            continue
        _safe_print(f"  {k} = {v!r}")
    if rec.get("_note"):
        _safe_print("")
        _safe_print(f"  # {rec['_note']}")
    _safe_print("")
    _safe_print("======== Config class snippet ========")
    lines = []
    for k, v in rec.items():
        if k.startswith("_"):
            continue
        t = type(v).__name__
        lines.append(f"    {k}: {t} = {v!r}")
    _safe_print("\n".join(lines))

    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2, ensure_ascii=False)
        _safe_print("")
        _safe_print(f"Wrote JSON: {args.json}")
    _safe_print("")
    _safe_print(
        "Note: EPOCHS_PER_UPDATE / LR / TEMP / QUICK_EVAL are not auto-tuned; "
        "keep your gomoku/config.py values unless you change training philosophy."
    )


if __name__ == "__main__":
    main()
