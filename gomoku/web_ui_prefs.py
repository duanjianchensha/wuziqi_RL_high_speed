"""
Web 界面偏好：对弈加载的权重路径等（models/web_ui_prefs.json）。
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from gomoku.config import config


def _path() -> str:
    return getattr(config, "WEB_UI_PREFS_PATH", os.path.join(config.MODEL_DIR, "web_ui_prefs.json"))


def load_prefs() -> Dict[str, Any]:
    p = _path()
    if not os.path.isfile(p):
        return {}
    try:
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def save_prefs(data: Dict[str, Any]) -> None:
    p = _path()
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def get_play_model_path() -> Optional[str]:
    """返回应加载的对弈权重路径（相对项目根或绝对）；无效则 None。"""
    d = load_prefs()
    raw = d.get("play_model_path")
    if not raw or not isinstance(raw, str):
        return None
    s = raw.strip()
    if not s:
        return None
    return s


def set_play_model_path(path: str) -> None:
    d = load_prefs()
    d["play_model_path"] = path.strip()
    save_prefs(d)
