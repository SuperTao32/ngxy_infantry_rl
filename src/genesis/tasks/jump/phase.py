"""不依赖 Genesis 的跳跃相位工具。"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch

from .config import MODE_NAMES, PHASE_NAMES


PHASE_ENCODING_DIM = 4


def validate_phase_durations(durations: Mapping[str, float]) -> tuple[float, ...]:
    values = tuple(float(durations[name]) for name in PHASE_NAMES)
    if not all(math.isfinite(value) and value > 0.0 for value in values):
        raise ValueError("Jump phase durations must be positive and finite")
    return values


def resolve_input_durations(env_cfg: Mapping) -> tuple[float, ...]:
    """按模式解析输入窗口；缺省时沿用阶段总时长。"""
    value = env_cfg.get("jump_input_duration_s")
    if value is None:
        value = sum(validate_phase_durations(env_cfg["jump_phase_durations_s"]))
    values = value if isinstance(value, (list, tuple)) else [value] * len(MODE_NAMES)
    durations = tuple(float(item) for item in values)
    if len(durations) != len(MODE_NAMES) or not all(math.isfinite(item) and item > 0.0 for item in durations):
        raise ValueError(f"jump_input_duration_s requires {len(MODE_NAMES)} positive finite durations")
    return durations


def resolve_input_duration(env_cfg: Mapping, mode: int = 0) -> float:
    """取得指定跳高模式的输入窗口，供交互评估使用。"""
    return resolve_input_durations(env_cfg)[mode]


def phase_encoding(elapsed_s: torch.Tensor, cycle_s: float) -> torch.Tensor:
    """四维 sin/cos 编码；cycle_s 为配置的输入时长，窗口结束后置零。"""
    progress = torch.clamp(elapsed_s / cycle_s, 0.0, 1.0)
    phase = math.pi * progress
    encoding = torch.stack(
        (
            torch.sin(phase),
            torch.cos(phase),
            torch.sin(phase / 2.0),
            torch.cos(phase / 2.0),
        ),
        dim=-1,
    )
    outside_window = (elapsed_s < 0.0) | (elapsed_s >= cycle_s)
    return encoding.masked_fill(outside_window.unsqueeze(-1), 0.0)


def scheduled_phase_index(elapsed_s: torch.Tensor, durations: tuple[float, ...]) -> torch.Tensor:
    """按固定时间表返回 0=takeoff, 1=flight, 2=landing。"""
    boundaries = torch.tensor(
        [sum(durations[:i]) for i in range(1, len(PHASE_NAMES))],
        dtype=elapsed_s.dtype,
        device=elapsed_s.device,
    )
    return torch.bucketize(elapsed_s.contiguous(), boundaries, right=True)
