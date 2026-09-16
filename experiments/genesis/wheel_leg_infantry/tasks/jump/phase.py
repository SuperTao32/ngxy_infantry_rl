"""不依赖 Genesis 的跳跃相位工具。"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch

from .config_25cm import PHASE_NAMES


def validate_phase_durations(durations: Mapping[str, float]) -> tuple[float, ...]:
    unknown = set(durations).difference(PHASE_NAMES)
    missing = set(PHASE_NAMES).difference(durations)
    if unknown or missing:
        raise ValueError(f"jump phase keys must be {PHASE_NAMES}; missing={sorted(missing)}, unknown={sorted(unknown)}")
    values = tuple(float(durations[name]) for name in PHASE_NAMES)
    if not math.isfinite(values[0]) or values[0] < 0.0:
        raise ValueError("crouch duration must be non-negative and finite; zero skips crouch")
    if not all(math.isfinite(value) and value > 0.0 for value in values[1:]):
        raise ValueError("takeoff, flight and landing durations must be positive and finite")
    return values


def phase_encoding(elapsed_s: torch.Tensor, cycle_s: float) -> torch.Tensor:
    """生成与 backflip 类似的 6 维多尺度 sin/cos 时间编码。"""
    if cycle_s <= 0.0 or not math.isfinite(cycle_s):
        raise ValueError("cycle_s must be positive and finite")
    progress = torch.clamp(elapsed_s / cycle_s, 0.0, 1.0)
    phase = math.pi * progress
    return torch.stack(
        (
            torch.sin(phase),
            torch.cos(phase),
            torch.sin(phase / 2.0),
            torch.cos(phase / 2.0),
            torch.sin(phase / 4.0),
            torch.cos(phase / 4.0),
        ),
        dim=-1,
    )


def scheduled_phase_index(elapsed_s: torch.Tensor, durations: tuple[float, ...]) -> torch.Tensor:
    """按固定时间表返回 0=crouch, 1=takeoff, 2=flight, 3=landing。"""
    if len(durations) != len(PHASE_NAMES):
        raise ValueError(f"durations must contain {len(PHASE_NAMES)} values")
    boundaries = torch.tensor(
        [durations[0], durations[0] + durations[1], sum(durations[:3])],
        dtype=elapsed_s.dtype,
        device=elapsed_s.device,
    )
    return torch.bucketize(elapsed_s.contiguous(), boundaries, right=True)
