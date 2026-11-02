"""只依赖计时器的目标 base 到轮底平均竖直距离；训练与部署共用，不读取接触状态。"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch


def validate_height_reference(config: Mapping[str, float]) -> tuple[float, float]:
    """所有模式共用一组 base 到轮底距离上下限，单位 m。"""
    if not isinstance(config, Mapping) or set(config) not in (
        {"min_distance_m", "max_distance_m"},
        {"min_distance_m", "max_distance_m", "retract_duration_s"},
    ):
        raise ValueError("jump_height_reference requires min_distance_m, max_distance_m and optional retract_duration_s")
    try:
        minimum, maximum = float(config["min_distance_m"]), float(config["max_distance_m"])
    except (ValueError, TypeError) as exc:
        raise ValueError("jump_height_reference distances must be numeric") from exc
    if not all(math.isfinite(value) and value > 0.0 for value in (minimum, maximum)) or minimum > maximum:
        raise ValueError("jump_height_reference must have finite 0 < min_distance_m <= max_distance_m")
    if "retract_duration_s" in config:
        _validate_retract_duration(config["retract_duration_s"])
    return minimum, maximum


def resolve_height_reference(env_cfg) -> tuple[float, float]:
    """新配置读取共享上下限；旧存档只提取原轨迹的共同最低/最高值。"""
    if "jump_height_reference" in env_cfg:
        return validate_height_reference(env_cfg["jump_height_reference"])
    legacy = env_cfg.get("jump_modes", {}).get("height_references")
    if not legacy or not any(legacy):
        raise ValueError("Missing jump_height_reference configuration")
    heights = [float(point[1]) for points in legacy for point in points]
    return validate_height_reference(dict(min_distance_m=min(heights), max_distance_m=max(heights)))


def _validate_retract_duration(value) -> float:
    try:
        duration = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError("jump_height_reference.retract_duration_s must be finite and positive") from exc
    if not math.isfinite(duration) or duration <= 0.0:
        raise ValueError("jump_height_reference.retract_duration_s must be finite and positive")
    return duration


def resolve_retract_duration(env_cfg) -> float:
    """收腿独立计时；缺少该字段的旧存档保留按 flight 时长收腿的行为。"""
    config = env_cfg.get("jump_height_reference", {})
    if "retract_duration_s" in config:
        return _validate_retract_duration(config["retract_duration_s"])
    return _validate_retract_duration(env_cfg["jump_phase_durations_s"]["flight"])


def sample_height_reference(
    elapsed_s: torch.Tensor, *, takeoff_s: float, retract_duration_s: float, min_distance_m: float, max_distance_m: float,
) -> torch.Tensor:
    """takeoff 从最低升到最高，随后在独立收腿时长内降到最低，再保持最低。

    两段使用五次平滑插值，边界速度和加速度为零。收腿不依赖 flight 的结束时刻。
    时长、上下限由配置层校验；
    elapsed_s 可为标量或一批独立计时器。参考为 base 原点到左右轮底的平均竖直距离。
    """
    def smoothstep(progress):
        progress = progress.clamp(0.0, 1.0)
        return progress**3 * (10.0 + progress * (-15.0 + 6.0 * progress))

    rise = smoothstep(elapsed_s / takeoff_s)
    fall = smoothstep((elapsed_s - takeoff_s) / retract_duration_s)
    return min_distance_m + (max_distance_m - min_distance_m) * (rise - fall)
