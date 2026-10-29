from collections.abc import Mapping

import torch


DEFAULT_TRACKING_GATE_CFG = {
    "height_full_error": 0.02,
    "height_zero_error": 0.05,
    "attitude_full_angle_deg": 5.0,
    "attitude_zero_angle_deg": 12.0,
    "floor": 0.10,
}


def merge_tracking_gate_config(current: Mapping | None, values: Mapping) -> dict[str, float]:
    """合并并校验一组可由课程阶段局部覆盖的门控参数。"""
    unknown = set(values).difference(DEFAULT_TRACKING_GATE_CFG)
    if unknown:
        raise KeyError(f"Unsupported tracking-gate keys: {sorted(unknown)}")

    updated = dict(DEFAULT_TRACKING_GATE_CFG)
    updated.update(current or {})
    updated.update({name: float(value) for name, value in values.items()})

    if not 0.0 <= updated["height_full_error"] < updated["height_zero_error"]:
        raise ValueError("tracking gate height errors must satisfy 0 <= full < zero")
    if not 0.0 <= updated["attitude_full_angle_deg"] < updated["attitude_zero_angle_deg"]:
        raise ValueError("tracking gate attitude angles must satisfy 0 <= full < zero")
    if not 0.0 <= updated["floor"] <= 1.0:
        raise ValueError("tracking gate floor must be in [0, 1]")

    return updated


def smooth_gate(error: torch.Tensor, full_threshold: float, zero_threshold: float) -> torch.Tensor:
    """C1 连续门控：误差不超过 full 时为 1，达到 zero 时平滑降为 0。"""
    if not 0.0 <= full_threshold < zero_threshold:
        raise ValueError("smooth-gate thresholds must satisfy 0 <= full < zero")
    normalized = ((error - full_threshold) / (zero_threshold - full_threshold)).clamp(0.0, 1.0)
    return 1.0 - normalized * normalized * (3.0 - 2.0 * normalized)
