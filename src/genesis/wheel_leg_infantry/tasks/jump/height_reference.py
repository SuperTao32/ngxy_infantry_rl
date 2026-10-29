"""只依赖计时器的目标 base 到轮底平均竖直距离；训练与部署共用，不读取接触状态。"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch


def validate_height_reference(
    points: Sequence[Sequence[float]], cycle_s: float,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """校验 [时间(s), 目标 base 到轮底平均竖直距离(m)] 节点，覆盖从跳跃触发到回合结束。"""
    if not math.isfinite(cycle_s) or cycle_s <= 0.0:
        raise ValueError("height reference cycle_s must be positive and finite")
    if len(points) < 2 or any(len(point) != 2 for point in points):
        raise ValueError("height_reference must contain at least two [time_s, distance_m] points")
    times, heights = zip(*((float(t), float(h)) for t, h in points))
    if not all(math.isfinite(t) for t in times) or not all(
        math.isfinite(h) and h > 0.0 for h in heights
    ):
        raise ValueError("height_reference times must be finite and distances positive and finite")
    if times[0] != 0.0 or not math.isclose(times[-1], cycle_s, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("height_reference must start at 0 and end at episode_length_s")
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("height_reference times must be strictly increasing")
    return times, heights


def sample_height_reference(
    elapsed_s: torch.Tensor, times: torch.Tensor, heights: torch.Tensor,
) -> torch.Tensor:
    """分段五次平滑插值，节点速度/加速度为零，范围外保持首尾值。

    times/heights 是校验后的一维同设备张量；elapsed_s 可为标量或一批计时器。
    参考值表示 base 原点到左右轮底的平均世界竖直距离，与地面高度无关。
    此函数仅生成目标值；实测距离 = base_z - mean(wheel_center_z - wheel_radius)。
    """
    segment = torch.bucketize(elapsed_s.contiguous(), times, right=True) - 1
    segment = segment.clamp(0, times.numel() - 2)
    progress = ((elapsed_s - times[segment]) / (times[segment + 1] - times[segment])).clamp(0.0, 1.0)
    blend = progress**3 * (10.0 + progress * (-15.0 + 6.0 * progress))
    return heights[segment] + (heights[segment + 1] - heights[segment]) * blend
