"""Locomotion 与 jump 的 actor 观测布局契约。"""

from __future__ import annotations

from collections.abc import Sequence


LOCOMOTION_POLICY_LAYOUT: tuple[tuple[str, int], ...] = (
    ("imu_ang_vel", 3),
    ("imu_lin_acc", 3),
    ("projected_gravity", 3),
    ("commands", 3),
    ("joint_pos_offset", 4),
    ("joint_vel", 4),
    ("wheel_vel", 2),
    ("leg_length", 2),
    ("leg_angle", 2),
    ("actions", 6),
)

# 前 32 维的顺序和缩放与 locomotion 一致，checkpoint 第一层可按前缀复制。
# warmup teacher 的 commands[:, 2] 为目标 base 离地高度。
# jump 时第三项为目标 base 到左右轮底的平均世界竖直距离，由模式和计时器生成。
JUMP_POLICY_LAYOUT = LOCOMOTION_POLICY_LAYOUT + (
    ("last_actions", 6),
    ("jump_phase", 6),
    ("jump_mode", 3),  # flat / step_20cm / step_40cm one-hot
)


def layout_dim(layout: Sequence[tuple[str, int]]) -> int:
    return sum(width for _, width in layout)


def layout_slices(layout: Sequence[tuple[str, int]]) -> dict[str, slice]:
    result: dict[str, slice] = {}
    start = 0
    for name, width in layout:
        if name in result:
            raise ValueError(f"duplicate observation component: {name}")
        if width <= 0:
            raise ValueError(f"observation width must be positive: {name}={width}")
        result[name] = slice(start, start + width)
        start += width
    return result
