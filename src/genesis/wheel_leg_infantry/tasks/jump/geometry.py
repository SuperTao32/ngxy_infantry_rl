"""不依赖 Genesis 的起跳距离和目标落地几何，可在 CPU 单测。"""

import torch


def active_step_heights(cfg):
    """任务编码保持不变；未启用台阶的通道按平地处理。"""
    enabled = cfg.get("platform_enabled", [True] * len(cfg["step_heights_m"]))
    return [height if active else 0.0 for height, active in zip(cfg["step_heights_m"], enabled)]


def trigger_distance(speed, mode, tables):
    """逐模式分段线性插值；显式拒绝未标定速度，避免默默外推。"""
    result = torch.zeros_like(speed)
    if torch.any((mode < 0) | (mode >= len(tables))):
        raise ValueError("invalid jump mode")
    for index, points in enumerate(tables):
        selected = mode == index
        if not torch.any(selected):
            continue
        knots = torch.as_tensor(points, dtype=speed.dtype, device=speed.device)
        values = speed[selected]
        if not torch.isfinite(values).all() or torch.any((values < knots[0, 0]) | (values > knots[-1, 0])):
            raise ValueError(f"mode {index}: speed outside configured distance table")
        segment = (torch.bucketize(values.contiguous(), knots[:, 0].contiguous(), right=True) - 1).clamp(0, len(points) - 2)
        weight = (values - knots[segment, 0]) / (knots[segment + 1, 0] - knots[segment, 0])
        result[selected] = torch.lerp(knots[segment, 1], knots[segment + 1, 1], weight)
    return result


def target_wheel_support(positions, contacts, mode, cfg, wheel_radius, *, margin=None):
    """返回 [N,2] 台面/平地支撑；侧壁接触或仅悬在台面上方不计入。"""
    margin = cfg["landing_margin_m"] if margin is None else margin
    heights = positions.new_tensor(active_step_heights(cfg))[mode, None]
    lane_y = mode.to(positions.dtype)[:, None] * cfg["lane_spacing_m"]
    x, y, z = positions.unbind(-1)
    height_match = torch.abs(z - wheel_radius - heights) <= cfg["landing_height_tolerance_m"]
    inside = ((x >= margin) & (x <= cfg["platform_length_m"] - margin)
              & (torch.abs(y - lane_y) <= cfg["platform_width_m"] / 2 - margin))
    inside = torch.where(heights == 0, torch.ones_like(inside), inside)
    return (contacts > 0.5) & height_match & inside
