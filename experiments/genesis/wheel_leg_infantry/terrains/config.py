"""地形预设、默认参数和配置校验；训练与 eval 共用。"""

from __future__ import annotations

import math
from copy import deepcopy

from .registry import TERRAIN_REGISTRY, default_parameters, level_parameters

TERRAIN_PRESETS = tuple(TERRAIN_REGISTRY)


def default_terrain_cfg(preset: str = "plane") -> dict:
    """返回适合当前 0.06 m 轮子的保守地形参数。"""
    return {
        "preset": preset,
        "bounded": preset == "flat",  # 兼容旧 flat；plane 默认不启用边界复位
        "tile_size": [24.0, 8.0],
        "horizontal_scale": 0.10,
        "vertical_scale": 0.005,
        "randomize": True,
        "assignment": "random",
        # 机器人接近 patch 接缝时按 timeout 重置，避免把接缝当成任务障碍。
        "boundary_margin": 0.75,
        "subterrain_parameters": default_parameters(),
    }


def resolve_terrain_cfg(config: dict | None) -> dict:
    """补齐并校验地形配置；旧实验没有该字段时继续使用 Plane。"""
    raw = {} if config is None else deepcopy(config)
    if not isinstance(raw, dict):
        raise TypeError("env_cfg['terrain'] must be a dictionary")

    if raw.get("mixture") is not None:
        from .mixture import resolve_mixture_cfg
        return resolve_mixture_cfg(raw)

    # 兼容旧 max_difficulty；新配置使用 difficulty。
    raw.pop("max_difficulty", None)
    preset = raw.get("preset", "plane")
    if preset == "flat":
        # 旧 flat 的表面也是 z=0，迁移到 Plane 并保留边界 timeout。
        preset = raw["preset"] = "plane"
        raw.setdefault("bounded", True)
    if preset in {"slope", "mixed"}:
        raise ValueError(f"terrain preset {preset!r} was removed; select a supported terrain with --terrain: {TERRAIN_PRESETS}")
    if preset not in TERRAIN_PRESETS:
        raise ValueError(f"unsupported terrain preset {preset!r}; choose from {TERRAIN_PRESETS}")

    resolved = default_terrain_cfg(preset)
    custom_parameters = raw.pop("subterrain_parameters", {})
    resolved.update(raw)
    for terrain_type, parameters in custom_parameters.items():
        # 旧实验可能携带已移除类型的参数，仅合并当前支持的地形。
        if terrain_type not in resolved["subterrain_parameters"]:
            continue
        resolved["subterrain_parameters"].setdefault(terrain_type, {}).update(parameters)
    if "difficulty" in raw:
        parameters = level_parameters(preset, raw["difficulty"], raw.get("level_catalog"))
        key = TERRAIN_REGISTRY[preset].PARAMETER_KEY
        if key:
            resolved["subterrain_parameters"][key] = parameters
    tile_size = tuple(float(value) for value in resolved["tile_size"])
    if len(tile_size) != 2 or not all(math.isfinite(value) and value > 0.0 for value in tile_size):
        raise ValueError("terrain.tile_size must contain two positive finite values")
    resolved["tile_size"] = list(tile_size)

    for name in ("horizontal_scale", "vertical_scale"):
        value = float(resolved[name])
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"terrain.{name} must be positive and finite")
        resolved[name] = value
    for size in tile_size:
        cells = size / resolved["horizontal_scale"]
        if not math.isclose(cells, round(cells), rel_tol=0.0, abs_tol=1e-6):
            raise ValueError("each terrain.tile_size value must be divisible by horizontal_scale")

    if not isinstance(resolved["bounded"], bool):
        raise TypeError("terrain.bounded must be a boolean")

    assignment = resolved["assignment"]
    if assignment not in {"random", "cyclic"}:
        raise ValueError("terrain.assignment must be 'random' or 'cyclic'")

    boundary_margin = resolved.get("boundary_margin")
    if boundary_margin is not None:
        boundary_margin = float(boundary_margin)
        if boundary_margin < 0.0 or 2.0 * boundary_margin >= min(tile_size):
            raise ValueError("terrain.boundary_margin must be non-negative and smaller than half a tile")
    resolved["boundary_margin"] = boundary_margin
    return resolved
