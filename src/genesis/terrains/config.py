"""地形预设、默认参数和配置校验；训练与 eval 共用。"""

from __future__ import annotations

import math
from copy import deepcopy

from .registry import TERRAIN_REGISTRY, default_parameters, level_catalog, level_parameters
from .curriculum import default_difficulty_curriculum_cfg

TERRAIN_PRESETS = tuple(TERRAIN_REGISTRY)


def default_terrain_cfg(preset: str = "plane") -> dict:
    """返回适合当前 0.06 m 轮子的保守地形参数。"""
    return {
        "preset": preset,
        "bounded": preset != "plane",
        "tile_size": [24.0, 8.0],
        "horizontal_scale": 0.10,
        "vertical_scale": 0.005,
        # 机器人接近 patch 接缝时按 timeout 重置，避免把接缝当成任务障碍。
        "boundary_margin": 0.75,
        "subterrain_parameters": default_parameters(),
        "level_catalog": level_catalog(),
        "curriculum": default_difficulty_curriculum_cfg(),
    }


def resolve_terrain_cfg(config: dict) -> dict:
    """合并默认配置，统一校验，再解析可存档的地形参数。"""
    preset = config["preset"]
    resolved = default_terrain_cfg(preset)
    resolved.update(deepcopy(config))
    resolved["curriculum"] = {**default_difficulty_curriculum_cfg(), **resolved["curriculum"]}
    _validate_terrain_cfg(resolved)

    parameters = default_parameters()
    for name, overrides in resolved["subterrain_parameters"].items():
        parameters[name].update(overrides)
    resolved["subterrain_parameters"] = parameters
    resolved["tile_size"] = [float(value) for value in resolved["tile_size"]]
    for name in ("horizontal_scale", "vertical_scale"):
        resolved[name] = float(resolved[name])
    if resolved["boundary_margin"] is not None:
        resolved["boundary_margin"] = float(resolved["boundary_margin"])

    if preset == "mixture":
        for tile in resolved["tiles"]:
            if resolved["curriculum"]["enabled"]:
                # 保留显式覆盖，避免再次 resolve 时用初始级参数覆盖其他等级。
                overrides = tile.setdefault("parameters", {})
                tile["resolved_levels"] = {
                    level: {**deepcopy(parameters), **overrides}
                    for level, parameters in resolved["level_catalog"][tile["preset"]].items()
                }
                tile.setdefault("difficulty", min(int(level) for level in tile["resolved_levels"]))
            else:
                parameters = _parameters(resolved, tile["preset"], tile)
                parameters.update(tile.get("parameters", {}))
                tile["parameters"] = parameters
            tile["weight"] = float(tile.get("weight", 1.0))
    else:
        key = TERRAIN_REGISTRY[preset].PARAMETER_KEY
        if key:
            resolved["subterrain_parameters"][key] = _parameters(resolved, preset, resolved)
    return resolved


def _parameters(config, preset, values):
    if "difficulty" in values:
        return level_parameters(preset, values["difficulty"], config["level_catalog"])
    key = TERRAIN_REGISTRY[preset].PARAMETER_KEY
    return deepcopy(config["subterrain_parameters"].get(key, {}))


def _validate_terrain_cfg(config):
    """只检查配置约束；一次报告所有问题，不参与参数合并。"""
    errors = []
    _check_keys(config, default_terrain_cfg().keys() | {"difficulty", "tiles"}, "terrain", errors)
    defaults = default_parameters()
    curriculum = config["curriculum"]
    _check_keys(curriculum, default_difficulty_curriculum_cfg(), "terrain.curriculum", errors)
    if type(curriculum["min_completed"]) is not int or curriculum["min_completed"] < 1:
        errors.append("curriculum.min_completed must be a positive integer")
    if not 0 <= curriculum["demote_threshold"] < curriculum["promote_threshold"] <= 1:
        errors.append("curriculum thresholds must satisfy 0 <= demote < promote <= 1")
    if curriculum["enabled"] and config["preset"] != "mixture":
        errors.append("terrain curriculum requires preset='mixture'")
    _check_keys(config["subterrain_parameters"], defaults, "subterrain_parameters", errors)
    for name in config["subterrain_parameters"].keys() & defaults.keys():
        _check_keys(config["subterrain_parameters"][name], defaults[name], name, errors)

    size = tuple(float(value) for value in config["tile_size"])
    valid_size = len(size) == 2 and all(math.isfinite(value) and value > 0 for value in size)
    if not valid_size:
        errors.append("tile_size must contain two positive finite values")
    for name in ("horizontal_scale", "vertical_scale"):
        value = float(config[name])
        if not math.isfinite(value) or value <= 0:
            errors.append(f"{name} must be positive and finite")
    margin = config["boundary_margin"]
    if margin is not None:
        margin = float(margin)
        if not math.isfinite(margin) or margin < 0 or (valid_size and 2 * margin >= min(size)):
            errors.append("boundary_margin must be non-negative and smaller than half a tile")

    if config["preset"] == "mixture":
        if "difficulty" in config:
            errors.append("mixture difficulty must be configured per tile")
        if not config["bounded"] or margin is None:
            errors.append("mixture requires bounded=True and a boundary_margin")
        tiles = config.get("tiles")
        if not isinstance(tiles, list) or not tiles:
            errors.append("tiles must be a non-empty list")
            tiles = []
        weights = [float(tile.get("weight", 1.0)) for tile in tiles]
        if any(not math.isfinite(weight) or weight < 0 for weight in weights) or not any(weight > 0 for weight in weights):
            errors.append("tile weights must be finite, non-negative and include a positive value")
        for tile in tiles:
            _check_keys(tile, {"preset", "difficulty", "weight", "parameters", "resolved_levels"}, "tile", errors)
        if curriculum["enabled"] and len({tile["preset"] for tile in tiles}) != len(tiles):
            errors.append("terrain curriculum requires one tile configuration per preset")
    else:
        if "tiles" in config:
            errors.append("tiles requires preset='mixture'")
        tiles = [config]

    for tile in tiles:
        preset = tile["preset"]
        if preset not in TERRAIN_PRESETS:
            errors.append(f"unsupported terrain preset {preset!r}")
            continue
        _check_keys(tile.get("parameters", {}), TERRAIN_REGISTRY[preset].PARAMETERS, preset, errors)
        if "difficulty" in tile:
            difficulty = tile["difficulty"]
            levels = config["level_catalog"][preset]
            if type(difficulty) is not int or str(difficulty) not in levels:
                errors.append(f"{preset}.difficulty must be one of {tuple(levels)}")
    if errors:
        raise ValueError("Invalid terrain config:\n- " + "\n- ".join(errors))


def _check_keys(values, allowed, name, errors):
    unknown = values.keys() - allowed
    if unknown:
        errors.append(f"unsupported {name} keys: {sorted(unknown)}")
