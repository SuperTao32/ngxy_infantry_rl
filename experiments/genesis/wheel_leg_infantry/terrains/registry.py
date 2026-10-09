"""统一注册地形模块；参数、难度和几何实现均来自对应模块。"""

from copy import deepcopy

from . import plane, stairs, platform_ridge, trapezoidal_wave, square_wave, random_rough

TERRAIN_REGISTRY = {
    module.PRESET: module
    for module in (plane, stairs, platform_ridge, trapezoidal_wave, square_wave, random_rough)
}


def default_parameters():
    return {m.PARAMETER_KEY: deepcopy(m.PARAMETERS) for m in TERRAIN_REGISTRY.values() if m.PARAMETER_KEY}


def level_catalog():
    """完整快照随训练配置保存，避免改参数文件后悄悄改变续训地形。"""
    return {
        name: {str(level): {**deepcopy(module.PARAMETERS), **deepcopy(overrides)}
               for level, overrides in module.LEVELS.items()}
        for name, module in TERRAIN_REGISTRY.items()
    }


def level_parameters(preset, difficulty, catalog=None):
    if isinstance(difficulty, bool) or not isinstance(difficulty, int):
        raise ValueError("terrain difficulty must be an integer")
    levels = (level_catalog() if catalog is None else catalog).get(preset, {})
    if str(difficulty) not in levels:
        raise ValueError(f"undefined difficulty {difficulty} for {preset}; available: {list(levels)}")
    parameters = levels[str(difficulty)]
    expected = set(TERRAIN_REGISTRY[preset].PARAMETERS)
    if set(parameters) != expected:
        raise ValueError(f"invalid {preset} level {difficulty} parameter keys: expected {sorted(expected)}")
    return deepcopy(parameters)
