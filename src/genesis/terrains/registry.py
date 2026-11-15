"""统一注册地形模块；参数、难度和几何实现均来自对应模块。"""

from copy import deepcopy

from . import trapezoidal_wave
from . import plane
from . import platform_ridge_down
from . import platform_ridge_up
from . import slope_down
from . import slope_up
from . import square_wave
from . import stairs
from . import stairs_up

TERRAIN_REGISTRY = {
    module.PRESET: module
    for module in (plane, stairs, platform_ridge_down, platform_ridge_up, trapezoidal_wave, square_wave, stairs_up, slope_up, slope_down)
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
    levels = (level_catalog() if catalog is None else catalog)[preset]
    return deepcopy(levels[str(difficulty)])
