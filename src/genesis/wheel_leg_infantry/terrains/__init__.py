"""跨任务共享的地形配置与管理入口。"""

from .config import TERRAIN_PRESETS, default_terrain_cfg, resolve_terrain_cfg
from .manager import TerrainManager

__all__ = [
    "TERRAIN_PRESETS",
    "TerrainManager",
    "default_terrain_cfg",
    "resolve_terrain_cfg",
]
