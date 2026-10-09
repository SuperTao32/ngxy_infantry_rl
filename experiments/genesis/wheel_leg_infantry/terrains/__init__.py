"""跨任务共享的地形配置与管理入口。"""

from .config import TERRAIN_PRESETS, default_terrain_cfg, resolve_terrain_cfg
from .manager import TerrainManager, bilinear_height_at, make_terrain

__all__ = [
    "TERRAIN_PRESETS",
    "TerrainManager",
    "make_terrain",
    "default_terrain_cfg",
    "resolve_terrain_cfg",
    "bilinear_height_at",
]
