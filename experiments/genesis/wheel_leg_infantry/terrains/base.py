"""单一地形的公共接口；具体几何与参数由各地形模块负责。"""

from abc import ABC, abstractmethod


class TerrainGeometry(ABC):
    def __init__(self, parameters, tile_size, horizontal_scale=0.1, *, boundary_margin=None):
        self.tile_size = tuple(tile_size)
        self.origin = (-self.tile_size[0] / 2, -self.tile_size[1] / 2, 0.0)
        self.boundary_margin = boundary_margin

    @abstractmethod
    def add_to_scene(self, scene):
        """创建实际碰撞实体，返回单个实体或实体元组。"""

    @abstractmethod
    def height_at(self, local_xy):
        """查询以 patch 中心为原点的局部地面高度。"""

    def bind(self, *, device, dtype):
        """按需创建查询缓存；解析几何无需绑定。"""

    def adjust_spawn(self, positions, rpy, *, mask=None, offset=None):
        """调整选中机器人的出生姿态；offset 是 patch 中心，默认保留采样结果。"""

    def create_morph(self):
        raise RuntimeError("segmented terrain contains multiple morphs; use add_to_scene()")
