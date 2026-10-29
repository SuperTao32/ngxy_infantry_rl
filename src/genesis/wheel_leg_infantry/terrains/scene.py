"""将共享地形几何放入独立 patch，不改变其局部坐标。"""

import torch


class OffsetScene:
    def __init__(self, scene, offset, tile_size):
        self.scene, self.offset, self.tile_size = scene, offset, tile_size

    def add_entity(self, morph, **kwargs):
        import genesis as gs

        if isinstance(morph, gs.morphs.Plane):
            x, y = self.tile_size
            morph = gs.morphs.Box(lower=(-x / 2, -y / 2, -.1), upper=(x / 2, y / 2, 0.),
                                  fixed=True, batch_fixed_verts=False)
        updates = {"pos": tuple(a + b for a, b in zip(morph.pos, self.offset))}
        if isinstance(morph, gs.morphs.Box):
            for name in ("lower", "upper"):
                updates[name] = tuple(a + b for a, b in zip(getattr(morph, name), self.offset))
        return self.scene.add_entity(morph.model_copy(update=updates), **kwargs)


class TerrainPatches:
    """有限地形区域的共享摆放与查询；类型分配和终止规则由调用方负责。"""

    is_plane = False

    def __init__(self, terrains):
        self.terrains = list(terrains)
        self.tile_size = tuple(max(t.tile_size[axis] for t in self.terrains) for axis in (0, 1))
        self.stride = self.tile_size[1] + 4.0
        self.entities = []
        # 各区域隔离，单环境的碰撞容量与预构建区域数量无关。
        self.max_collision_pairs = max(t.max_collision_pairs for t in self.terrains)

    def add_to_scene(self, scene):
        self.entities = []
        for index, terrain in enumerate(self.terrains):
            offset = (0., index * self.stride, 0.)
            entities = terrain.add_to_scene(OffsetScene(scene, offset, terrain.tile_size))
            self.entities.append(entities if isinstance(entities, tuple) else (entities,))
        return tuple(entity for group in self.entities for entity in group)

    def bind_entity(self, entity, *, device, dtype):
        for index, terrain in enumerate(self.terrains):
            group = self.entities[index] if self.entities else ()
            terrain.bind_entity(group, device=device, dtype=dtype)
        self.centers = torch.zeros((len(self.terrains), 2), device=device, dtype=dtype)
        self.centers[:, 1] = torch.arange(len(self.terrains), device=device) * self.stride

    def height_at(self, world_xy):
        indices = torch.round(world_xy[:, 1] / self.stride).long()
        heights = torch.zeros_like(world_xy[:, 0])
        for index, terrain in enumerate(self.terrains):
            local = world_xy - self.centers[index]
            inside = (local.abs() <= world_xy.new_tensor(terrain.tile_size) / 2).all(dim=-1)
            heights = torch.where((indices == index) & inside, terrain.height_at(local), heights)
        return heights

    def adjust_spawn(self, positions, rpy, tiles):
        for index, terrain in enumerate(self.terrains):
            terrain.adjust_spawn(positions, rpy, mask=tiles == index, offset=self.centers[index])

    def reset(self, env_ids):
        """静态区域无需在回合复位时修改几何。"""
