"""把各训练任务的几何放进同一个 Genesis Scene。"""

import torch

from .scene import OffsetScene


class TaskTerrainRegion:
    """仅平移 y 坐标，保留任务内部的 tile 编号和几何参数表。"""

    def __init__(self, terrain, offset_y):
        self.terrain, self.offset_y = terrain, offset_y

    def __getattr__(self, name):
        return getattr(self.terrain, name)

    def _offset(self, values):
        return values.new_tensor([0.0, self.offset_y])

    def sample_spawn_tiles(self, env_ids):
        centers, tiles = self.terrain.sample_spawn_tiles(env_ids)
        return centers + self._offset(centers), tiles

    def height_at(self, xy):
        return self.terrain.height_at(xy - self._offset(xy))

    def adjust_spawn(self, positions, rpy, tiles):
        local = positions.clone()
        local[:, 1] -= self.offset_y
        self.terrain.adjust_spawn(local, rpy, tiles)
        positions.copy_(local)
        positions[:, 1] += self.offset_y

    def out_of_bounds(self, xy, centers):
        offset = self._offset(xy)
        if self.terrain.is_plane:
            # Plane morphs become finite floor patches in the shared scene.
            # Jump lane spawn centers are not floor centers.
            margin = self.terrain.config.get("boundary_margin") or 0.5
            extent = xy.new_tensor(self.terrain.tile_size) / 2 - margin
            return ((xy - offset).abs() >= extent).any(dim=-1)
        return self.terrain.out_of_bounds(xy - offset, centers - offset)


class TrainingTerrain:
    preset = "training_tasks"
    is_plane = False
    config = {"preset": "training_tasks", "boundary_margin": 0.25}

    def __init__(self, tasks):
        self.tasks = tasks
        # Jump support geometry uses absolute lane y; keep it at y=0.
        offsets = {"jump": 0.0, "locomotion": -24.0, "recovery": -48.0, "multi_terrain": 24.0}
        for name, task in tasks.items():
            task.terrain = TaskTerrainRegion(task.terrain, offsets[name])
        self.tile_size = tuple(max(task.terrain.tile_size[i] for task in tasks.values()) for i in (0, 1))
        self.max_collision_pairs = max(task.terrain.max_collision_pairs for task in tasks.values())

    def add_to_scene(self, scene):
        entities = []
        for task in self.tasks.values():
            terrain = task.terrain
            added = terrain.add_to_scene(OffsetScene(scene, (0., terrain.offset_y, 0.), terrain.tile_size))
            entities.extend(added if isinstance(added, tuple) else (added,))
        return tuple(entities)

    def bind_entity(self, entity, *, device, dtype):
        for task in self.tasks.values():
            task.terrain.bind_entity(entity, device=device, dtype=dtype)

    def sample_spawn_tiles(self, env_ids):
        env = next(iter(self.tasks.values())).env
        centers = torch.empty((len(env_ids), 2), dtype=env.base_pos.dtype, device=env.device)
        tiles = torch.empty_like(env_ids)
        for task in self.tasks.values():
            selected = task.mask[env_ids]
            ids = env_ids[selected]
            if ids.numel():
                centers[selected], tiles[selected] = task.terrain.sample_spawn_tiles(ids)
        return centers, tiles

    def height_at(self, xy):
        # All non-flat regions occupy disjoint y ranges; flat surfaces are z=0.
        return sum(task.terrain.height_at(xy) for task in self.tasks.values())

    def out_of_bounds(self, xy, centers):
        result = torch.zeros(len(xy), dtype=torch.bool, device=xy.device)
        for task in self.tasks.values():
            result[task.mask] = task.terrain.out_of_bounds(xy[task.mask], centers[task.mask])
        return result

    def reset(self, env_ids):
        for task in self.tasks.values():
            selected = env_ids[task.mask[env_ids]]
            if selected.numel():
                task.terrain.reset(selected)
