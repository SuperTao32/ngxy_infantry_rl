"""将 terrains 注册地形及难度放入独立任务区域；这里只管理通关终点和分组。"""

import torch

from ...terrains.manager import TerrainManager
from ...terrains.scene import TerrainPatches
from .curriculum import TaskCurriculum, positive


class TaskTrack:
    """共享几何的任务适配器；不再自行生成碰撞体或计算地形高度。"""

    def __init__(self, name, stage):
        self.kind = name
        self.terrain = TerrainManager(stage["terrain"])
        self.geometry = self.terrain.geometry
        self.direction = getattr(self.geometry, "travel_direction", 1.0)
        self.tile_size = self.terrain.tile_size
        self.boundary_margin = self.terrain.config["boundary_margin"]
        if self.boundary_margin is None:
            raise ValueError("task terrain requires a finite boundary_margin")
        self.obstacle_start = self.obstacle_end = self.goal_x = 0.0
        traversal = stage["traversal"]
        positive(traversal["landing_length"], "traversal.landing_length")
        # 台阶/坡的末端由真实几何提供；连续随机地面由 task 指定通行距离。
        self.obstacle_end = getattr(self.geometry, "traversal_end", None)
        if self.obstacle_end is None:
            self.obstacle_end = traversal.get("distance")
            if self.obstacle_end is None:
                raise ValueError(f"{name}: terrain without an obstacle end requires traversal.distance")
        elif "distance" in traversal:
            raise ValueError(f"{name}: traversal.distance cannot replace the geometry's obstacle end")
        positive(self.obstacle_end, "obstacle_end")
        # 平台凸台下行先遇到凸台，上行先遇到第一平台。
        if name in {"platform_ridge_down", "platform_ridge_up"}:
            self.obstacle_start = self.geometry.boxes[1 if self.geometry.ascending else 2][0][0]
        else:
            self.obstacle_start = self.geometry.start_x
        self.goal_x = self.obstacle_end + self.direction * traversal["landing_length"]
        if abs(self.goal_x) + max(0.5, self.boundary_margin) >= self.tile_size[0] / 2 or self.tile_size[1] / 2 <= 1.0:
            raise ValueError("track needs safe finish and lateral margins within tile_size")

    def height_at(self, xy):
        return self.terrain.height_at(xy)


class TaskTerrain(TerrainPatches):
    preset = "multi_terrain"

    def __init__(self, tasks, counts, *, env_ids, num_envs):
        self.env_ids, self.num_envs = env_ids, num_envs
        self.tasks, self.counts = tasks, counts
        self.controller = TaskCurriculum(tasks)
        self.tracks, self.variants, self.stages, self.lookup = [], [], [], {}
        for i, (name, task) in enumerate(tasks.items()):
            for stage in task["resolved_stages"]:
                level = stage["difficulty"]
                self.lookup[i, level] = len(self.tracks)
                track = TaskTrack(name, stage)
                if stage["termination"]["corridor_half_width"] >= track.tile_size[1] / 2 - track.boundary_margin:
                    raise ValueError("corridor must fit inside the terrain")
                if stage["success"]["wheel_clearance_m"] >= stage["traversal"]["landing_length"]:
                    raise ValueError("finish must leave room for wheel clearance")
                self.tracks.append(track)
                self.variants.append((i, level))
                self.stages.append(stage)
        super().__init__(track.terrain for track in self.tracks)
        self.config = dict(preset=self.preset, boundary_margin=0.25)

    def bind_entity(self, entity, *, device, dtype):
        super().bind_entity(entity, device=device, dtype=dtype)
        self.fixed_types = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        self.fixed_types[self.env_ids] = torch.repeat_interleave(
            torch.arange(len(self.tasks), device=device), torch.tensor(self.counts, device=device))
        self.target_types = self.fixed_types.clone()
        self.assignment_epoch = torch.zeros_like(self.fixed_types)

    def set_counts(self, counts):
        """阶段切换只更新目标分组，实际任务在各环境下一次 reset 时变更。"""
        self.counts = list(counts)
        if hasattr(self, "target_types"):
            self.target_types[self.env_ids] = torch.repeat_interleave(
                torch.arange(len(self.tasks), device=self.target_types.device), self.target_types.new_tensor(counts))

    def set_env_ids(self, env_ids):
        self.env_ids = env_ids

    def sample_spawn_tiles(self, env_ids):
        self.fixed_types[env_ids] = self.target_types[env_ids]
        types = self.fixed_types[env_ids]
        selected = torch.tensor([self.lookup[i, level] for i, level in enumerate(self.controller.levels)], device=env_ids.device)
        tiles = selected[types]
        self.assignment_epoch[env_ids] = torch.tensor(self.controller.epochs, device=env_ids.device)[types]
        return self.centers[tiles], tiles

    def out_of_bounds(self, xy, centers):
        # 越界由任务判为失败，环境层不另作超时截断。
        return torch.zeros_like(xy[:, 0], dtype=torch.bool)

    def tile_type(self, tile_index):
        task, level = self.variants[int(tile_index)]
        return f"{list(self.tasks)[task]}/{self.stages[int(tile_index)]['name']}"
