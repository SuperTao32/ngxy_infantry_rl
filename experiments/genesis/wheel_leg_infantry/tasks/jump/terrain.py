"""三条并排通道：课程可移走/放回台阶，保持任务通道编码不变。"""

import torch

from ...core.terrain import TerrainManager
from .config import validate_platform_enabled
from .geometry import active_step_heights


class JumpTerrain(TerrainManager):
    def __init__(self, task_cfg):
        super().__init__({"preset": "plane"})
        self.task_cfg = task_cfg
        self.platform_entities = {}

    def add_to_scene(self, scene):
        import genesis as gs

        entities = [scene.add_entity(gs.morphs.Plane())]
        cfg = self.task_cfg
        for mode, height in enumerate(cfg["step_heights_m"]):
            if height == 0:
                continue
            center = self._platform_center(mode)
            entity = scene.add_entity(gs.morphs.Box(
                pos=center,
                size=(cfg["platform_length_m"], cfg["platform_width_m"], height),
                fixed=True, batch_fixed_verts=False,
            ))
            self.platform_entities[mode] = entity
            entities.append(entity)
        return tuple(entities)

    def _platform_center(self, mode):
        cfg = self.task_cfg
        height = cfg["step_heights_m"][mode]
        enabled = cfg.get("platform_enabled", [True, True, True])[mode]
        # 移至地下使整个箱体远离轮子；放回时恢复台面高度，不重建场景。
        center_z = height / 2 if enabled else -1.0 - height / 2
        return (cfg["platform_length_m"] / 2, mode * cfg["lane_spacing_m"], center_z)

    def set_platform_enabled(self, values):
        validate_platform_enabled(values)
        previous = self.task_cfg.get("platform_enabled", [True, True, True])
        self.task_cfg["platform_enabled"] = list(values)
        for mode, entity in self.platform_entities.items():
            if values[mode] != previous[mode]:
                # 同时移动所有并行环境中的共享固定几何。
                entity.set_pos(self._platform_center(mode), relative=False)

    def sample_spawn_tiles(self, env_ids):
        cfg = self.task_cfg
        if cfg["assignment"] == "cyclic":
            modes = env_ids.to(torch.long) % 3
        else:
            probabilities = torch.tensor(cfg["mode_probabilities"], dtype=self._dtype or torch.float32, device=env_ids.device)
            modes = torch.multinomial(probabilities, env_ids.numel(), replacement=True)
        centers = torch.zeros((env_ids.numel(), 2), dtype=self._dtype or torch.float32, device=env_ids.device)
        centers[:, 0] = -cfg["warmup_distance_m"]
        centers[:, 1] = modes * cfg["lane_spacing_m"]
        return centers, modes

    def height_at(self, world_xy):
        cfg = self.task_cfg
        height = torch.zeros_like(world_xy[:, 0])
        for mode, top in enumerate(active_step_heights(cfg)):
            if top == 0:
                continue
            inside = ((world_xy[:, 0] >= 0) & (world_xy[:, 0] <= cfg["platform_length_m"])
                      & (torch.abs(world_xy[:, 1] - mode * cfg["lane_spacing_m"]) <= cfg["platform_width_m"] / 2))
            height = torch.where(inside, top, height)
        return height

    def tile_type(self, tile_index):
        return self.task_cfg["mode_names"][tile_index]
