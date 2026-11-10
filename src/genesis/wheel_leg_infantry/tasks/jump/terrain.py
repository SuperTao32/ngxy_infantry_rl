"""固定的多高度台阶通道；课程只改变新回合的通道分配。"""

import torch

from ...terrains.manager import TerrainManager
from .config import get_approach_cfg, validate_platform_enabled
from .curriculum import JumpHeightCurriculum
from .geometry import jump_lane_y, terrain_tile_heights, terrain_variants


class JumpTerrain(TerrainManager):
    def __init__(self, task_cfg, approach_cfg=None, *, height_curriculum=None, curriculum_active=False):
        super().__init__({"preset": "plane"})
        self.task_cfg = task_cfg
        self.approach_cfg = get_approach_cfg() if approach_cfg is None else approach_cfg
        self.variants = terrain_variants(task_cfg)
        self.tile_size = (40.0, len(self.variants) * task_cfg["lane_spacing_m"])
        self.controller = JumpHeightCurriculum(task_cfg, height_curriculum)
        self.curriculum_active = curriculum_active and self.controller.options["enabled"]
        self.lookup = {(mode, height): tile for tile, (mode, height) in enumerate(self.variants)}
        self.platform_entities = {}
        self.cyclic_mode_offset = 0

    def add_to_scene(self, scene):
        import genesis as gs

        entities = [scene.add_entity(gs.morphs.Plane())]
        cfg = self.task_cfg
        for tile, (_, height) in enumerate(self.variants):
            if height == 0:
                continue
            center = self._platform_center(tile)
            entity = scene.add_entity(gs.morphs.Box(
                pos=center,
                size=(cfg["platform_length_m"], cfg["platform_width_m"], height),
                fixed=True, batch_fixed_verts=False,
            ))
            self.platform_entities[tile] = entity
            entities.append(entity)
        return tuple(entities)

    def _platform_center(self, tile):
        cfg = self.task_cfg
        mode, height = self.variants[tile]
        enabled = cfg.get("platform_enabled", [True] * len(cfg["mode_names"]))[mode]
        # 移至地下使整个箱体远离轮子；放回时恢复台面高度，不重建场景。
        center_z = height / 2 if enabled else -1.0 - height / 2
        return (cfg["platform_length_m"] / 2, jump_lane_y(tile, cfg), center_z)

    def set_platform_enabled(self, values):
        validate_platform_enabled(values)
        previous = self.task_cfg.get("platform_enabled", [True] * len(self.task_cfg["mode_names"]))
        self.task_cfg["platform_enabled"] = list(values)
        for tile, entity in self.platform_entities.items():
            mode, _ = self.variants[tile]
            if values[mode] != previous[mode]:
                # 同时移动所有并行环境中的共享固定几何。
                entity.set_pos(self._platform_center(tile), relative=False)

    def sample_spawn_tiles(self, env_ids):
        cfg = self.task_cfg
        if cfg["assignment"] == "cyclic":
            # eval 在批次边界轮换；同一批预热中的局部 reset 保持模式不变。
            modes = (env_ids.to(torch.long) + self.cyclic_mode_offset) % len(cfg["mode_names"])
        else:
            probabilities = torch.tensor(cfg["mode_probabilities"], dtype=self._dtype or torch.float32, device=env_ids.device)
            modes = torch.multinomial(probabilities, env_ids.numel(), replacement=True)
        platform_probabilities = torch.tensor(cfg["platform_probabilities"], dtype=self._dtype or torch.float32, device=env_ids.device)
        enabled = torch.tensor(cfg.get("platform_enabled", [True] * len(cfg["mode_names"])), device=env_ids.device)
        has_platform = (torch.rand(env_ids.numel(), device=env_ids.device) < platform_probabilities[modes]) & enabled[modes]
        heights = [self.controller.heights[mode][level] for mode, level in enumerate(self.controller.levels)]
        if not self.curriculum_active:
            heights = cfg["step_heights_m"]
        step_tiles = torch.tensor([self.lookup[mode, h] for mode, h in enumerate(heights)], device=env_ids.device)
        tiles = torch.where(has_platform, step_tiles[modes], 2 * modes)
        centers = torch.zeros((env_ids.numel(), 2), dtype=self._dtype or torch.float32, device=env_ids.device)
        # 出生点与平台通道分开，先在没有障碍物的平地实际加速。
        centers[:, 0] = self.approach_cfg["spawn_x_m"]
        centers[:, 1] = self.approach_cfg["spawn_y_m"]
        return centers, tiles

    def height_at(self, world_xy):
        cfg = self.task_cfg
        height = torch.zeros_like(world_xy[:, 0])
        for tile, top in enumerate(terrain_tile_heights(cfg)):
            if top == 0:
                continue
            inside = ((world_xy[:, 0] >= 0) & (world_xy[:, 0] <= cfg["platform_length_m"])
                      & (torch.abs(world_xy[:, 1] - jump_lane_y(tile, cfg)) <= cfg["platform_width_m"] / 2))
            height = torch.where(inside, top, height)
        return height

    def tile_type(self, tile_index):
        mode, height = self.variants[int(tile_index)]
        enabled = self.task_cfg.get("platform_enabled", [True] * len(self.task_cfg["mode_names"]))[mode]
        terrain = "step" if height > 0 and enabled else "flat"
        return f"{self.task_cfg['mode_names'][mode]}/{terrain}"
