"""三条并排静态通道：平地、20 cm 平台、40 cm 平台。"""

import torch

from ...core.terrain import TerrainManager


class JumpTerrain(TerrainManager):
    def __init__(self, task_cfg):
        super().__init__({"preset": "plane"})
        self.task_cfg = task_cfg

    def add_to_scene(self, scene):
        import genesis as gs

        entities = [scene.add_entity(gs.morphs.Plane())]
        cfg = self.task_cfg
        for mode, height in enumerate(cfg["step_heights_m"]):
            if height == 0:
                continue
            center_y = mode * cfg["lane_spacing_m"]
            entities.append(scene.add_entity(gs.morphs.Box(
                lower=(0.0, center_y - cfg["platform_width_m"] / 2, 0.0),
                upper=(cfg["platform_length_m"], center_y + cfg["platform_width_m"] / 2, height),
                fixed=True, batch_fixed_verts=False,
            )))
        return tuple(entities)

    def sample_spawn_tiles(self, env_ids):
        cfg = self.task_cfg
        if cfg["assignment"] == "cyclic":
            modes = env_ids.to(torch.long) % 3
        else:
            probabilities = torch.tensor(cfg["mode_probabilities"], device=env_ids.device)
            modes = torch.multinomial(probabilities, env_ids.numel(), replacement=True)
        centers = torch.zeros((env_ids.numel(), 2), dtype=self._dtype, device=env_ids.device)
        centers[:, 0] = -cfg["warmup_distance_m"]
        centers[:, 1] = modes * cfg["lane_spacing_m"]
        return centers, modes

    def height_at(self, world_xy):
        cfg = self.task_cfg
        height = torch.zeros_like(world_xy[:, 0])
        for mode, top in enumerate(cfg["step_heights_m"]):
            if top == 0:
                continue
            inside = ((world_xy[:, 0] >= 0) & (world_xy[:, 0] <= cfg["platform_length_m"])
                      & (torch.abs(world_xy[:, 1] - mode * cfg["lane_spacing_m"]) <= cfg["platform_width_m"] / 2))
            height = torch.where(inside, top, height)
        return height

    def tile_type(self, tile_index):
        return self.task_cfg["mode_names"][tile_index]
