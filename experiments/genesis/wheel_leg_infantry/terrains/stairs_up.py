"""沿 +x 升高的箱体台阶；与 stairs 的下楼方向相反。"""

import torch

from .stairs import Stairs

PRESET = "stairs_up"
PARAMETER_KEY = "box_stairs_up"
PARAMETERS = dict(step_depth=2.0, step_height=0.20, num_steps=1, approach_length=1.3, base_thickness=0.10)
LEVELS = {
    0: {"step_height": 0.12},
    1: {"step_height": 0.14},
    2: {"step_height": 0.16},
    3: {"step_height": 0.18},
    4: {"step_height": 0.20},
}


class StairsUp(Stairs):
    def _box_bounds(self):
        min_x, min_y, _ = self.origin
        max_x, max_y = min_x + self.tile_size[0], min_y + self.tile_size[1]
        boxes = [((min_x, min_y, -self.base_thickness), (max_x, max_y, 0.0))]
        for index in range(self.num_steps):
            lower_x = self.start_x + index * self.step_depth
            upper_x = max_x if index == self.num_steps - 1 else lower_x + self.step_depth
            boxes.append(((lower_x, min_y, 0.0), (upper_x, max_y, (index + 1) * self.step_height)))
        return tuple(boxes)

    def height_at(self, local_xy):
        index = torch.floor((local_xy[:, 0] - self.start_x) / self.step_depth + 1e-6) + 1
        return index.clamp(0, self.num_steps) * self.step_height + self.origin[2]


TERRAIN_CLASS = StairsUp
