"""预构建课程地形；在回合边界整体切换实体和解析几何。"""

from copy import deepcopy

from .config import default_terrain_cfg, resolve_terrain_cfg
from .manager import make_terrain


def merge_terrain_config(base, values):
    """课程使用与 env_cfg.terrain 相同的字段，子地形参数逐项继承。"""
    unknown = set(values).difference({*default_terrain_cfg(), "max_difficulty", "difficulty", "mixture", "adaptive", "level_catalog"})
    if unknown:
        raise KeyError(f"Unsupported terrain curriculum keys: {sorted(unknown)}")
    config = deepcopy(base)
    for name, value in values.items():
        if name == "max_difficulty":
            continue
        if name == "subterrain_parameters":
            parameters = config.setdefault(name, {})
            for preset, overrides in value.items():
                if preset not in default_terrain_cfg()[name]:
                    raise KeyError(f"Unsupported subterrain parameters: {preset}")
                unknown_parameters = set(overrides).difference(default_terrain_cfg()[name][preset])
                if unknown_parameters:
                    raise KeyError(f"Unsupported {preset} parameters: {sorted(unknown_parameters)}")
                parameters.setdefault(preset, {}).update(deepcopy(overrides))
        elif name == "adaptive":
            config.setdefault(name, {}).update(deepcopy(value))
        else:
            config[name] = deepcopy(value)
    return resolve_terrain_cfg(config)


def terrain_course_configs(base, curriculum):
    """按累计覆盖规则解析并校验全部阶段，也支持恢复到较早阶段。"""
    current = resolve_terrain_cfg(base)
    stages = curriculum.get("stages", []) if curriculum.get("enabled", False) else []
    configs = []
    if not stages or stages[0]["start_iteration"] > 0:
        configs.append(current)
    for stage in stages:
        current = merge_terrain_config(current, stage.get("targets", {}).get("terrain", {}))
        make_terrain(current)  # 启动时验证所有阶段尺寸，避免训练到中途才失败。
        if current not in configs:
            configs.append(current)
    return configs


class TerrainCourse:
    """复用固定形状实体；未选地形停放在训练区域之外，不在运行期缩放网格。"""

    def __init__(self, configs):
        self.terrains = [make_terrain(config) for config in configs]
        self.entities = []
        self.home_positions = []
        self.active_index = None
        # 只有一个课程地形处于训练区；停放的地形不增加同时碰撞容量。
        self.max_collision_pairs = max(terrain.max_collision_pairs for terrain in self.terrains)
        # 有限地面保证各组停放实体互不重叠；课程平地同样启用 tile 边界复位。
        self.parking_stride = 2 * max(terrain.tile_size[0] for terrain in self.terrains) + 100.0

    def index(self, config):
        return next(i for i, terrain in enumerate(self.terrains) if terrain.config == config)

    def add_to_scene(self, scene):
        import genesis as gs

        class FiniteGroundScene:
            def __init__(self, terrain):
                self.terrain = terrain

            def add_entity(self, morph, **kwargs):
                if isinstance(morph, gs.morphs.Plane):
                    length, width = self.terrain.tile_size
                    morph = gs.morphs.Box(
                        lower=(-length / 2, -width / 2, -.1),
                        upper=(length / 2, width / 2, 0.),
                        fixed=True, batch_fixed_verts=False,
                    )
                return scene.add_entity(morph, **kwargs)

        for terrain in self.terrains:
            entities = terrain.add_to_scene(FiniteGroundScene(terrain))
            self.entities.append(entities if isinstance(entities, tuple) else (entities,))
        return tuple(entity for group in self.entities for entity in group)

    def bind(self, *, device, dtype, active_config):
        for terrain, entities in zip(self.terrains, self.entities):
            terrain.bind_entity(entities, device=device, dtype=dtype)
            # 保存用户坐标系位置；固定地形在所有并行环境中的位置相同。
            self.home_positions.append(tuple(entity.get_pos()[0].clone() for entity in entities))
        for index in range(len(self.terrains)):
            self._place(index, active=False)
        return self.activate(self.index(active_config))

    def _place(self, index, *, active):
        for entity, home in zip(self.entities[index], self.home_positions[index]):
            pos = home.clone()
            if not active:
                pos[0] += (index + 1) * self.parking_stride
            entity.set_pos(pos, zero_velocity=True)

    def activate(self, index):
        if self.active_index != index:
            if self.active_index is not None:
                self._place(self.active_index, active=False)
            self._place(index, active=True)
            self.active_index = index
        return self.terrains[index]
