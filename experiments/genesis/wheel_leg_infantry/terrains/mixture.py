"""按回合采样多种地形；每种地形独立按成功率升降级。"""

from copy import deepcopy
import math

import torch

from .config import default_terrain_cfg, resolve_terrain_cfg
from .registry import level_catalog, level_parameters
from .manager import TerrainManager


ADAPTIVE_DEFAULTS = {
    "enabled": True,
    "window_episodes": 100,
    "promote_threshold": 0.80,
    "demote_threshold": 0.30,
    "min_distance": 2.0,             # 出生点到结束点的水平净位移，m
    "min_duration_s": 2.0,
    "max_lin_vel_rmse": 0.5,         # 机身前向速度跟踪，m/s
    "max_ang_vel_rmse": 1.0,         # yaw 角速度跟踪，rad/s
    "max_tilt_deg": 60.0,
}


def resolve_mixture_cfg(raw):
    common = deepcopy(raw)
    entries = common.pop("mixture")
    adaptive = common.pop("adaptive", {})
    catalog = common.pop("level_catalog", None)
    catalog = level_catalog() if catalog is None else deepcopy(catalog)
    if not isinstance(entries, list) or not entries:
        raise ValueError("terrain.mixture must be a nonempty list")
    if not isinstance(adaptive, dict) or set(adaptive) - set(ADAPTIVE_DEFAULTS):
        raise ValueError("unknown terrain.adaptive fields")
    adaptive = {**ADAPTIVE_DEFAULTS, **adaptive}
    if not isinstance(adaptive["enabled"], bool):
        raise ValueError("adaptive.enabled must be boolean")
    window = adaptive["window_episodes"]
    if isinstance(window, bool) or not isinstance(window, int) or window < 1:
        raise ValueError("window_episodes must be a positive integer")
    if not 0 <= adaptive["demote_threshold"] < adaptive["promote_threshold"] <= 1:
        raise ValueError("require 0 <= demote_threshold < promote_threshold <= 1")
    for key in ("min_distance", "min_duration_s", "max_lin_vel_rmse", "max_ang_vel_rmse", "max_tilt_deg"):
        if not math.isfinite(adaptive[key]) or adaptive[key] <= 0:
            raise ValueError(f"adaptive.{key} must be positive and finite")
    normalized, names = [], set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) - {"preset", "weight", "difficulty", "min_difficulty", "max_difficulty"}:
            raise ValueError("invalid terrain.mixture entry")
        preset = entry.get("preset")
        if preset in names:
            raise ValueError(f"duplicate mixture preset: {preset}")
        names.add(preset)
        initial = entry.get("difficulty", 0)
        low, high = entry.get("min_difficulty", initial), entry.get("max_difficulty", initial)
        for value in (low, initial, high):
            level_parameters(preset, value, catalog)
        if not low <= initial <= high:
            raise ValueError("require min_difficulty <= difficulty <= max_difficulty")
        for level in range(low, high + 1):
            level_parameters(preset, level, catalog)
        weight = float(entry.get("weight", 1.0))
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("mixture weight must be finite and nonnegative")
        normalized.append(dict(preset=preset, weight=weight, difficulty=initial,
                               min_difficulty=low, max_difficulty=high))
    if sum(e["weight"] for e in normalized) <= 0:
        raise ValueError("at least one mixture weight must be positive")
    # 公共几何与单地形使用同一套校验；mixture 不是可被 --terrain 选择的旧 mixed 预设。
    common.pop("difficulty", None)
    common["preset"] = "plane"
    common["bounded"] = True
    common = resolve_terrain_cfg(common)
    if common["boundary_margin"] is None:
        common["boundary_margin"] = min(.75, min(common["tile_size"]) / 4)
    common.update(mixture=normalized, adaptive=adaptive, level_catalog=catalog)
    return common


class SuccessRateCurriculum:
    """窗口按地形分开；每批结束回合最多升降一级，等级变化后清空窗口。"""

    def __init__(self, config):
        self.entries = deepcopy(config["mixture"])
        self.options = deepcopy(config["adaptive"])
        self.levels = [entry["difficulty"] for entry in self.entries]
        self.counts = [0] * len(self.entries)
        self.successes = [0] * len(self.entries)
        self.epochs = [0] * len(self.entries)
        self.rates = [0.0] * len(self.entries)

    def observe(self, terrain_type, level, epoch, successes, count):
        i = terrain_type
        if not self.options["enabled"] or level != self.levels[i] or epoch != self.epochs[i] or count == 0:
            return
        self.counts[i] += count
        self.successes[i] += successes
        if self.counts[i] < self.options["window_episodes"]:
            return
        rate = self.successes[i] / self.counts[i]
        self.rates[i] = rate
        entry = self.entries[i]
        if rate >= self.options["promote_threshold"]:
            self.levels[i] = min(level + 1, entry["max_difficulty"])
        elif rate <= self.options["demote_threshold"]:
            self.levels[i] = max(level - 1, entry["min_difficulty"])
        if self.levels[i] != level:
            self.epochs[i] += 1
        self.counts[i] = self.successes[i] = 0

    def state_dict(self):
        return deepcopy({"entries": self.entries, "options": self.options,
                         **{key: getattr(self, key) for key in ("levels", "counts", "successes", "epochs", "rates")}})

    def load_state_dict(self, state):
        if state["entries"] != self.entries or state["options"] != self.options:
            raise ValueError("checkpoint terrain curriculum differs from current configuration")
        for key in ("levels", "counts", "successes", "epochs", "rates"):
            values = state[key]
            if len(values) != len(self.entries):
                raise ValueError(f"invalid terrain checkpoint {key}")
        for i, entry in enumerate(self.entries):
            if not entry["min_difficulty"] <= state["levels"][i] <= entry["max_difficulty"]:
                raise ValueError("checkpoint difficulty outside configured range")
        for key in ("levels", "counts", "successes", "epochs", "rates"):
            setattr(self, key, deepcopy(state[key]))


class MixedTerrain:
    """各类型/等级为互不重叠的有限 patch；只在 reset 时改变机器人出生 patch。

    静态碰撞体在所有并行环境中共享，每个机器人可独立分配类型和难度，
    无需运行期修改网格或逐环境移动固定实体。
    """

    is_plane = False
    is_platform_ridge = False
    preset = "mixture"

    def __init__(self, config):
        self.config = resolve_mixture_cfg(config)
        self.tile_size = tuple(self.config["tile_size"])
        self.controller = SuccessRateCurriculum(self.config)
        self.terrains, self.variants = [], []
        self.lookup = {}
        for i, entry in enumerate(self.config["mixture"]):
            for level in range(entry["min_difficulty"], entry["max_difficulty"] + 1):
                single = {key: deepcopy(value) for key, value in self.config.items() if key not in {"mixture", "adaptive"}}
                single.update(preset=entry["preset"], difficulty=level)
                self.lookup[i, level] = len(self.terrains)
                self.terrains.append(TerrainManager(single))
                self.variants.append((i, level))
        self.stride = self.tile_size[1] + 4.0
        # Genesis 的碰撞容量按单个并行环境计；机器人只在一个隔离 patch 内活动。
        # 预构建更多类型/等级不增加单环境同时接触的地形数量。
        self.max_collision_pairs = max(t.max_collision_pairs for t in self.terrains)
        self.entities = []

    def add_to_scene(self, scene):
        import genesis as gs

        class OffsetScene:
            def __init__(self, offset, tile_size):
                self.offset, self.tile_size = offset, tile_size

            def add_entity(self, morph, **kwargs):
                if isinstance(morph, gs.morphs.Plane):
                    x, y = self.tile_size
                    morph = gs.morphs.Box(lower=(-x / 2, -y / 2, -.1), upper=(x / 2, y / 2, 0.), fixed=True, batch_fixed_verts=False)
                updates = {"pos": tuple(a + b for a, b in zip(morph.pos, self.offset))}
                if isinstance(morph, gs.morphs.Box):
                    for name in ("lower", "upper"):
                        updates[name] = tuple(a + b for a, b in zip(getattr(morph, name), self.offset))
                return scene.add_entity(morph.model_copy(update=updates), **kwargs)

        for index, terrain in enumerate(self.terrains):
            entities = terrain.add_to_scene(OffsetScene((0., index * self.stride, 0.), self.tile_size))
            self.entities.append(entities if isinstance(entities, tuple) else (entities,))
        return tuple(entity for group in self.entities for entity in group)

    def bind_entity(self, entity, *, device, dtype):
        for terrain, group in zip(self.terrains, self.entities):
            terrain.bind_entity(group, device=device, dtype=dtype)
        self.centers = torch.zeros((len(self.terrains), 2), device=device, dtype=dtype)
        self.centers[:, 1] = torch.arange(len(self.terrains), device=device) * self.stride
        self.weights = torch.tensor([e["weight"] for e in self.config["mixture"]], device=device, dtype=dtype)
        self.assignment_epoch = None

    def sample_spawn_tiles(self, env_ids):
        types = torch.multinomial(self.weights, len(env_ids), replacement=True)
        selected = torch.tensor([self.lookup[i, level] for i, level in enumerate(self.controller.levels)], device=env_ids.device)
        tiles = selected[types]
        if self.assignment_epoch is None:
            # 环境初始化总是先 reset 全体，后续只更新已结束的子环境。
            self.assignment_epoch = torch.zeros(int(env_ids.max()) + 1, device=env_ids.device, dtype=torch.long)
        epochs = torch.tensor(self.controller.epochs, device=env_ids.device)
        self.assignment_epoch[env_ids] = epochs[types]
        return self.centers[tiles], tiles

    def adjust_spawn(self, positions, rpy, tiles):
        for index, terrain in enumerate(self.terrains):
            terrain.adjust_spawn(positions, rpy, mask=tiles == index, offset=self.centers[index])

    def height_at(self, world_xy):
        indices = torch.round(world_xy[:, 1] / self.stride).long()
        heights = torch.zeros_like(world_xy[:, 0])
        for index, terrain in enumerate(self.terrains):
            local = world_xy - self.centers[index]
            heights = torch.where(indices == index, terrain.height_at(local), heights)
        return heights

    def out_of_bounds(self, world_xy, spawn_centers):
        extent = world_xy.new_tensor(self.tile_size) / 2 - self.config["boundary_margin"]
        return ((world_xy - spawn_centers).abs() >= extent).any(dim=-1)

    def reset(self, env_ids):
        pass

    def tile_type(self, tile_index):
        i, level = self.variants[int(tile_index)]
        return f'{self.config["mixture"][i]["preset"]}/level_{level}'

    def record(self, env_ids, tile_indices, success):
        # 一次 CPU 传输汇总，避免每种地形逐次同步 GPU；旧等级的在途回合不污染新窗口。
        rows = torch.stack((tile_indices, self.assignment_epoch[env_ids], success.long()), dim=-1).cpu().tolist()
        counts = {}
        for tile, epoch, passed in rows:
            i, level = self.variants[tile]
            count, wins = counts.get((i, level, epoch), (0, 0))
            counts[i, level, epoch] = (count + 1, wins + passed)
        for (i, level, epoch), (count, wins) in counts.items():
            self.controller.observe(i, level, epoch, wins, count)

    def metrics(self):
        return {
            f"terrain/{entry['preset']}/{key}": float(values[i])
            for i, entry in enumerate(self.controller.entries)
            for key, values in (("difficulty", self.controller.levels), ("success_rate", self.controller.rates),
                                ("window_episodes", self.controller.counts))
        }
