"""不依赖仿真的任务配置解析、固定环境配额和独立成功率课程。"""

from copy import deepcopy
import math

from ...curriculum import CurriculumManager, _deep_update
from ...terrains.config import resolve_terrain_cfg
from ...terrains.registry import TERRAIN_REGISTRY, level_catalog, level_parameters
from .config import TASK_NAMES
from .intermittent import resolve_intermittent
from .parking import resolve_parking
from ...config import validate_high_base_height_velocity_limits


def positive(value, name):
    if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")


def resolve_tasks(config):
    resolve_parking(config.get("parking_curriculum", {}))
    tasks = deepcopy(config["tasks"])
    catalog = deepcopy(config.get("level_catalog")) if "level_catalog" in config else level_catalog()
    if set(tasks) - set(TASK_NAMES):
        raise ValueError("unknown multi_terrain task")
    active = {}
    for name, task in tasks.items():
        if not isinstance(task["enabled"], bool):
            raise ValueError(f"{name}.enabled must be boolean")
        if not math.isfinite(task["weight"]) or task["weight"] < 0:
            raise ValueError(f"{name}.weight must be finite and nonnegative")
        if not task["enabled"] or task["weight"] == 0:
            continue
        course = task["curriculum"]
        if "terrain" not in task or "stages" in course:
            raise ValueError("legacy multi_terrain geometry/stages config; start a new run with --load-weights")
        if not isinstance(course["enabled"], bool):
            raise ValueError(f"{name}: curriculum.enabled must be boolean")
        if type(course["window_episodes"]) is not int or course["window_episodes"] < 1:
            raise ValueError(f"{name}: window_episodes must be a positive integer")
        if not 0 <= course["demote_threshold"] < course["promote_threshold"] <= 1:
            raise ValueError(f"{name}: invalid curriculum thresholds")
        preset = task["terrain"]["preset"]
        if set(task["terrain"]) - {"preset", "tile_size", "horizontal_scale", "vertical_scale", "boundary_margin"}:
            raise ValueError(f"{name}: terrain parameters must come from LEVELS, not task overrides")
        available = catalog.get(preset, {})
        if not available:
            raise ValueError(f"{name}: no LEVELS defined for terrain {preset}")
        low, initial = course["min_difficulty"], course["difficulty"]
        high = course["max_difficulty"]
        if high is None:
            high = max(int(level) for level in available)
        if any(type(level) is not int for level in (low, initial, high)) or not 0 <= low <= initial <= high:
            raise ValueError(f"{name}: require 0 <= min_difficulty <= difficulty <= max_difficulty")
        course["max_difficulty"] = high
        overrides = course.get("overrides", {})
        if not isinstance(overrides, dict) or any(type(level) is not int or not low <= level <= high for level in overrides):
            raise ValueError(f"{name}: curriculum.overrides keys must be difficulty integers within the configured range")
        resolved = []
        current = {k: deepcopy(v) for k, v in task.items() if k not in {"curriculum", "resolved_stages"}}
        for level in range(low, high + 1):
            level_parameters(preset, level, catalog)  # 包括范围中间的等级，不能静默跳过缺级。
            override = overrides.get(level, {})
            if not isinstance(override, dict) or set(override) - {
                "commands", "rewards", "termination", "success", "resampling_time_s", "landing_stability", "ascent", "intermittent"
            }:
                raise ValueError(f"{name}: difficulty overrides only support task behavior, not terrain geometry")
            _deep_update(current, override)
            current["name"] = f"level_{level}"
            current["difficulty"] = level
            current["terrain"] = resolve_terrain_cfg({
                **task["terrain"], "difficulty": level, "level_catalog": catalog, "bounded": True,
                "boundary_margin": task["terrain"].get("boundary_margin", 0.25),
            })
            # 配置快照只保存当前地形参数，避免其他模块的默认参数变化干扰续训比较。
            key = TERRAIN_REGISTRY[preset].PARAMETER_KEY
            parameters = current["terrain"]["subterrain_parameters"]
            current["terrain"]["subterrain_parameters"] = {key: parameters[key]} if key else {}
            validate_stage(name, current)
            resolved.append(deepcopy(current))
        task["resolved_stages"] = resolved
        active[name] = task
    if not active:
        raise ValueError("at least one task must be enabled with positive weight")
    return active


def validate_stage(name, stage):
    validate_high_base_height_velocity_limits(stage["commands"])
    intermittent = resolve_intermittent(stage.get("intermittent", {}), task=name)
    if intermittent["probability"] and any(stage["commands"]["ang_vel_range"]):
        raise ValueError(f"{name}: intermittent traversal requires zero angular commands")
    for key in ("lin_vel_range", "ang_vel_range", "base_height_range"):
        value = stage["commands"][key]
        if len(value) != 2 or not all(math.isfinite(v) for v in value) or value[0] > value[1]:
            raise ValueError(f"{name}: invalid commands.{key}")
    if stage["commands"].get("num_commands", 3) != 3:
        raise ValueError("task commands must retain the three-component policy interface")
    if not 0 <= stage["commands"].get("standing_probability", 0) <= 1:
        raise ValueError("standing_probability must be in [0, 1]")
    if stage["commands"]["base_height_range"][0] <= 0:
        raise ValueError("base_height_range must be positive")
    for key, value in stage["rewards"].items():
        if not math.isfinite(value):
            raise ValueError(f"{name}: non-finite reward {key}")
    landing = stage.get("landing_stability", {})
    if not isinstance(landing, dict) or set(landing) - {"duration_s", "ang_vel_weight"}:
        raise ValueError(f"{name}: invalid landing_stability options")
    positive(landing.get("duration_s", 0.3), f"{name}.landing_stability.duration_s")
    angular_weight = landing.get("ang_vel_weight", 0.25)
    if isinstance(angular_weight, bool) or not math.isfinite(angular_weight) or angular_weight < 0:
        raise ValueError(f"{name}.landing_stability.ang_vel_weight must be finite and nonnegative")
    landing_scale = stage["rewards"].get("stairs_down_landing_stability", 0.0)
    if landing_scale > 0 or (landing_scale != 0 and (name != "stairs_down" or stage["terrain"]["preset"] != "stairs")):
        raise ValueError("stairs_down_landing_stability requires stairs_down on stairs and a nonpositive weight")
    ascent = stage.get("ascent", {})
    ascent_keys = {
        "approach_distance_m", "approach_clearance_m",
        "backswing_angle_deg", "forward_swing_angle_deg", "landing_extension_m",
        "support_duration_s", "support_force_n", "support_normal_z", "support_surface_tolerance_m",
        "support_margin_m", "retract_shortening_m", "wheel_edge_margin_m", "wheel_bottom_tolerance_m",
        "fall_height_tolerance_m",
        "wall_normal_z_max", "wall_surface_tolerance_m",
        "success_duration_s", "standing_height_m", "max_vertical_speed_m_s", "max_roll_pitch_speed_rad_s",
    }
    if not isinstance(ascent, dict) or set(ascent) - ascent_keys:
        raise ValueError(f"{name}: invalid ascent options")
    if ascent and (name != "stairs_up" or stage["terrain"]["preset"] != "stairs_up"):
        raise ValueError("ascent requires stairs_up on stairs_up terrain")
    for key, value in ascent.items():
        if key == "approach_distance_m":
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name}.ascent.{key} must be finite and nonnegative")
        else:
            positive(value, f"{name}.ascent.{key}")
    if ascent.get("support_normal_z", 0.9) > 1:
        raise ValueError("ascent.support_normal_z must be <= 1")
    if ascent.get("wall_normal_z_max", 0.3) >= 1:
        raise ValueError("ascent.wall_normal_z_max must be < 1")
    if ascent.get("backswing_angle_deg", 20.0) >= 90:
        raise ValueError("ascent.backswing_angle_deg must be < 90")
    if ascent.get("forward_swing_angle_deg", 20.0) >= 90:
        raise ValueError("ascent.forward_swing_angle_deg must be < 90")
    for key in ("stairs_up_support", "stairs_up_retract", "stairs_up_takeoff", "stairs_up_landing_distance",
                "stairs_up_approach_lift", "stairs_up_backswing", "stairs_up_forward_swing",
                "stairs_up_extend", "stairs_up_wheel_support", "stairs_up_wheel_forward_action"):
        scale = stage["rewards"].get(key, 0.0)
        if scale < 0 or (scale != 0 and (name != "stairs_up" or stage["terrain"]["preset"] != "stairs_up")):
            raise ValueError(f"{key} requires stairs_up on stairs_up terrain and a nonnegative weight")
    wall_scale = stage["rewards"].get("stairs_up_wheel_wall_force", 0.0)
    if wall_scale > 0 or (wall_scale != 0 and (name != "stairs_up" or stage["terrain"]["preset"] != "stairs_up")):
        raise ValueError("stairs_up_wheel_wall_force requires stairs_up on stairs_up terrain and a nonpositive weight")
    if stage["terrain"]["preset"] == "plane":
        raise ValueError(f"{name}: traversal task requires a non-plane terrain")
    if stage["rewards"].get("alive", 0) != 0:
        raise ValueError(f"{name}: traversal tasks cannot use alive reward")
    for key in ("task_progress", "task_success"):
        positive(stage["rewards"].get(key, 0), f"{name}.rewards.{key}")
    if stage["commands"]["lin_vel_range"][0] <= 0:
        raise ValueError(f"{name}: traversal commands must move forward")
    for key, value in stage["success"].items():
        positive(value, f"{name}.success.{key}")
    for key, value in stage["termination"].items():
        positive(value, f"{name}.termination.{key}")
    positive(stage["resampling_time_s"], f"{name}.resampling_time_s")


def allocate_counts(tasks, num_envs, weights=None):
    """最大余数分配；避免静默舍弃比重过小的已启用任务。"""
    if type(num_envs) is not int or num_envs < 1:
        raise ValueError("num_envs must be a positive integer")
    weights = {} if weights is None else weights
    if not isinstance(weights, dict) or set(weights) - set(TASK_NAMES):
        raise ValueError("task_weights must map known task names to weights")
    for value in weights.values():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError("task_weights must be finite and nonnegative")
    # enabled=False / 初始 weight=0 的任务未构建，课程不能重新启用它们。
    values = [weights.get(name, task["weight"]) for name, task in tasks.items()]
    total = sum(values)
    if total <= 0:
        raise ValueError("task_weights must retain at least one task")
    quotas = [num_envs * value / total for value in values]
    counts = [math.floor(q) for q in quotas]
    order = sorted(range(len(counts)), key=lambda i: quotas[i] - counts[i], reverse=True)
    for i in order[:num_envs - sum(counts)]:
        counts[i] += 1
    if any(n == 0 and value > 0 for n, value in zip(counts, values)):
        raise ValueError("num_envs is too small for the task weights; increase it or disable tasks")
    return counts


def resolve_task_command_ranges(tasks, overrides):
    """合并全局阶段命令覆盖；保留任务/难度快照，顺序与 TaskTerrain.stages 一致。"""
    if not isinstance(overrides, dict) or set(overrides) - set(TASK_NAMES):
        raise ValueError("task_command_ranges must map known task names to command ranges")
    allowed = {"lin_vel_range", "ang_vel_range", "base_height_range", "standing_probability"}
    for name, values in overrides.items():
        if not isinstance(values, dict) or set(values) - allowed:
            raise ValueError(f"{name}: task_command_ranges only supports {sorted(allowed)}")
        for key, value in values.items():
            if key == "standing_probability":
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError(f"{name}: standing_probability must be finite and in [0, 1]")
            elif (not isinstance(value, (list, tuple)) or len(value) != 2
                  or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in value)
                  or value[0] > value[1]):
                raise ValueError(f"{name}: invalid commands.{key}")
            elif key == "base_height_range" and value[0] <= 0:
                raise ValueError(f"{name}: base_height_range must be positive")
            elif key == "lin_vel_range" and value[0] <= 0:
                raise ValueError(f"{name}: traversal commands must move forward")
    commands = []
    for name, task in tasks.items():
        for stage in task["resolved_stages"]:
            cfg = deepcopy(stage["commands"])
            cfg.update(deepcopy(overrides.get(name, {})))
            validate_stage(name, {**stage, "commands": cfg})
            high_speed = cfg.get("high_speed_ang_vel")
            if high_speed is not None:
                lower, upper = cfg["ang_vel_range"]
                limit = high_speed["max_abs_ang_vel"]
                if max(lower, -limit) > min(upper, limit):
                    raise ValueError(f"{name}: ang_vel_range must overlap the high-speed angular velocity limits")
            commands.append(cfg)
    return commands


def initial_task_counts(tasks, num_envs, curriculum_cfg=None):
    """场景构建前校验全部阶段配额与命令，返回第 0 轮的任务分配。"""
    curriculum = CurriculumManager(curriculum_cfg)
    initial = allocate_counts(tasks, num_envs)
    allowed_targets = {"domain_rand", "sensor_noise", "wheel_slip", "task_weights", "task_command_ranges"}
    weights = {}
    command_ranges = {}
    for stage in curriculum.stages:
        targets = stage.get("targets", {})
        if set(targets) - allowed_targets:
            raise ValueError("multi_terrain global curriculum only controls domain_rand/sensor_noise/wheel_slip/task_weights/task_command_ranges")
        changes = targets.get("task_weights", {})
        if not isinstance(changes, dict):
            raise ValueError("task_weights must map known task names to weights")
        weights.update(changes)
        counts = allocate_counts(tasks, num_envs, weights)
        changes = targets.get("task_command_ranges", {})
        resolve_task_command_ranges(tasks, changes)
        _deep_update(command_ranges, changes)
        resolve_task_command_ranges(tasks, command_ranges)
        if curriculum.enabled and stage["start_iteration"] == 0:
            initial = counts
    return initial


class TaskCurriculum:
    """各任务独立统计成功率、升降级和保存状态；旧回合不污染新等级窗口。"""

    def __init__(self, tasks):
        self.tasks = deepcopy(tasks)
        self.task_options = [task["curriculum"] for task in self.tasks.values()]
        self.entries = [dict(preset=name, difficulty=task["curriculum"]["difficulty"],
                        min_difficulty=task["curriculum"]["min_difficulty"],
                        max_difficulty=task["curriculum"]["max_difficulty"])
                   for name, task in self.tasks.items()]
        self.levels = [entry["difficulty"] for entry in self.entries]
        self.counts = [0] * len(self.entries)
        self.successes = [0] * len(self.entries)
        self.epochs = [0] * len(self.entries)
        self.rates = [0.0] * len(self.entries)

    def observe(self, task_type, level, epoch, successes, count):
        i = task_type
        options = self.task_options[i]
        if not options["enabled"] or level != self.levels[i] or epoch != self.epochs[i] or count == 0:
            return
        self.counts[i] += count
        self.successes[i] += successes
        if self.counts[i] < options["window_episodes"]:
            return
        rate = self.successes[i] / self.counts[i]
        self.rates[i] = rate
        entry = self.entries[i]
        if rate >= options["promote_threshold"]:
            self.levels[i] = min(level + 1, entry["max_difficulty"])
        elif rate <= options["demote_threshold"]:
            self.levels[i] = max(level - 1, entry["min_difficulty"])
        if self.levels[i] != level:
            self.epochs[i] += 1
        self.counts[i] = self.successes[i] = 0

    def state_dict(self):
        # 保持已有 multi_terrain 检查点字段；options 原本就是空字典。
        return deepcopy({"tasks": self.tasks, "entries": self.entries, "options": {},
                         **{key: getattr(self, key) for key in ("levels", "counts", "successes", "epochs", "rates")}})

    def load_state_dict(self, state):
        if state.get("tasks") != self.tasks:
            raise ValueError("saved task curriculum differs; use --load-weights for a new course")
        if state["entries"] != self.entries or state["options"] != {}:
            raise ValueError("checkpoint terrain curriculum differs from current configuration")
        for key in ("levels", "counts", "successes", "epochs", "rates"):
            values = state[key]
            if len(values) != len(self.entries):
                raise ValueError(f"invalid terrain checkpoint {key}")
        for i, entry in enumerate(self.entries):
            if type(state["levels"][i]) is not int or not entry["min_difficulty"] <= state["levels"][i] <= entry["max_difficulty"]:
                raise ValueError("checkpoint difficulty outside configured range")
            if any(type(state[key][i]) is not int or state[key][i] < 0 for key in ("counts", "successes", "epochs")):
                raise ValueError("invalid terrain checkpoint counts")
            if state["successes"][i] > state["counts"][i] or not 0 <= state["rates"][i] <= 1:
                raise ValueError("invalid terrain checkpoint success rate")
        for key in ("levels", "counts", "successes", "epochs", "rates"):
            setattr(self, key, deepcopy(state[key]))
