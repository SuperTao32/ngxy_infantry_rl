""" 各任务共享的、配置驱动的课程学习调度器。"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Mapping


TargetHandler = Callable[[Mapping[str, Any]], None]


class TaskCurriculumProgress:
    """用累计环境控制步折算任务课程轮数，配额变化只影响后续推进速度。"""

    def __init__(self, config: Mapping[str, Any], *, steps_per_iteration: int | None = None):
        self.reference_num_envs = config.get("reference_num_envs", 4096)
        self.reference_steps_per_iteration = config.get("reference_steps_per_iteration", steps_per_iteration)
        for name in ("reference_num_envs", "reference_steps_per_iteration"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"task curriculum {name} must be a positive integer")
        self.sampled_env_steps = 0
        self.started_at_iteration = None

    @property
    def iteration(self) -> float:
        return self.sampled_env_steps / (self.reference_num_envs * self.reference_steps_per_iteration)

    def advance(self, num_envs: int, *, world_size: int, iteration: int) -> None:
        if num_envs:
            self.mark_allocated(iteration)
            self.sampled_env_steps += num_envs * world_size

    def mark_allocated(self, iteration: int) -> None:
        if self.started_at_iteration is None:
            self.started_at_iteration = iteration

    def state_dict(self) -> dict[str, Any]:
        return dict(reference_num_envs=self.reference_num_envs,
                    reference_steps_per_iteration=self.reference_steps_per_iteration,
                    sampled_env_steps=self.sampled_env_steps, started_at_iteration=self.started_at_iteration)

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        restored = TaskCurriculumProgress(state)
        samples, started = state["sampled_env_steps"], state["started_at_iteration"]
        if type(samples) is not int or samples < 0:
            raise ValueError("sampled_env_steps must be a nonnegative integer")
        if started is not None and (type(started) is not int or started < 0):
            raise ValueError("started_at_iteration must be a nonnegative integer or None")
        self.reference_num_envs = restored.reference_num_envs
        self.reference_steps_per_iteration = restored.reference_steps_per_iteration
        self.sampled_env_steps, self.started_at_iteration = samples, started


class CurriculumManager:
    """根据全局 PPO 轮数或任务样本折算轮数选择阶段，分发合并后的课程参数。

    调度器不了解机器人和 Genesis；环境通过“目标名称 + 处理函数”注册可调项，
    因而新增课程目标时不需要修改阶段选择逻辑。
    """

    def __init__(self, config: Mapping[str, Any] | None = None):
        # 深拷贝避免运行期合并阶段参数时反向修改训练时保存的原始配置。
        self.config = deepcopy(dict(config or {}))
        self.enabled = bool(self.config.get("enabled", False))

        self.stages = deepcopy(list(self.config.get("stages", [])))
        self._validate_stages()
        self._target_handlers: dict[str, TargetHandler] = {}
        self.current_stage_index = -1
        self.current_iteration = 0

    @property
    def current_stage_name(self) -> str:
        if not self.enabled:
            return "disabled"
        if self.current_stage_index < 0:
            return "pending"
        return str(self.stages[self.current_stage_index]["name"])

    def register_target(self, name: str, handler: TargetHandler) -> None:
        """注册一个课程目标，以及将目标参数应用到环境的处理函数。"""
        if name in self._target_handlers:
            raise ValueError(f"Curriculum target is already registered: {name}")
        self._target_handlers[name] = handler

    def update(self, iteration: float, *, force: bool = False) -> bool:
        """检查并应用当前应进入的阶段；返回本次是否发生阶段切换。"""
        if iteration < 0:
            raise ValueError("curriculum iteration cannot be negative")
        self.current_iteration = iteration
        if not self.enabled or not self.stages:
            return False

        next_index = -1
        for index, stage in enumerate(self.stages):
            if int(stage["start_iteration"]) <= iteration:
                next_index = index
            else:
                break

        if next_index == self.current_stage_index and not force:
            return False

        # 阶段采用累计覆盖语义：后续阶段可以只写变化项，未写的参数继承前一阶段。
        merged_targets: dict[str, Any] = {}
        for stage in self.stages[: next_index + 1]:
            _deep_update(merged_targets, stage.get("targets", {}))

        # 在调用任何处理函数前一次性检查目标名，避免只应用一半后才报错。
        unknown = set(merged_targets).difference(self._target_handlers)
        if unknown:
            available = sorted(self._target_handlers)
            raise KeyError(f"Unknown curriculum targets {sorted(unknown)}; registered targets: {available}")

        for name, values in merged_targets.items():
            self._target_handlers[name](deepcopy(values))
        self.current_stage_index = next_index
        return True

    def state_dict(self) -> dict[str, Any]:
        return {
            "current_iteration": self.current_iteration,
            "current_stage_index": self.current_stage_index,
            "current_stage_name": self.current_stage_name,
        }

    def _validate_stages(self) -> None:
        previous_start = -1
        names: set[str] = set()
        for index, stage in enumerate(self.stages):
            if not isinstance(stage, dict):
                raise TypeError(f"curriculum stage {index} must be a mapping")
            if "name" not in stage or "start_iteration" not in stage:
                raise ValueError(f"curriculum stage {index} requires name and start_iteration")
            name = str(stage["name"])
            if name in names:
                raise ValueError(f"duplicate curriculum stage name: {name}")
            names.add(name)
            start_iteration = int(stage["start_iteration"])
            if start_iteration < 0 or start_iteration <= previous_start:
                raise ValueError(
                    "curriculum stage start_iteration values must be non-negative and strictly increasing"
                )
            previous_start = start_iteration
            if not isinstance(stage.get("targets", {}), dict):
                raise TypeError(f"curriculum stage {name!r} targets must be a mapping")


def _deep_update(destination: dict[str, Any], source: Mapping[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, Mapping) and isinstance(destination.get(key), dict):
            _deep_update(destination[key], value)
        else:
            destination[key] = deepcopy(value)
