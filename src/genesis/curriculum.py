""" 配置驱动的环境课程学习调度器。"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Mapping


TargetHandler = Callable[[Mapping[str, Any]], None]


class CurriculumManager:
    """根据全局 PPO 轮数选择阶段，分发合并后的课程参数。

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
        self._target_handlers[name] = handler

    def update(self, iteration: float, *, force: bool = False) -> bool:
        """检查并应用当前应进入的阶段；返回本次是否发生阶段切换。"""
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
        starts = [stage["start_iteration"] for stage in self.stages]
        if any(start < 0 for start in starts) or any(current <= previous for previous, current in zip(starts, starts[1:])):
            raise ValueError("Curriculum stages must have nonnegative, strictly increasing start iterations")


def _deep_update(destination: dict[str, Any], source: Mapping[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, Mapping) and isinstance(destination.get(key), dict):
            _deep_update(destination[key], value)
        else:
            destination[key] = deepcopy(value)
