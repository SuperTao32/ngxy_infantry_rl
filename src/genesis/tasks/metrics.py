"""组件独立统计尝试和最终结果，每个 PPO rollout 输出一次。"""

import torch


class TaskMetrics:
    def __init__(self, device, groups=("",)):
        self.groups = groups
        self.counts = {name: torch.zeros(len(groups), device=device) for name in ("attempts", "completed", "successes", "invalid")}

    def clear(self):
        for value in self.counts.values():
            value.zero_()

    def _add(self, name, mask, modes):
        if modes is None:
            self.counts[name][0].add_(mask.sum())
        else:
            self.counts[name].add_(torch.bincount(modes[mask], minlength=len(self.groups)))

    def start(self, mask, modes=None):
        self._add("attempts", mask, modes)

    def finish(self, mask, success, *, invalid=None, modes=None):
        self._add("completed", mask, modes)
        self._add("successes", mask & success, modes)
        if invalid is not None:
            self._add("invalid", mask & invalid, modes)

    def collect(self, prefix):
        values = {name: value.clone() for name, value in self.counts.items()}
        self.clear()
        result = {}
        for index, group in enumerate(self.groups):
            label = f"{prefix}/{group}" if group else prefix
            result.update({f"{label}/{name}": value[index] for name, value in values.items()})
            completed = values["completed"][index]
            # 没有结束尝试时不写成功率，避免把未触发/尚未完成记作失败。
            if completed > 0:
                result[f"{label}/success_rate"] = values["successes"][index] / completed
                result[f"{label}/failure_rate"] = 1.0 - result[f"{label}/success_rate"]
                result[f"{label}/invalid_rate"] = values["invalid"][index] / completed
        return result
