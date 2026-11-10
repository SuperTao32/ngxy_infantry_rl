"""两个跳高模式各自按台阶成功率升降级，不依赖仿真。"""

from copy import deepcopy
import math


class JumpHeightCurriculum:
    def __init__(self, task_cfg, options=None):
        self.heights = deepcopy(task_cfg.get("platform_height_levels_m", [[h] for h in task_cfg["step_heights_m"]]))
        if len(self.heights) != len(task_cfg["step_heights_m"]):
            raise ValueError("platform_height_levels_m requires one height list per jump mode")
        for heights, target in zip(self.heights, task_cfg["step_heights_m"]):
            if (not heights or any(isinstance(h, bool) or not math.isfinite(h) or h <= 0 for h in heights)
                    or any(b <= a for a, b in zip(heights, heights[1:])) or heights[-1] != target):
                raise ValueError("platform height levels must be positive, strictly increasing and end at the mode height")
        self.options = {"enabled": False, "window_episodes": 100, "promote_threshold": 0.80, "demote_threshold": 0.40,
                        **(options or {})}
        if set(self.options) - {"enabled", "window_episodes", "promote_threshold", "demote_threshold"}:
            raise ValueError("unknown jump height_curriculum options")
        if type(self.options["enabled"]) is not bool:
            raise ValueError("height_curriculum.enabled must be boolean")
        if type(self.options["window_episodes"]) is not int or self.options["window_episodes"] < 1:
            raise ValueError("height_curriculum.window_episodes must be a positive integer")
        low, high = self.options["demote_threshold"], self.options["promote_threshold"]
        if any(isinstance(v, bool) or not math.isfinite(v) for v in (low, high)) or not 0 <= low < high <= 1:
            raise ValueError("height_curriculum requires 0 <= demote_threshold < promote_threshold <= 1")
        n = len(self.heights)
        self.levels = [0] * n
        self.epochs = [0] * n
        self.counts = [0] * n
        self.successes = [0] * n
        self.rates = [0.0] * n

    def observe(self, mode, level, epoch, successes, count):
        if (not self.options["enabled"] or count == 0
                or level != self.levels[mode] or epoch != self.epochs[mode]):
            return
        self.counts[mode] += count
        self.successes[mode] += successes
        if self.counts[mode] < self.options["window_episodes"]:
            return
        rate = self.successes[mode] / self.counts[mode]
        self.rates[mode] = rate
        if rate >= self.options["promote_threshold"]:
            self.levels[mode] = min(level + 1, len(self.heights[mode]) - 1)
        elif rate <= self.options["demote_threshold"]:
            self.levels[mode] = max(level - 1, 0)
        if self.levels[mode] != level:
            self.epochs[mode] += 1
        self.counts[mode] = self.successes[mode] = 0

    def state_dict(self):
        return deepcopy({"heights": self.heights, "options": self.options,
                         **{key: getattr(self, key) for key in ("levels", "epochs", "counts", "successes", "rates")}})

    def load_state_dict(self, state):
        if state["heights"] != self.heights or state["options"] != self.options:
            raise ValueError("saved jump height curriculum differs; use --load-weights for a new course")
        for key in ("levels", "epochs", "counts", "successes", "rates"):
            if len(state[key]) != len(self.heights):
                raise ValueError(f"invalid jump height curriculum {key}")
        for mode, heights in enumerate(self.heights):
            if type(state["levels"][mode]) is not int or not 0 <= state["levels"][mode] < len(heights):
                raise ValueError("saved jump difficulty outside configured range")
            if any(type(state[key][mode]) is not int or state[key][mode] < 0 for key in ("epochs", "counts", "successes")):
                raise ValueError("invalid jump height curriculum counts")
            if state["successes"][mode] > state["counts"][mode] or not 0 <= state["rates"][mode] <= 1:
                raise ValueError("invalid jump height curriculum success rate")
        for key in ("levels", "epochs", "counts", "successes", "rates"):
            setattr(self, key, deepcopy(state[key]))
