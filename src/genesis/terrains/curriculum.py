"""按地形独立升降级；只消费当前等级已完成的越障结果。"""

from copy import deepcopy


def default_difficulty_curriculum_cfg():
    return {"enabled": False, "min_completed": 100, "promote_threshold": 0.8, "demote_threshold": 0.2}


class TerrainDifficultyCurriculum:
    def __init__(self, config, level_sets, initial_levels):
        self.config = deepcopy(config)
        self.enabled = config["enabled"]
        self.level_sets = level_sets
        self.levels = dict(initial_levels)
        self.windows = {name: {"completed": 0, "successes": 0} for name in initial_levels}

    def update(self, results):
        """results 为 (地形, 等级, 完成数, 成功数)；一批最多升降一级。"""
        if not self.enabled:
            return False
        for name, level, completed, successes in results:
            if level == self.levels[name]:
                self.windows[name]["completed"] += completed
                self.windows[name]["successes"] += successes
        changed = False
        for name, window in self.windows.items():
            if window["completed"] < self.config["min_completed"]:
                continue
            rate = window["successes"] / window["completed"]
            levels = self.level_sets[name]
            index = levels.index(self.levels[name])
            if rate >= self.config["promote_threshold"]:
                index = min(index + 1, len(levels) - 1)
            elif rate <= self.config["demote_threshold"]:
                index = max(index - 1, 0)
            changed |= self.levels[name] != levels[index]
            self.levels[name] = levels[index]
            window.update(completed=0, successes=0)
        return changed

    def metrics(self):
        result = {}
        for name, level in self.levels.items():
            prefix = f"terrain_curriculum/{name}"
            result[f"{prefix}/level"] = float(level)
            result[f"{prefix}/window_completed"] = float(self.windows[name]["completed"])
        return result

    def state_dict(self):
        return deepcopy({"levels": self.levels, "windows": self.windows})

    def load_state_dict(self, state):
        for name, levels in self.level_sets.items():
            if name not in state["levels"]:
                continue
            level = state["levels"][name]
            if level in levels:
                self.levels[name] = level
                self.windows[name] = deepcopy(state["windows"][name])
