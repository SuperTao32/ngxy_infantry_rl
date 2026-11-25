"""按累计课程权重选择本阶段需要构建的地形。"""


def terrain_weights_at_iteration(terrain_config, curriculum_config, iteration):
    """与 CurriculumManager 一致：阶段间累计覆盖，首次覆盖未列出的类型为零。"""
    if terrain_config["preset"] != "mixture":
        return None
    weights = None
    if curriculum_config and curriculum_config.get("enabled", False):
        for stage in curriculum_config.get("stages", ()):
            if stage["start_iteration"] > iteration:
                break
            overrides = stage.get("targets", {}).get("terrain_weights")
            if overrides is not None:
                if weights is None:
                    weights = {}
                weights.update(overrides)
    if weights is None:
        weights = {}
        for tile in terrain_config["tiles"]:
            name = tile["preset"]
            weights[name] = weights.get(name, 0.0) + float(tile.get("weight", 1.0))
    return weights


def active_terrain_types(weights):
    return frozenset(name for name, weight in weights.items() if float(weight) > 0)
