"""任务注册、固定 batch 分组、任务 hook 分发和课程状态管理。"""

from importlib import import_module
import math

import torch


TASK_NAMES = ("locomotion", "recovery", "jump", "multi_terrain")
TASK_CLASSES = {"locomotion": "LocomotionTask", "recovery": "RecoveryTask",
                "jump": "JumpTask", "multi_terrain": "MultiTerrainTask"}


def task_class(name):
    if name not in TASK_CLASSES:
        raise ValueError(f"Unknown task {name!r}; expected one of {TASK_NAMES}")
    return getattr(import_module(f"{__package__}.{name}.task"), TASK_CLASSES[name])


def task_configs(name):
    if name not in TASK_CLASSES:
        raise ValueError(f"Unknown task {name!r}")
    return import_module(f"{__package__}.{name}.config").get_cfgs()


def allocate_task_counts(configs, num_envs):
    """每个正权重任务至少分配一个环境，剩余数量按最大余数法分配。"""
    weights = [float(config["weight"]) for config in configs.values()]
    if any(not math.isfinite(w) or w <= 0 for w in weights):
        raise ValueError("Enabled task weights must be finite and positive")
    if not weights or num_envs < len(weights):
        raise ValueError("num_envs must be at least the number of enabled tasks")
    remaining = num_envs - len(weights)
    shares = [remaining * w / sum(weights) for w in weights]
    counts = [1 + math.floor(value) for value in shares]
    for i in sorted(range(len(weights)), key=lambda i: shares[i] % 1, reverse=True)[:num_envs - sum(counts)]:
        counts[i] += 1
    return counts


class TaskManager:
    def __init__(self, env, configs):
        self.env = env
        self.tasks = {}
        start = 0
        for (name, config), count in zip(configs.items(), allocate_task_counts(configs, env.num_envs)):
            ids = torch.arange(start, start + count, device=env.device)
            self.tasks[name] = task_class(name)(env, name, config, ids)
            start += count

    def call(self, method, *args):
        for task in self.tasks.values():
            getattr(task, method)(*args)

    def masked(self, method):
        result = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.env.device)
        for task in self.tasks.values():
            result |= task.mask & getattr(task, method)()
        return result

    def state_dict(self):
        return {name: task.state_dict() for name, task in self.tasks.items()}

    def load_state_dict(self, state):
        if set(state) != set(self.tasks):
            raise ValueError("Checkpoint task set differs from the current training configuration")
        for name, task in self.tasks.items():
            task.load_state_dict(state[name])
