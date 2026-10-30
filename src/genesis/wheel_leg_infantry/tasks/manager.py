"""任务注册、动态环境分配、任务 hook 分发和独立课程进度。"""

from importlib import import_module
import math

import torch
import torch.distributed as dist

from ..curriculum import CurriculumManager, TaskCurriculumProgress


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


def allocate_task_counts(configs, num_envs, weights=None):
    """每个正权重任务至少分配一个环境，剩余数量按最大余数法分配。"""
    if type(num_envs) is not int or num_envs < 1:
        raise ValueError("num_envs must be a positive integer")
    weights = {} if weights is None else weights
    if not isinstance(weights, dict) or set(weights) - configs.keys():
        raise ValueError("task_weights must map selected task names to weights")
    values = [weights.get(name, config["weight"]) for name, config in configs.items()]
    if any(isinstance(w, bool) or not isinstance(w, (int, float)) or not math.isfinite(w) or w < 0 for w in values):
        raise ValueError("Task weights must be finite and nonnegative")
    active = sum(w > 0 for w in values)
    if not active:
        raise ValueError("task_weights must retain at least one positive task")
    if num_envs < active:
        raise ValueError("num_envs must be at least the number of enabled tasks")
    remaining = num_envs - active
    scale = max(values)
    normalized = [w / scale for w in values]
    shares = [remaining * w / sum(normalized) for w in normalized]
    counts = [int(w > 0) + math.floor(value) for w, value in zip(values, shares)]
    for i in sorted(range(len(values)), key=lambda i: shares[i] % 1, reverse=True)[:num_envs - sum(counts)]:
        counts[i] += 1
    return counts


def initial_task_counts(configs, num_envs, curriculum_cfg=None):
    """训练前校验全局分配的全部阶段，返回初始配额。"""
    curriculum = CurriculumManager(curriculum_cfg)
    initial = None
    weights = {}
    allocations = []
    for stage in curriculum.stages:
        changes = stage.get("targets", {}).get("task_weights", {})
        if not isinstance(changes, dict):
            raise ValueError("task_weights must map selected task names to weights")
        weights.update(changes)
        counts = allocate_task_counts(configs, num_envs, weights)
        if curriculum.enabled:
            allocations.append(counts)
            if stage["start_iteration"] == 0:
                initial = counts
    if initial is None:
        initial = allocate_task_counts(configs, num_envs)
    allocations.append(initial)
    # 多地形的每种可能配额必须容纳其内部课程的所有阶段。
    if "multi_terrain" in configs:
        from .multi_terrain.curriculum import initial_task_counts as terrain_counts, resolve_tasks
        config = configs["multi_terrain"]
        tasks = resolve_tasks(config["env_cfg"]["multi_terrain"])
        index = list(configs).index("multi_terrain")
        for count in {counts[index] for counts in allocations}:
            terrain_counts(tasks, count, config["curriculum_cfg"])
    return initial


class TaskManager:
    def __init__(self, env, configs):
        self.env = env
        self.configs = configs
        self.tasks = {}
        self.weights = {name: config["weight"] for name, config in configs.items()}
        self.progress = {name: TaskCurriculumProgress(config["curriculum_cfg"], steps_per_iteration=env.steps_per_iteration)
                         for name, config in configs.items()}
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self._legacy_progress = set()
        start = 0
        counts = initial_task_counts(configs, env.num_envs, env.curriculum_cfg)
        for (name, config), count in zip(configs.items(), counts):
            ids = torch.arange(start, start + count, device=env.device)
            self.tasks[name] = task_class(name)(env, name, config, ids)
            if count:
                self.progress[name].mark_allocated(0)
            start += count

    def set_weights(self, values):
        """课程处理函数仅设置目标；回合统计完成后再迁移环境。"""
        allocate_task_counts(self.configs, self.env.num_envs, values)
        self.weights = {name: values.get(name, config["weight"]) for name, config in self.configs.items()}

    def reallocate(self, iteration):
        """尽量保留原有环境，只迁移超出新配额的行；返回需要 reset 的 mask。"""
        counts = allocate_task_counts(self.configs, self.env.num_envs, self.weights)
        changed = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.env.device)
        if all(len(task.env_ids) == count for task, count in zip(self.tasks.values(), counts)):
            return changed
        released = []
        retained = {}
        for (name, task), count in zip(self.tasks.items(), counts):
            retained[name] = task.env_ids[:count]
            released.append(task.env_ids[count:])
        pool = torch.cat(released)
        changed[pool] = True
        offset = 0
        for (name, task), count in zip(self.tasks.items(), counts):
            keep = retained[name]
            missing = count - len(keep)
            ids = torch.cat((keep, pool[offset:offset + missing])).sort().values
            offset += missing
            if not torch.equal(ids, task.env_ids):
                task.set_env_ids(ids)
            if count:
                self.progress[name].mark_allocated(iteration)
        return changed

    def advance_progress(self, iteration):
        for name, task in self.tasks.items():
            self.progress[name].advance(len(task.env_ids), world_size=self.world_size, iteration=iteration)

    def update_curricula(self, *, force=False):
        for name, task in self.tasks.items():
            task.curriculum.update(self.progress[name].iteration, force=force)

    def restore_legacy_progress(self, iteration):
        # 旧 checkpoint 没有样本计数，沿用其全局课程轮次，避免续训退回第 0 阶段。
        for name in self._legacy_progress:
            progress = self.progress[name]
            progress.sampled_env_steps = iteration * progress.reference_num_envs * progress.reference_steps_per_iteration
            progress.started_at_iteration = 0
        if self._legacy_progress:
            print("[curriculum] legacy checkpoint: task progress initialized from the saved global iteration")
        self._legacy_progress.clear()

    def metrics(self):
        return {f"Episode/{name}/{key}": value
                for name, task in self.tasks.items()
                for key, value in dict(num_envs=float(len(task.env_ids)), env_fraction=len(task.env_ids) / self.env.num_envs,
                                      curriculum_iteration=self.progress[name].iteration,
                                      sampled_env_steps=float(self.progress[name].sampled_env_steps)).items()}

    def call(self, method, *args):
        for task in self.tasks.values():
            if len(task.env_ids) or method in {"bind_buffers", "clear_episode_sums", "after_step"}:
                getattr(task, method)(*args)

    def masked(self, method):
        result = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.env.device)
        for task in self.tasks.values():
            if len(task.env_ids):
                result |= task.mask & getattr(task, method)()
        return result

    def state_dict(self):
        return {name: {**task.state_dict(), "curriculum_progress": self.progress[name].state_dict()}
                for name, task in self.tasks.items()}

    def load_state_dict(self, state):
        if set(state) != set(self.tasks):
            raise ValueError("Checkpoint task set differs from the current training configuration")
        for name, task in self.tasks.items():
            task.load_state_dict(state[name])
            if "curriculum_progress" in state[name]:
                self.progress[name].load_state_dict(state[name]["curriculum_progress"])
            else:
                self._legacy_progress.add(name)
