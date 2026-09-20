"""按物理控制步调度短时外力；每次事件只推一个机身/轮子 link。"""

import math

import torch


def default_push_cfg():
    return {
        "enabled": True,
        "targets": ["base", "left_wheel", "right_wheel"],
        "interval_s_range": [3.0, 8.0],  # 相邻脉冲起点之间的间隔
        "duration_s": 0.02,
        "force_x_range": [-30.0, 30.0],  # N，世界坐标系三轴独立采样
        "force_y_range": [-30.0, 30.0],
        "force_z_range": [-30.0, 30.0],
    }


class PushRandomizer:
    """独立于 PPO episode 时钟；只有物理 reset 才重置当前脉冲和倒计时。"""

    def __init__(self, config, *, enabled):
        if "force_xy_norm_range" in config:
            raise ValueError("push.force_xy_norm_range was replaced by force_x_range, force_y_range, force_z_range")
        self.config = config
        self.enabled = enabled and config["enabled"]
        targets = config["targets"]
        allowed = ("base", "left_wheel", "right_wheel")
        if not isinstance(targets, (list, tuple)) or not targets or any(t not in allowed for t in targets):
            raise ValueError(f"push.targets must be a nonempty list drawn from {allowed}")
        if len(set(targets)) != len(targets):
            raise ValueError("push.targets must not contain duplicates")
        self.targets = tuple(targets)
        for name in ("interval_s_range", "force_x_range", "force_y_range", "force_z_range"):
            values = config[name]
            if (not isinstance(values, (list, tuple)) or len(values) != 2
                    or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values)
                    or values[0] > values[1]):
                raise ValueError(f"push.{name} must contain finite bounds with lower <= upper")
        duration = config["duration_s"]
        if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
            raise ValueError("push.duration_s must be finite and positive")
        if config["interval_s_range"][0] < duration:
            raise ValueError("push interval must be at least duration_s, so pulses cannot overlap")

    def bind(self, *, solver, links, dt, num_envs, reference):
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("push requires a finite positive control dt")
        if self.enabled and any(name not in links for name in self.targets):
            raise ValueError("push requires link mappings for every configured target")
        self.solver = solver
        # solver API 要全局 link 索引；由环境按配置的 link 名称解析。
        self.link_indices = [links[name].idx for name in self.targets] if self.enabled else []
        self.dt = dt
        self.duration_steps = max(1, math.ceil(self.config["duration_s"] / dt - 1e-9))
        self.env_ids = torch.arange(num_envs, device=reference.device)
        self.steps_until_next = torch.zeros_like(self.env_ids)
        self.steps_remaining = torch.zeros_like(self.env_ids)
        self.target_index = torch.full_like(self.env_ids, -1)
        self.force = reference.new_zeros((num_envs, 3))
        self.strength = reference.new_ones(num_envs)
        self.last_applied_force = torch.zeros_like(self.force)
        self.last_applied_target = torch.full_like(self.env_ids, -1)

    def _sample_intervals(self, env_ids):
        low, high = self.config["interval_s_range"]
        if low == high:
            return torch.full_like(env_ids, max(self.duration_steps, math.ceil(low / self.dt - 1e-9)))
        seconds = self.force.new_empty((env_ids.numel(),)).uniform_(low, high)
        return torch.ceil(seconds / self.dt - 1e-9).long().clamp_min(self.duration_steps)

    def reset(self, env_ids, *, strength=1.0):
        if not self.enabled or env_ids.numel() == 0:
            return
        self.strength[env_ids] = strength
        self.steps_until_next[env_ids] = self._sample_intervals(env_ids) if strength > 0 else 0
        self.steps_remaining[env_ids] = 0
        self.force[env_ids] = 0
        self.target_index[env_ids] = -1
        self.last_applied_force[env_ids] = 0
        self.last_applied_target[env_ids] = -1

    def before_step(self):
        """每个 scene.step 前调用一次；Genesis 在整个控制步结束后自动清除外力。"""
        if not self.enabled:
            return
        self.last_applied_force.zero_()
        self.last_applied_target.fill_(-1)
        eligible = self.strength > 0
        self.steps_until_next -= eligible.long()
        due = self.env_ids[(self.steps_until_next <= 0) & eligible]
        if due.numel():
            count = due.numel()
            for axis, name in enumerate(("force_x_range", "force_y_range", "force_z_range")):
                low, high = self.config[name]
                self.force[due, axis] = (
                    self.force.new_full((count,), low) if low == high
                    else self.force.new_empty((count,)).uniform_(low, high)
                )
            self.force[due] *= self.strength[due, None]
            self.target_index[due] = torch.randint(len(self.targets), (count,), device=due.device)
            self.steps_remaining[due] = self.duration_steps
            self.steps_until_next[due] = self._sample_intervals(due)
        # 按目标分组调用，支持不同环境在同一拍被推不同的轮子。
        for target, link_idx in enumerate(self.link_indices):
            ids = self.env_ids[(self.steps_remaining > 0) & (self.target_index == target)]
            if not ids.numel():
                continue
            forces = self.force[ids]
            self.solver.apply_links_external_force(
                forces[:, None, :], links_idx=[link_idx], envs_idx=ids,
                ref="link_com", local=False,
            )
            self.last_applied_force[ids] = forces
            self.last_applied_target[ids] = target
        self.steps_remaining.sub_(1).clamp_min_(0)

    def diagnostics(self, env_idx):
        target = int(self.last_applied_target[env_idx])
        return {
            "push_target": None if target < 0 else self.targets[target],
            "push_force_world_N": self.last_applied_force[env_idx].detach().clone(),
            "push_steps_remaining": int(self.steps_remaining[env_idx]),
            "push_steps_until_next": int(self.steps_until_next[env_idx]),
        }
