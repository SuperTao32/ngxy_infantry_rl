"""左右轮独立的短时低摩擦事件；不改变轮速目标或测量值。"""

from copy import deepcopy
import math

import torch

from .dynamics_randomization import DynamicsRandomizationManager as _Validation


def default_wheel_slip_cfg(*, enabled=False):
    return {
        "enabled": enabled,
        "strength": 1.0,
        "interval_s_range": [3.0, 8.0],  # 每个轮子相邻事件起点之间的间隔
        "duration_s_range": [0.2, 0.8],
        "ratio_range": [0.1, 0.3],  # 相对于本回合摩擦倍率，1 表示恢复抓地
    }


def resolve_wheel_slip_cfg(config=None):
    cfg = default_wheel_slip_cfg()
    if config is not None:
        _Validation._check_keys(config, cfg, "wheel_slip")
        cfg.update(deepcopy(config))
    if not isinstance(cfg["enabled"], bool):
        raise ValueError("wheel_slip.enabled must be bool")
    _Validation._validate_strength(cfg["strength"])
    for name in ("interval_s_range", "duration_s_range", "ratio_range"):
        _Validation._validate_range(cfg[name], f"wheel_slip.{name}")
    if cfg["ratio_range"][1] > 1:
        raise ValueError("wheel_slip.ratio_range must be within (0, 1]")
    if cfg["interval_s_range"][0] < cfg["duration_s_range"][1]:
        raise ValueError("wheel_slip minimum interval must be at least the maximum duration")
    return cfg


class WheelSlipRandomizer:
    """先 bind，再于物理 reset 后 reset、每个 scene.step 前 before_step。

    所有事件状态为 [环境, 左/右轮]。课程强度在 reset 锁定，任务交接保持事件。
    地形使用两轮倍率的最小值，避免 Genesis 的 max 接触混合抵消单轮低摩擦。
    要求车轮原始摩擦不低于所有地形，保证另一侧轮子的抓地力不变。
    """

    def __init__(self, config=None):
        self.config = resolve_wheel_slip_cfg(config)
        self.enabled = self.config["enabled"]
        self.strength = float(self.config["strength"])

    def apply_curriculum(self, values):
        _Validation._check_keys(values, {"strength"}, "wheel_slip curriculum")
        self.strength = _Validation._validate_strength(values.get("strength", self.config["strength"]))

    def bind(self, *, robot, wheel_links, terrain_entities, baseline_ratio, dt):
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("wheel_slip requires a finite positive control dt")
        if len(wheel_links) != 2 or wheel_links[0].idx_local == wheel_links[1].idx_local:
            raise ValueError("wheel_slip requires distinct left and right wheel links")
        if baseline_ratio.ndim != 2 or baseline_ratio.shape[1] != 1:
            raise ValueError("wheel_slip baseline_ratio must have shape (num_envs, 1)")
        self.robot = robot
        self.wheel_indices = [link.idx_local for link in wheel_links]
        self.terrain_entities = tuple(terrain_entities)
        if self.enabled:
            wheel_friction = [float(geom.friction) for link in wheel_links for geom in link.geoms]
            terrain_friction = [float(geom.friction) for entity in self.terrain_entities for geom in entity.geoms]
            if any(not link.geoms for link in wheel_links) or not terrain_friction:
                raise ValueError("wheel_slip requires wheel and terrain collision geometry")
            if any(not math.isfinite(mu) or mu <= 0 for mu in (*wheel_friction, *terrain_friction)):
                raise ValueError("wheel_slip requires positive finite model friction")
            if min(wheel_friction) < max(terrain_friction):
                raise ValueError("wheel_slip requires wheel friction >= terrain friction; lower terrain model friction first")
        # 引用域随机化的回合基准；每次写入从基准计算，不读取已降低的求解器值。
        self.baseline_ratio = baseline_ratio
        self.dt = dt
        self.env_ids = torch.arange(len(baseline_ratio), device=baseline_ratio.device)
        self.episode_strength = baseline_ratio.new_zeros(len(baseline_ratio))
        self.ratio = baseline_ratio.new_ones((len(baseline_ratio), 2))
        self.active = torch.zeros_like(self.ratio, dtype=torch.bool)
        self.steps_until_next = torch.zeros_like(self.ratio, dtype=torch.long)
        self.steps_remaining = torch.zeros_like(self.steps_until_next)

    def _sample_steps(self, key, reference):
        seconds = _Validation._sample(self.config[key], reference)
        return torch.ceil(seconds / self.dt - 1e-9).long().clamp_min(1)

    def reset(self, env_ids):
        if not self.enabled or env_ids.numel() == 0:
            return
        # 调用方先重置域随机化；无域随机化时也必须恢复上次事件写入的摩擦。
        dirty = env_ids[self.active[env_ids].any(dim=1)]
        self.episode_strength[env_ids] = self.strength
        self.ratio[env_ids] = 1
        self.active[env_ids] = False
        self.steps_remaining[env_ids] = 0
        self.steps_until_next[env_ids] = (
            self._sample_steps("interval_s_range", self.ratio[env_ids]) if self.strength > 0 else 0
        )
        self._write_friction(dirty)

    def before_step(self):
        if not self.enabled:
            return
        ended = self.active & (self.steps_remaining == 0)
        self.active[ended] = False
        self.ratio[ended] = 1
        eligible = self.episode_strength[:, None] > 0
        self.steps_until_next -= eligible.long()
        due = (self.steps_until_next <= 0) & eligible
        env_ids, wheels = due.nonzero(as_tuple=True)
        if env_ids.numel():
            reference = self.ratio[env_ids, wheels]
            sampled = _Validation._sample(self.config["ratio_range"], reference)
            self.ratio[env_ids, wheels] = 1 + self.episode_strength[env_ids] * (sampled - 1)
            self.steps_remaining[env_ids, wheels] = self._sample_steps("duration_s_range", reference)
            self.steps_until_next[env_ids, wheels] = self._sample_steps("interval_s_range", reference)
            self.active[env_ids, wheels] = True
        changed = self.env_ids[(ended | due).any(dim=1)]
        self._write_friction(changed)
        # 当前摩擦用于随后整个 scene.step；到下一次 before_step 才恢复。
        self.steps_remaining.sub_(1).clamp_min_(0)

    def _write_friction(self, env_ids):
        if env_ids.numel() == 0:
            return
        wheel_ratios = self.baseline_ratio[env_ids] * self.ratio[env_ids]
        self.robot.set_friction_ratio(wheel_ratios, links_idx_local=self.wheel_indices, envs_idx=env_ids)
        terrain_ratios = wheel_ratios.amin(dim=1, keepdim=True)
        for entity in self.terrain_entities:
            entity.set_friction_ratio(terrain_ratios.expand(-1, entity.n_links).contiguous(), envs_idx=env_ids)

    def diagnostics(self, env_idx=0):
        return {
            "wheel_slip_strength": self.strength,
            "wheel_slip_episode_strength": self.episode_strength[env_idx].detach().clone(),
            "wheel_slip_active": self.active[env_idx].detach().clone(),
            "wheel_slip_ratio": self.ratio[env_idx].detach().clone(),
            "wheel_friction_ratio": (self.baseline_ratio[env_idx] * self.ratio[env_idx]).detach().clone(),
            "wheel_slip_steps_remaining": self.steps_remaining[env_idx].detach().clone(),
            "wheel_slip_steps_until_next": self.steps_until_next[env_idx].detach().clone(),
        }
