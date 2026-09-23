"""动力学参数在物理 reset 采样；外力按物理步调度，任务交接保持随机状态。"""

from copy import deepcopy
import math

import torch

from .push import PushRandomizer, default_push_cfg


GAIN_NAMES = ("joint_kp", "joint_kd", "wheel_kd")
MOTOR_NAMES = (*GAIN_NAMES, "joint_force_limit", "wheel_force_limit")
SPRING_NAMES = ("preload_force", "stiffness", "damping")
EXTENDED_GROUPS = ("base_mass", "com_displacement", "motor_strength", "motor_offset", "gas_spring", "passive_joints", "push")


def default_domain_rand_cfg(*, enabled=False):
    """倍率相对于模型/当前控制模式的标称值；每次返回独立配置。"""
    return {
        "enabled": enabled,
        "strength": 1.0,  # 0=标称模型，1=完整随机化范围；课程可覆盖
        "push": default_push_cfg(),
        "friction": {"enabled": True, "ratio_range": [0.6, 1.4]},
        "base_mass": {"enabled": True, "added_mass_range": [-1.0, 2.0]},  # kg
        "com_displacement": {"enabled": True, "displacement_range": [-0.01, 0.01]},  # m, xyz 独立
        "motor_strength": {"enabled": True, "ratio_range": [0.9, 1.1]},
        "motor_offset": {"enabled": True, "offset_range": [-0.02, 0.02]},  # rad, 仅腿部位置控制
        "passive_joints": {
            "enabled": True,
            "frictionloss_range": [0.005, 0.015],  # 被动铰链库仑摩擦，N·m
            "damping_range": [0.005, 0.015],  # 被动铰链黏性阻尼，N·m·s/rad
            "overrides": {},  # joint_name -> 单独的 frictionloss_range / damping_range
        },
        "gas_spring": {
            "enabled": True,
            "preload_force_range": [0.95, 1.05],
            "stiffness_range": [1.0, 1.0],
            "damping_range": [1.0, 1.0],
        },
        "motor_gains": {
            "enabled": True,
            "joint_kp_enabled": True,
            "joint_kd_enabled": True,
            "wheel_kd_enabled": True,
            "joint_kp_range": [0.9, 1.1],
            "joint_kd_range": [0.9, 1.1],
            "wheel_kd_range": [0.9, 1.1],
        },
    }


class DomainRandomizationManager:
    def __init__(self, config=None):
        self.config = default_domain_rand_cfg()
        # 已保存的第一版配置缺少这些组时，不悄悄改变原实验的随机化分布。
        for name in EXTENDED_GROUPS:
            self.config[name]["enabled"] = False
        raw = {} if config is None else deepcopy(config)
        self._check_keys(raw, self.config, "domain_rand")
        for name, value in raw.items():
            if isinstance(self.config[name], dict):
                self._check_keys(value, self.config[name], f"domain_rand.{name}")
                self.config[name].update(value)
            else:
                self.config[name] = value
        for group in (self.config, *(value for value in self.config.values() if isinstance(value, dict))):
            for name, value in group.items():
                if name.endswith("enabled") and not isinstance(value, bool):
                    raise ValueError("domain_rand enabled flags must be bool")
        self._validate_range(self.config["friction"]["ratio_range"], "friction.ratio_range")
        for name in GAIN_NAMES:
            self._validate_range(self.config["motor_gains"][f"{name}_range"], f"motor_gains.{name}_range")
        self.friction_enabled = self.config["enabled"] and self.config["friction"]["enabled"]
        self.randomized_gains = tuple(
            name for name in GAIN_NAMES
            if self.config["enabled"] and self.config["motor_gains"]["enabled"]
            and self.config["motor_gains"][f"{name}_enabled"]
        )
        self.motor_gains_enabled = bool(self.randomized_gains)
        for name in EXTENDED_GROUPS:
            setattr(self, f"{name}_enabled", self.config["enabled"] and self.config[name]["enabled"])
        self._validate_range(self.config["base_mass"]["added_mass_range"], "base_mass.added_mass_range", positive=False)
        self._validate_range(self.config["com_displacement"]["displacement_range"], "com_displacement.displacement_range", positive=False)
        self._validate_range(self.config["motor_offset"]["offset_range"], "motor_offset.offset_range", positive=False)
        self._validate_range(self.config["motor_strength"]["ratio_range"], "motor_strength.ratio_range")
        for name in SPRING_NAMES:
            self._validate_range(self.config["gas_spring"][f"{name}_range"], f"gas_spring.{name}_range")
        self._validate_passive_config()
        self.strength = self._validate_strength(self.config["strength"])
        self.push = PushRandomizer(self.config["push"], enabled=self.config["enabled"])
        self.requires_batched_dofs = self.motor_gains_enabled or self.motor_strength_enabled or self.passive_joints_enabled

    @staticmethod
    def _validate_strength(value):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("domain_rand.strength must be a finite number in [0, 1]")
        return float(value)

    def apply_curriculum(self, values):
        """只更新后续 reset 的采样强度，不改当前回合参数或原始完整范围。"""
        self._check_keys(values, {"strength"}, "domain_rand curriculum")
        self.strength = self._validate_strength(values.get("strength", self.config["strength"]))

    def _validate_passive_config(self):
        cfg = self.config["passive_joints"]
        keys = ("frictionloss_range", "damping_range")
        if not isinstance(cfg["overrides"], dict):
            raise ValueError("passive_joints.overrides must map joint names to ranges")
        for joint_name, values in cfg["overrides"].items():
            if not isinstance(joint_name, str):
                raise ValueError("passive joint names must be strings")
            self._check_keys(values, keys, f"passive_joints.overrides.{joint_name}")
        for values in (cfg, *cfg["overrides"].values()):
            for key in keys:
                if key in values:
                    self._validate_range(values[key], f"passive_joints.{key}", positive=False)
                    if values[key][0] < 0:
                        raise ValueError(f"passive_joints.{key} must be nonnegative")

    @staticmethod
    def _check_keys(values, allowed, name):
        if not isinstance(values, dict):
            raise ValueError(f"{name} must be a dict")
        unknown = set(values).difference(allowed)
        if unknown:
            raise ValueError(f"unsupported {name} keys: {sorted(unknown)}")

    @staticmethod
    def _validate_range(values, name, *, positive=True):
        if not isinstance(values, (list, tuple)) or len(values) != 2:
            raise ValueError(f"domain_rand.{name} must be [lower, upper]")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
            raise ValueError(f"domain_rand.{name} must contain finite numbers")
        if values[0] > values[1] or (positive and values[0] <= 0):
            condition = "0 < lower <= upper" if positive else "lower <= upper"
            raise ValueError(f"domain_rand.{name} requires {condition}")

    def bind(self, *, robot, friction_entities, motor_params, joint_indices, wheel_indices, num_envs, batched_dofs,
             base_link=None, spring_params=None, num_springs=0, passive_joint_names=(), passive_indices=None,
             push_links=None, dt=0.02):
        """在 scene.build 和 DOF 索引就绪后绑定；标称值不可从随机化后的参数反推。"""
        self.robot = robot
        self.friction_entities = tuple(friction_entities)
        self.joint_indices = joint_indices
        self.wheel_indices = wheel_indices
        self.batched_dofs = batched_dofs
        if self.requires_batched_dofs and not batched_dofs:
            raise ValueError("DOF randomization requires batch_dofs_info=True")
        if self.base_mass_enabled or self.com_displacement_enabled:
            if base_link is None:
                raise ValueError("base mass/COM randomization requires a named base_link")
            self.base_link_indices = [base_link.idx_local]
            self.nominal_base_mass = float(base_link.get_mass())
            if self.base_mass_enabled and self.nominal_base_mass + self.config["base_mass"]["added_mass_range"][0] <= 0:
                raise ValueError("base mass plus the minimum added mass must remain positive")
        self.nominal_motor_params = {name: motor_params[name].clone() for name in MOTOR_NAMES}
        self.motor_nominals = {name: value.expand(num_envs, -1).clone() for name, value in self.nominal_motor_params.items()}
        self.motor_scales = {name: torch.ones_like(value) for name, value in self.motor_nominals.items() if name in GAIN_NAMES}
        self.motor_values = {name: value.clone() for name, value in self.motor_nominals.items()}
        reference = motor_params["joint_kp"]
        self.env_ids = torch.arange(num_envs, device=reference.device)
        self.episode_strength = reference.new_zeros(num_envs)
        self.friction_ratio = torch.ones((num_envs, 1), device=reference.device, dtype=reference.dtype)
        self.added_mass = torch.zeros_like(self.friction_ratio)
        self.com_displacement = torch.zeros((num_envs, 1, 3), device=reference.device, dtype=reference.dtype)
        self.motor_strengths = torch.ones_like(self.friction_ratio)
        self.motor_offsets = torch.zeros_like(self.motor_values["joint_kp"])
        self._bind_springs(spring_params, num_springs, num_envs, reference)
        self._bind_passive_joints(passive_joint_names, passive_indices, num_envs, reference)
        self.push.bind(solver=robot.solver, links={} if push_links is None else push_links,
                       dt=dt, num_envs=num_envs, reference=reference)

    def _bind_passive_joints(self, names, indices, num_envs, reference):
        self.passive_joint_names = tuple(names)
        self.passive_indices = indices
        self.passive_joint_values = {}
        self.nominal_passive_joint_values = {}
        self.passive_joint_ranges = {}
        cfg = self.config["passive_joints"]
        unknown = set(cfg["overrides"]).difference(names)
        if unknown:
            raise ValueError(f"passive_joints overrides contain unknown or controlled joints: {sorted(unknown)}")
        if not self.passive_joints_enabled:
            return
        if not names or indices is None or len(indices) != len(names):
            raise ValueError("passive joint randomization requires matching hinge names and DOF indices")
        for name in ("frictionloss", "damping"):
            key = f"{name}_range"
            self.passive_joint_ranges[name] = reference.new_tensor([
                cfg["overrides"].get(joint_name, {}).get(key, cfg[key]) for joint_name in names
            ])
            baseline = getattr(self.robot, f"get_dofs_{name}")(indices)
            self.passive_joint_values[name] = baseline.expand(num_envs, -1).clone()
            self.nominal_passive_joint_values[name] = self.passive_joint_values[name].clone()

    def _bind_springs(self, params, num_springs, num_envs, reference):
        self.nominal_spring_params = {}
        self.gas_spring_scales = {}
        self.gas_spring_values = {}
        if params is None:
            if self.gas_spring_enabled:
                raise ValueError("gas spring randomization requires nominal spring_params")
            return
        self._check_keys(params, SPRING_NAMES, "spring_params")
        if set(params) != set(SPRING_NAMES) or num_springs <= 0:
            raise ValueError("spring_params requires preload_force, stiffness, damping and num_springs > 0")
        for name in SPRING_NAMES:
            nominal = float(params[name])
            if not math.isfinite(nominal) or nominal < 0:
                raise ValueError(f"gas_spring_{name} must be finite and nonnegative")
            self.nominal_spring_params[name] = nominal
            self.gas_spring_values[name] = reference.new_full((num_envs, num_springs), nominal)
            self.gas_spring_scales[name] = reference.new_ones((num_envs, num_springs))

    @staticmethod
    def _sample(limits, reference):
        lower, upper = limits
        # 固定范围不消耗随机数，关闭随机化也不会改变原来的采样序列。
        if lower == upper:
            return torch.full_like(reference, lower)
        return torch.empty_like(reference).uniform_(lower, upper)

    def _sample_scaled(self, limits, reference, *, nominal):
        if self.strength == 0:
            return torch.full_like(reference, nominal)
        sampled = self._sample(limits, reference)
        return sampled if self.strength == 1 else nominal + self.strength * (sampled - nominal)

    def reset(self, env_ids=None):
        """仅重采样所选环境；调用者必须传入整数 ID，任务阶段切换不得调用。"""
        env_ids = self.env_ids if env_ids is None else env_ids
        if env_ids.numel() == 0:
            return
        self.episode_strength[env_ids] = self.strength if self.config["enabled"] else 0.0
        self._randomize_rigids(env_ids)
        self._randomize_controls(env_ids)
        self._randomize_gas_springs(env_ids)
        self._randomize_passive_joints(env_ids)
        self.push.reset(env_ids, strength=self.strength)

    def before_step(self):
        self.push.before_step()

    def _randomize_passive_joints(self, env_ids):
        if not self.passive_joints_enabled:
            return
        for name, values in self.passive_joint_values.items():
            limits = self.passive_joint_ranges[name]
            lower, upper = limits[:, 0], limits[:, 1]
            nominal = self.nominal_passive_joint_values[name][env_ids]
            if self.strength == 0:
                sampled = nominal
            else:
                sampled = lower + (upper - lower) * torch.rand_like(values[env_ids])
                if self.strength != 1:
                    sampled = nominal + self.strength * (sampled - nominal)
            getattr(self.robot, f"set_dofs_{name}")(sampled, self.passive_indices, envs_idx=env_ids)
            values[env_ids] = sampled

    def _randomize_gas_springs(self, env_ids):
        if not self.gas_spring_enabled:
            return
        # 每环境、每根弹簧、每项参数独立采样，始终从标称值计算，避免重复 reset 累乘。
        for name, scales in self.gas_spring_scales.items():
            sampled = self._sample_scaled(self.config["gas_spring"][f"{name}_range"], scales[env_ids], nominal=1.0)
            scales[env_ids] = sampled
            self.gas_spring_values[name][env_ids] = self.nominal_spring_params[name] * sampled

    def _randomize_rigids(self, env_ids):
        if self.friction_enabled:
            ratios = self._sample_scaled(self.config["friction"]["ratio_range"], self.friction_ratio[env_ids], nominal=1.0)
            self.friction_ratio[env_ids] = ratios
            # Genesis 接触摩擦取双方系数的 max。两侧使用同一倍率，才能同时降低/提高摩擦。
            for entity in self.friction_entities:
                entity.set_friction_ratio(ratios.expand(-1, entity.n_links).contiguous(), envs_idx=env_ids)
        if self.base_mass_enabled:
            values = self._sample_scaled(self.config["base_mass"]["added_mass_range"], self.added_mass[env_ids], nominal=0.0)
            self.robot.set_mass_shift(values, self.base_link_indices, envs_idx=env_ids)
            self.added_mass[env_ids] = values
        if self.com_displacement_enabled:
            values = self._sample_scaled(self.config["com_displacement"]["displacement_range"], self.com_displacement[env_ids], nominal=0.0)
            self.robot.set_COM_shift(values, self.base_link_indices, envs_idx=env_ids)
            self.com_displacement[env_ids] = values

    def _randomize_controls(self, env_ids):
        if self.motor_strength_enabled:
            self.motor_strengths[env_ids] = self._sample_scaled(self.config["motor_strength"]["ratio_range"], self.motor_strengths[env_ids], nominal=1.0)
        if self.motor_offset_enabled:
            self.motor_offsets[env_ids] = self._sample_scaled(self.config["motor_offset"]["offset_range"], self.motor_offsets[env_ids], nominal=0.0)
        for name in self.randomized_gains:
            scales = self.motor_scales[name]
            scales[env_ids] = self._sample_scaled(self.config["motor_gains"][f"{name}_range"], scales[env_ids], nominal=1.0)
        for name in MOTOR_NAMES if self.motor_strength_enabled else self.randomized_gains:
            self._write_motor_param(name, env_ids)

    def offset_joint_targets(self, targets):
        """rad 偏移在动作延迟之后、几何限位之前加入；轮速目标保持原单位。"""
        return targets + self.motor_offsets if self.motor_offset_enabled else targets

    def apply_motor_params(self, params, envs_idx=None):
        """切换控制模式，保留当前随机倍率，支持部分环境交回 locomotion。"""
        if envs_idx is not None and not self.batched_dofs:
            raise ValueError("per-environment motor profiles require batch_dofs_info=True")
        ids = self.env_ids if envs_idx is None else envs_idx
        if ids.numel() == 0:
            return
        for name, value in params.items():
            self.motor_nominals[name][ids] = value
            self._write_motor_param(name, ids)

    def _write_motor_param(self, name, ids):
        applied = self.motor_nominals[name][ids]
        if name in self.motor_scales:
            applied = applied * self.motor_scales[name][ids]
        # 同时缩放 PD 增益和力矩限幅，等价于 strength * clip(PD, ±nominal_limit)。
        # 保留 Genesis 每个物理子步的闭环控制，不在控制步手算一次固定力矩。
        applied = applied * self.motor_strengths[ids]
        indices = self.wheel_indices if name.startswith("wheel_") else self.joint_indices
        kwargs = {"envs_idx": ids} if self.batched_dofs else {}
        values = applied.contiguous() if self.batched_dofs else applied[0]
        if name.endswith("force_limit"):
            self.robot.set_dofs_force_range(-values, values, indices, **kwargs)
        elif name.endswith("kp"):
            self.robot.set_dofs_kp(values, indices, **kwargs)
        else:
            self.robot.set_dofs_kv(values, indices, **kwargs)
        self.motor_values[name][ids] = applied

    def diagnostics(self, env_idx=0):
        return {
            "domain_rand_strength": self.strength,
            "domain_rand_episode_strength": self.episode_strength[env_idx].detach().clone(),
            "friction_ratio": self.friction_ratio[env_idx, 0].detach().clone(),
            "added_mass_kg": self.added_mass[env_idx, 0].detach().clone(),
            "com_displacement_m": self.com_displacement[env_idx, 0].detach().clone(),
            "motor_strength": self.motor_strengths[env_idx, 0].detach().clone(),
            "joint_motor_offsets_rad": self.motor_offsets[env_idx].detach().clone(),
            **{f"{name}_scale": value[env_idx].detach().clone() for name, value in self.motor_scales.items()},
            **{f"gas_spring_{name}_scale": value[env_idx].detach().clone() for name, value in self.gas_spring_scales.items()},
            **{f"gas_spring_{name}": value[env_idx].detach().clone() for name, value in self.gas_spring_values.items()},
            "passive_joint_names": self.passive_joint_names,
            **{f"passive_joint_{name}": value[env_idx].detach().clone() for name, value in self.passive_joint_values.items()},
            **self.push.diagnostics(env_idx),
        }
