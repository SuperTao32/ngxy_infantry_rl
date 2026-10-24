"""按回合抽样的间断通行状态机；只使用训练真值安排命令，不改变策略观测。"""

import math

import torch

from .braking import BrakingEnvelope, STOP_CONFIRMATION_S


DEFAULTS = dict(
    probability=0.0,
    predictive_braking=False,  # 旧快照缺字段时保持固定触发距离。
    crossing_speed_m_s=2.0,
    braking_distance_scale=1.25,
    braking_distance_margin_m=0.05,
    braking_time_margin_s=0.5,
    approach_speed_m_s=0.5,
    pre_stop_distance_range_m=[0.3, 0.4],
    post_stop_distance_range_m=[0.1, 0.2],
    hold_duration_range_s=[0.5, 1.5],
    stop_timeout_s=4.0,
    stop_lin_vel_m_s=0.08,
    stop_ang_vel_rad_s=0.15,
    stop_position_tolerance_m=0.15,
    max_braking_distance_m=0.4,
    edge_margin_m=0.2,
    restart_distance_m=0.5,
    extra_episode_time_s=6.0,
)


def resolve_intermittent(options, *, task):
    if not isinstance(options, dict) or set(options) - set(DEFAULTS):
        raise ValueError(f"{task}: invalid intermittent options")
    cfg = {**DEFAULTS, **options}
    for key, value in cfg.items():
        if key == "predictive_braking":
            if not isinstance(value, bool):
                raise ValueError(f"{task}: intermittent.predictive_braking must be boolean")
            continue
        values = value if key.endswith("_range_m") or key.endswith("_range_s") else [value]
        if (not isinstance(values, (list, tuple)) or len(values) != (2 if "_range_" in key else 1)
                or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values)):
            raise ValueError(f"{task}: invalid intermittent.{key}")
        if key == "probability":
            if not 0 <= value <= 1:
                raise ValueError(f"{task}: intermittent.probability must be in [0, 1]")
        elif min(values) <= 0 or values[0] > values[-1]:
            raise ValueError(f"{task}: intermittent.{key} must be positive and ordered")
    if cfg["braking_distance_scale"] < 1:
        raise ValueError(f"{task}: intermittent.braking_distance_scale must be at least 1")
    if cfg["hold_duration_range_s"][1] >= cfg["stop_timeout_s"]:
        raise ValueError(f"{task}: intermittent.stop_timeout_s must exceed maximum hold duration")
    if task == "locomotion" and cfg["probability"]:
        raise ValueError("intermittent mode requires a traversal task")
    return cfg


class IntermittentTraversal:
    APPROACH, PRE_STOP, CROSS, POST_STOP, RESTART, COMPLETE = range(6)

    def __init__(self, env):
        options = [resolve_intermittent(s.get("intermittent", {}), task=t.kind)
                   for s, t in zip(env.terrain.stages, env.terrain.tracks)]
        for cfg in options:
            if cfg["probability"] and cfg["pre_stop_distance_range_m"][0] <= env.wheel_radius + cfg["edge_margin_m"]:
                raise ValueError("intermittent pre-stop distance must leave room for wheel radius and edge margin")
        for cfg, stage in zip(options, env.terrain.stages):
            if cfg["probability"] and cfg["predictive_braking"]:
                clearance = stage.get("success", {}).get("wheel_clearance_m")
                if clearance is not None and cfg["post_stop_distance_range_m"][1] < (
                        env.wheel_radius + clearance + cfg["braking_distance_margin_m"]):
                    raise ValueError("intermittent post-stop range must include wheel clearance and braking margin")
        self.tables = {key: env.base_pos.new_tensor([cfg[key] for cfg in options]) for key in DEFAULTS}
        self.braking = BrakingEnvelope(env.base_pos) if any(cfg["predictive_braking"] for cfg in options) else None
        self.stop_timeout = torch.zeros_like(env.base_height)
        self.edge = env.base_pos.new_tensor([t.obstacle_start for t in env.terrain.tracks])
        self.enabled = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.phase = torch.zeros_like(env.episode_length_buf)
        self.move_commands = torch.zeros_like(env.commands)
        self.pre_distance = torch.zeros_like(env.base_height)
        self.post_distance = torch.zeros_like(env.base_height)
        self.hold_steps = torch.zeros_like(env.episode_length_buf)
        self.quiet_steps = torch.zeros_like(env.episode_length_buf)
        self.pause_steps = torch.zeros_like(env.episode_length_buf)
        self.total_pause_steps = torch.zeros_like(env.episode_length_buf)
        self.stop_anchor = env.base_pos.new_zeros((env.num_envs, 2))
        self.restart_origin = torch.zeros_like(env.base_height)
        self.completed_stops = torch.zeros_like(env.episode_length_buf)
        self.failed = torch.zeros_like(self.enabled)
        self.reward_paused = torch.zeros_like(self.enabled)
        self.stop_event = torch.zeros_like(self.enabled)
        self.hold_started = torch.zeros_like(self.enabled)
        self.max_stop_drift = torch.zeros_like(env.base_height)

    @property
    def paused(self):
        return self.enabled & ((self.phase == self.PRE_STOP) | (self.phase == self.POST_STOP))

    def value(self, env, key):
        return self.tables[key][env.terrain_tile_index]

    def reset(self, env, env_idx=None):
        ids = slice(None) if env_idx is None else env_idx
        probability = self.value(env, "probability")[ids]
        # 老配置以及全部关闭时不消耗额外随机数，保持原有命令采样序列。
        self.enabled[ids] = (torch.rand_like(probability) < probability) if probability.any() else False
        self.enabled[ids] &= env.traversal[ids]
        for name in ("phase", "quiet_steps", "pause_steps", "total_pause_steps", "completed_stops",
                     "failed", "reward_paused", "stop_event", "hold_started", "max_stop_drift", "restart_origin",
                     "stop_anchor", "move_commands", "hold_steps", "pre_distance", "post_distance"):
            getattr(self, name)[ids] = 0
        self.stop_timeout[ids] = self.value(env, "stop_timeout_s")[ids]
        selected = self.enabled.clone()
        if env_idx is not None:
            selected &= env_idx
        if not selected.any():
            return
        for name, key in (("pre_distance", "pre_stop_distance_range_m"),
                          ("post_distance", "post_stop_distance_range_m"), ("hold_steps", "hold_duration_range_s")):
            bounds = self.value(env, key)[selected]
            sampled = bounds[:, 0] + torch.rand_like(bounds[:, 0]) * (bounds[:, 1] - bounds[:, 0])
            if name == "hold_steps":
                sampled = (sampled / env.dt).ceil().to(self.hold_steps.dtype)
            getattr(self, name)[selected] = sampled

    def apply_commands(self, env):
        moving = self.enabled & ~self.paused
        env.commands[moving] = self.move_commands[moving]
        predictive = moving & self.value(env, "predictive_braking").bool()
        env.commands[predictive, 0] = torch.minimum(
            env.commands[predictive, 0], self.value(env, "crossing_speed_m_s")[predictive])
        approach = self.enabled & (self.phase == self.APPROACH)
        env.commands[approach, 0] = torch.minimum(
            env.commands[approach, 0], self.value(env, "approach_speed_m_s")[approach])
        # 静止只覆盖线速度和偏航速度，机身高度保留本回合的采样值。
        env.commands[self.paused, :2] = 0

    def update(self, env, wheels):
        # 此拍奖励对应刚执行的命令，阶段转换后的命令在奖励结算后才写入。
        paused = self.paused
        self.reward_paused.copy_(paused)
        self.stop_event.zero_()
        if not self.enabled.any():
            return
        phase = self.phase.clone()
        direction = env._value("direction")
        wheel_x = (wheels[..., 0] - env.terrain_spawn_centers[:, :1]) * direction[:, None]
        edge = self.edge[env.terrain_tile_index] * direction
        end = env._value("obstacle_end") * direction
        before_edge = wheel_x.max(dim=1).values + env.wheel_radius + self.value(env, "edge_margin_m") < edge
        after_edge = (wheel_x.min(dim=1).values - env.wheel_radius
                      >= end + env._value("wheel_clearance_m"))
        drift = (env.base_pos[:, :2] - self.stop_anchor).norm(dim=1)
        self.max_stop_drift.copy_(torch.maximum(self.max_stop_drift, torch.where(paused & self.hold_started, drift, 0)))
        # 制动距离和停稳后的漂移分开限制；高速度经过后的制动不能要求瞬间完成。
        limit = torch.where(self.hold_started, self.value(env, "stop_position_tolerance_m"),
                            self.value(env, "max_braking_distance_m"))
        position_ok = drift <= limit
        region_ok = torch.where(phase == self.PRE_STOP, before_edge, after_edge)
        # 使用真实世界水平速度，避免坡面上的机体系水平分量掩盖滑动。
        vertical = (env.base_lin_vel * env.projected_gravity).sum(dim=1)
        horizontal_speed = (env.base_lin_vel.square().sum(dim=1) - vertical.square()).clamp_min(0).sqrt()
        quiet = (position_ok & region_ok & (horizontal_speed <= self.value(env, "stop_lin_vel_m_s"))
                 & (env.base_ang_vel.norm(dim=1) <= self.value(env, "stop_ang_vel_rad_s"))
                 & (env.wheel_contact > 0.5).all(dim=1) & (env.base_contact < 0.5))
        first_quiet = paused & quiet & ~self.hold_started
        self.stop_anchor[first_quiet] = env.base_pos[first_quiet, :2]
        self.hold_started |= first_quiet
        self.quiet_steps.copy_(torch.where(paused & quiet, self.quiet_steps + 1, 0))
        self.pause_steps.copy_(torch.where(paused, self.pause_steps + 1, 0))
        self.total_pause_steps += paused.long()
        self.failed |= paused & (~position_ok | ~region_ok)
        completed = paused & (self.quiet_steps >= self.hold_steps) & ~self.failed
        self.failed |= paused & ~completed & (self.pause_steps * env.dt >= self.stop_timeout)
        self.stop_event.copy_(completed)
        self.completed_stops += completed.long()
        self.phase[completed & (phase == self.PRE_STOP)] = self.CROSS
        post_done = completed & (phase == self.POST_STOP)
        self.phase[post_done] = self.RESTART
        along, _ = env._task_path_coordinates(env.base_pos[:, :2])
        self.restart_origin[post_done] = along[post_done]
        restarted = (self.enabled & (phase == self.RESTART)
                     & (along - self.restart_origin >= self.value(env, "restart_distance_m")))
        self.phase[restarted] = self.COMPLETE

        pre_trigger = self.pre_distance
        post_trigger = self.post_distance
        predicted_timeout = self.value(env, "stop_timeout_s")
        if self.braking is not None:
            predictive = self.value(env, "predictive_braking").bool()
            distance, duration = self.braking.predict(horizontal_speed, env.base_height)
            margin = self.value(env, "braking_distance_margin_m")
            # pre_distance 是最大前冲后仍希望保留的轮心到沿距离。
            # 另留一拍采样行程；策略自身的动作延迟已经包含在标定数据内。
            lead = distance * self.value(env, "braking_distance_scale") + margin + horizontal_speed * env.dt
            pre_trigger = pre_trigger + torch.where(predictive, lead, 0)
            # 沿后不能为了提前制动而在轮子尚未离开障碍时发 stop。
            # 额外余量覆盖标定中的小幅回退（最大约 1.1 cm）。
            post_trigger = torch.where(predictive, torch.maximum(
                post_trigger, env.wheel_radius + env._value("wheel_clearance_m") + margin), post_trigger)
            # 标定时间含 0.5 s 连续确认，替换为本回合要求的停留时间。
            total = duration + self.hold_steps * env.dt - STOP_CONFIRMATION_S + self.value(env, "braking_time_margin_s")
            predicted_timeout = torch.where(predictive, torch.maximum(predicted_timeout, total), predicted_timeout)

        pre_start = (self.enabled & (phase == self.APPROACH)
                     & (edge - wheel_x.max(dim=1).values <= pre_trigger))
        post_start = (self.enabled & (phase == self.CROSS) & after_edge
                      & (wheel_x.min(dim=1).values - end >= post_trigger)
                      & (env.wheel_contact > 0.5).all(dim=1))
        self.phase[pre_start] = self.PRE_STOP
        self.phase[post_start] = self.POST_STOP
        entered = pre_start | post_start
        # 发出零命令时冻结时限，制动过程中不能随速度下降而缩短。
        self.stop_timeout[entered] = predicted_timeout[entered]
        self.stop_anchor[entered] = env.base_pos[entered, :2]
        self.quiet_steps[entered] = 0
        self.pause_steps[entered] = 0
        self.hold_started[entered] = False

    def success_mask(self):
        return ~self.enabled | ((self.phase == self.COMPLETE) & ~self.failed)

    def timeout_extension(self, env):
        return self.enabled * (self.value(env, "extra_episode_time_s") + self.total_pause_steps * env.dt)
