"""按回合抽样的间断通行状态机；只使用训练真值安排命令，不改变策略观测。"""

import math

import torch

# locomotion_v6/model_6100 的平地制动标定；运行时不依赖 logs 目录。
#
# 来源：version_0002/braking_eval/20261024_164833_149169，16 次成功试验。
# 每行：制动起点实际离地高度、实际速度、双轮最大前冲、含 0.5 s 确认的停车时间。
# 按目标高度分组，但查询使用每组实测高度范围；随机化关闭，每格仅一次。
STOP_CONFIRMATION_S = 0.5
SAMPLES = (
    (
        (0.209614173, 0.489313932, 0.175640387, 2.870000000),
        (0.209529877, 1.010166817, 0.322170001, 2.570000000),
        (0.208968297, 1.506756623, 0.505879012, 2.500000000),
        (0.208360344, 2.020054454, 0.749282475, 2.570000000),
    ),
    (
        (0.255318433, 0.490003740, 0.169660255, 3.120000000),
        (0.254309148, 1.027870332, 0.318924018, 2.450000000),
        (0.253593594, 1.537979734, 0.504655801, 2.360000000),
        (0.252999455, 2.057480666, 0.749109336, 2.440000000),
    ),
    (
        (0.300257504, 0.495597023, 0.160616931, 2.200000000),
        (0.299645156, 1.026689539, 0.332486056, 1.670000000),
        (0.300211310, 1.541341177, 0.532704965, 1.760000000),
        (0.300012350, 2.054837115, 0.794840004, 1.880000000),
    ),
    (
        (0.379225105, 0.475250960, 0.170076796, 2.620000000),
        (0.379248619, 0.966162971, 0.340694986, 2.530000000),
        (0.379387200, 1.442933065, 0.543843155, 2.620000000),
        (0.379400134, 1.927560167, 0.825807365, 2.770000000),
    ),
)


class BrakingEnvelope:
    """速度线性插值，邻接高度取较大值；超出标定范围使用保守包络。"""

    def __init__(self, reference):
        self.samples = reference.new_tensor(SAMPLES)
        # 每组按速度变化的实际高度范围均被覆盖。
        self.height_low = self.samples[:, :, 0].amin(dim=1)
        self.height_high = self.samples[:, :, 0].amax(dim=1)

    def predict(self, speed, height):
        velocities = self.samples[:, :, 1].contiguous()
        query = speed.abs()[None, :].expand(self.samples.shape[0], -1).contiguous()
        hi = torch.searchsorted(velocities, query).clamp(1, velocities.shape[1] - 1)
        lo = hi - 1
        lower = velocities.gather(1, lo)
        upper = velocities.gather(1, hi)
        fraction = ((query - lower) / (upper - lower)).clamp(0, 1)
        distances = self.samples[:, :, 2]
        distance = distances.gather(1, lo).lerp(distances.gather(1, hi), fraction)
        times = self.samples[:, :, 3]
        # 时间并不随速度单调变化，使用邻接速度中较慢的停车时间。
        duration = torch.maximum(times.gather(1, lo), times.gather(1, hi))
        # 低速保留最低速标定量；超速不能静默截断制动距离。
        excess = (query / velocities[:, -1:]).clamp_min(1)
        distance = distance * excess.square()
        duration = duration * excess
        low = torch.searchsorted(self.height_low, height.contiguous(), right=True).sub(1).clamp(0, self.samples.shape[0] - 1)
        high = torch.searchsorted(self.height_high, height.contiguous()).clamp(0, self.samples.shape[0] - 1)
        index = torch.arange(speed.numel(), device=speed.device)
        d = torch.maximum(distance[low, index], distance[high, index])
        t = torch.maximum(duration[low, index], duration[high, index])
        outside = (height < self.height_low[0]) | (height > self.height_high[-1])
        return (torch.where(outside, distance.amax(dim=0), d),
                torch.where(outside, duration.amax(dim=0), t))


DEFAULTS = dict(
    probability=0.0,
    predictive_braking=False,  # 旧快照缺字段时保持固定触发距离。
    crossing_speed_m_s=2.0,
    braking_distance_scale=1.25,
    braking_distance_margin_m=0.05,
    braking_time_margin_s=0.5,
    approach_speed_m_s=0.5,
    pre_stop_distance_range_m=[0.3, 0.4],
    hold_duration_range_s=[0.5, 1.5],
    stop_timeout_s=4.0,
    stop_lin_vel_m_s=0.08,
    stop_ang_vel_rad_s=0.15,
    stop_position_tolerance_m=0.15,
    max_braking_distance_m=0.4,
    edge_margin_m=0.2,
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
    return cfg


class IntermittentTraversal:
    APPROACH, PRE_STOP, CROSS = range(3)

    def __init__(self, env):
        options = [resolve_intermittent(s.get("intermittent", {}), task=t.kind)
                   for s, t in zip(env.terrain.stages, env.terrain.tracks)]
        for cfg in options:
            if cfg["probability"] and cfg["pre_stop_distance_range_m"][0] <= env.wheel_radius + cfg["edge_margin_m"]:
                raise ValueError("intermittent pre-stop distance must leave room for wheel radius and edge margin")
        self.tables = {key: env.base_pos.new_tensor([cfg[key] for cfg in options]) for key in DEFAULTS}
        self.braking = BrakingEnvelope(env.base_pos) if any(cfg["predictive_braking"] for cfg in options) else None
        self.stop_timeout = torch.zeros_like(env.base_height)
        self.edge = env.base_pos.new_tensor([t.obstacle_start for t in env.terrain.tracks])
        self.enabled = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.phase = torch.zeros_like(env.episode_length_buf)
        self.move_commands = torch.zeros_like(env.commands)
        self.pre_distance = torch.zeros_like(env.base_height)
        self.hold_steps = torch.zeros_like(env.episode_length_buf)
        self.quiet_steps = torch.zeros_like(env.episode_length_buf)
        self.pause_steps = torch.zeros_like(env.episode_length_buf)
        self.total_pause_steps = torch.zeros_like(env.episode_length_buf)
        self.stop_anchor = env.base_pos.new_zeros((env.num_envs, 2))
        self.completed_stops = torch.zeros_like(env.episode_length_buf)
        self.failed = torch.zeros_like(self.enabled)
        self.reward_paused = torch.zeros_like(self.enabled)
        self.stop_event = torch.zeros_like(self.enabled)
        self.hold_started = torch.zeros_like(self.enabled)
        self.max_stop_drift = torch.zeros_like(env.base_height)

    @property
    def paused(self):
        return self.enabled & (self.phase == self.PRE_STOP)

    def value(self, env, key):
        return self.tables[key][env.terrain_tile_index]

    def reset(self, env, env_idx=None):
        ids = slice(None) if env_idx is None else env_idx
        probability = self.value(env, "probability")[ids]
        # 老配置以及全部关闭时不消耗额外随机数，保持原有命令采样序列。
        self.enabled[ids] = (torch.rand_like(probability) < probability) if probability.any() else False
        self.enabled[ids] &= env.traversal[ids]
        self.enabled[ids] |= env.parking.drill[ids]
        for name in ("phase", "quiet_steps", "pause_steps", "total_pause_steps", "completed_stops",
                     "failed", "reward_paused", "stop_event", "hold_started", "max_stop_drift",
                     "stop_anchor", "move_commands", "hold_steps", "pre_distance"):
            getattr(self, name)[ids] = 0
        self.stop_timeout[ids] = self.value(env, "stop_timeout_s")[ids]
        selected = self.enabled.clone()
        if env_idx is not None:
            selected &= env_idx
        if not selected.any():
            return
        for name, key in (("pre_distance", "pre_stop_distance_range_m"), ("hold_steps", "hold_duration_range_s")):
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
        drills = env.parking.drill & moving
        env.commands[drills, 0] = torch.minimum(env.commands[drills, 0], self.value(env, "approach_speed_m_s")[drills])
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
        before_edge = wheel_x.max(dim=1).values + env.wheel_radius + self.value(env, "edge_margin_m") < edge
        drift = (env.base_pos[:, :2] - self.stop_anchor).norm(dim=1)
        self.max_stop_drift.copy_(torch.maximum(self.max_stop_drift, torch.where(paused & self.hold_started, drift, 0)))
        # 制动距离和停稳后的漂移分开限制；高速度经过后的制动不能要求瞬间完成。
        limit = torch.where(self.hold_started, self.value(env, "stop_position_tolerance_m"),
                            self.value(env, "max_braking_distance_m"))
        position_ok = drift <= limit
        # 使用真实世界水平速度，避免坡面上的机体系水平分量掩盖滑动。
        vertical = (env.base_lin_vel * env.projected_gravity).sum(dim=1)
        horizontal_speed = (env.base_lin_vel.square().sum(dim=1) - vertical.square()).clamp_min(0).sqrt()
        env.parking.motion_ready |= (env.parking.drill & ~paused
                                     & (horizontal_speed >= env.parking.config["required_speed_m_s"]))
        quiet = (position_ok & before_edge & (horizontal_speed <= self.value(env, "stop_lin_vel_m_s"))
                 & (env.base_ang_vel.norm(dim=1) <= self.value(env, "stop_ang_vel_rad_s"))
                 & (env.wheel_contact > 0.5).all(dim=1) & (env.base_contact < 0.5))
        first_quiet = paused & quiet & ~self.hold_started
        self.stop_anchor[first_quiet] = env.base_pos[first_quiet, :2]
        self.hold_started |= first_quiet
        self.quiet_steps.copy_(torch.where(paused & quiet, self.quiet_steps + 1, 0))
        self.pause_steps.copy_(torch.where(paused, self.pause_steps + 1, 0))
        self.total_pause_steps += paused.long()
        self.failed |= paused & (~position_ok | ~before_edge)
        completed = paused & (self.quiet_steps >= self.hold_steps) & ~self.failed
        self.failed |= paused & ~completed & (self.pause_steps * env.dt >= self.stop_timeout)
        self.stop_event.copy_(completed)
        self.completed_stops += completed.long()
        self.phase[completed & (phase == self.PRE_STOP)] = self.CROSS
        # 独立练习在停稳达标这一拍结束，不进入跨越。
        self.phase[completed & env.parking.drill] = phase[completed & env.parking.drill]

        pre_trigger = self.pre_distance
        predicted_timeout = self.value(env, "stop_timeout_s")
        if self.braking is not None:
            predictive = self.value(env, "predictive_braking").bool()
            distance, duration = self.braking.predict(horizontal_speed, env.base_height)
            margin = self.value(env, "braking_distance_margin_m")
            # pre_distance 是最大前冲后仍希望保留的轮心到沿距离。
            # 另留一拍采样行程；策略自身的动作延迟已经包含在标定数据内。
            lead = distance * self.value(env, "braking_distance_scale") + margin + horizontal_speed * env.dt
            pre_trigger = pre_trigger + torch.where(predictive, lead, 0)
            # 标定时间含 0.5 s 连续确认，替换为本回合要求的停留时间。
            total = duration + self.hold_steps * env.dt - STOP_CONFIRMATION_S + self.value(env, "braking_time_margin_s")
            predicted_timeout = torch.where(predictive, torch.maximum(predicted_timeout, total), predicted_timeout)

        pre_start = (self.enabled & (phase == self.APPROACH)
                     & (edge - wheel_x.max(dim=1).values <= pre_trigger))
        pre_start &= ~env.parking.drill | env.parking.motion_ready
        self.phase[pre_start] = self.PRE_STOP
        # 发出零命令时冻结时限，制动过程中不能随速度下降而缩短。
        self.stop_timeout[pre_start] = predicted_timeout[pre_start]
        self.stop_anchor[pre_start] = env.base_pos[pre_start, :2]
        self.quiet_steps[pre_start] = 0
        self.pause_steps[pre_start] = 0
        self.hold_started[pre_start] = False

    def success_mask(self):
        return ~self.enabled | ((self.phase == self.CROSS) & ~self.failed)

    def timeout_extension(self, env):
        return self.enabled * (self.value(env, "extra_episode_time_s") + self.total_pause_steps * env.dt)
