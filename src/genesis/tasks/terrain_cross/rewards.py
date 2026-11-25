"""每种地形独立的奖励权重、当步值及回合累计。"""

import math

import torch

from ...reward_buffer import RewardBuffer


class TerrainCrossRewards(RewardBuffer):
    def __init__(self, env, buffers, scales, *, wheel_wall_config=None):
        self.env, self.buffers = env, buffers
        self.configure_wheel_wall_force(wheel_wall_config or {"normal_x_min": 0.5, "surface_tolerance_m": 0.01})
        self._configure_contact_geometry(env)
        super().__init__(
            env.num_envs, env.device, env.dt,
            lambda name: getattr(self, "_reward_" + name, None) or getattr(env, "_reward_" + name),
            scales, event_rewards={
                "progress", "success", "failure", "stairs_up_leg_retraction", "stairs_up_wheel_airborne",
                "platform_ridge_up_leg_retraction", "platform_ridge_up_wheel_airborne",
            },
        )

    def _configure_contact_geometry(self, env):
        """缓存外露立面 (x, 下沿 z, 上沿 z) 和升高台面 (起点 x, 终点 x, z)。"""
        walls, treads = [], []
        for name, geometry in zip(env.terrain.tile_types, env.terrain.geometries):
            tile_walls, tile_treads = [], []
            if name == "stairs_up":
                for index in range(geometry.num_steps):
                    x = geometry.start_x + index * geometry.step_depth
                    lower_z = geometry.origin[2] + index * geometry.step_height
                    upper_z = lower_z + geometry.step_height
                    end_x = env.terrain.tile_size[0] / 2 if index == geometry.num_steps - 1 else x + geometry.step_depth
                    tile_walls.append([x, lower_z, upper_z])
                    tile_treads.append([x, end_x, upper_z])
            elif name == "platform_ridge_up":
                boxes = geometry.boxes[1:]
                heights = [geometry.origin[2], *(upper[2] for _, upper in boxes), geometry.origin[2]]
                edges = [*(lower[0] for lower, _ in boxes), boxes[-1][1][0]]
                tile_walls = [[x, min(a, b), max(a, b)] for x, a, b in zip(edges, heights, heights[1:])]
                tile_treads = [[lower[0], upper[0], upper[2]] for lower, upper in boxes]
            elif name == "square_wave":
                tile_walls = [[geometry.start_x, geometry.origin[2], geometry.origin[2] + geometry.height]]
            walls.append(tile_walls)
            treads.append(tile_treads)
        for attr, tiles in (("wheel_walls", walls), ("wheel_treads", treads)):
            slots = max(1, max(map(len, tiles), default=0))
            setattr(self, attr, env.commands.new_tensor([tile + [[0., 0., 0.]] * (slots - len(tile)) for tile in tiles]))

    def configure_wheel_wall_force(self, config):
        normal_x = float(config["normal_x_min"])
        tolerance = float(config["surface_tolerance_m"])
        if not (math.isfinite(normal_x) and 0 < normal_x <= 1 and math.isfinite(tolerance) and tolerance > 0):
            raise ValueError("wheel_wall_force requires 0 < normal_x_min <= 1 and positive surface_tolerance_m")
        self.wall_normal_x_min, self.wall_surface_tolerance = normal_x, tolerance

    def _reward_square_wave_wheel_wall_force(self):
        """逐点累计轮子撞入口立面的法向力；失败当拍仍计入，单位 N。"""
        return self._wheel_wall_force()

    def _reward_stairs_up_wheel_wall_force(self):
        """仅双轮无地面支撑时累计撞压台阶立面的法向力，单位 N。"""
        return self._wheel_wall_force(airborne_only=True)

    def _wheel_wall_force(self, *, airborne_only=False):
        wall_force, support_force = self._wheel_contact_forces()
        if airborne_only:
            both_airborne = (support_force <= self.env.wheel_contact_force_threshold).all(dim=1)
            wall_force = torch.where(both_airborne, wall_force, 0.0)
        return wall_force

    def stairs_up_wheels_airborne(self):
        """撞立面和墙面摩擦不算地面支撑，与立面力惩罚共用判据。"""
        _, support_force = self._wheel_contact_forces()
        return (support_force <= self.env.wheel_contact_force_threshold).all(dim=1)

    def _reward_stairs_up_wheel_airborne(self):
        """每回合首次双轮离地的事件，不乘 dt。"""
        return self.buffers.wheel_airborne.event

    def _reward_stairs_up_wheel_support(self):
        """双轮均在升高的台阶面上承重时为 1，按 dt 积分。"""
        env = self.env
        _, support_force = self._wheel_contact_forces(step_support_only=True)
        supported = (support_force > env.wheel_contact_force_threshold).all(dim=1)
        valid = ~env.reset_buf & ~env.terminated_buf & (env.base_contact < 0.5)
        return (supported & valid).to(dtype=env.commands.dtype)

    def _wheel_contact_forces(self, *, step_support_only=False):
        env = self.env
        contacts = env.robot.get_contacts(exclude_self_contact=True)
        links = torch.as_tensor(env.wheel_links_idx, device=env.device) + env.robot.link_start
        wheel_a = contacts["link_a"][..., None] == links
        wheel_b = contacts["link_b"][..., None] == links
        is_a, is_b = wheel_a.any(dim=-1), wheel_b.any(dim=-1)
        force = torch.where(is_a[..., None], contacts["force_a"], contacts["force_b"])
        normal = contacts["normal"]
        normal = normal / normal.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        position = contacts["position"]
        local_x = position[..., 0] - env.terrain_spawn_centers[:, :1]
        local_y = position[..., 1] - env.terrain_spawn_centers[:, 1:2]
        walls = self.wheel_walls[env.terrain_tile_index, None]
        tolerance = self.wall_surface_tolerance
        on_riser = (
            (walls[..., 2] > walls[..., 1])
            & ((local_x[..., None] - walls[..., 0]).abs() <= tolerance)
            & (position[..., 2, None] >= walls[..., 1] - tolerance)
            & (position[..., 2, None] <= walls[..., 2] + tolerance)
        ).any(dim=-1)
        on_wall = (
            self.buffers.assigned[:, None] & contacts["valid_mask"] & (is_a | is_b)
            & on_riser
            & (normal[..., 0].abs() >= self.wall_normal_x_min)
            & (local_y.abs() <= env.terrain.tile_size[1] / 2)
        )
        projection = (force * normal).sum(dim=-1)
        wall_force = torch.where(on_wall, projection.abs(), 0.0).sum(dim=1)
        # wheel_contact 将撞墙也算接触；只用非立面、向上承重的接触法向力判断支撑。
        support = contacts["valid_mask"] & ~on_wall & (normal[..., 2].abs() >= 0.5)
        if step_support_only:
            # 双轮允许位于不同台面；凸台后的第二平台也按实际高度、长度判定。
            treads = self.wheel_treads[env.terrain_tile_index, None]
            on_tread = (
                (treads[..., 1] > treads[..., 0])
                & (local_x[..., None] >= treads[..., 0]) & (local_x[..., None] <= treads[..., 1])
                & ((position[..., 2, None] - treads[..., 2]).abs() <= tolerance)
            ).any(dim=-1)
            support &= (
                self.buffers.assigned[:, None] & on_tread
                & (local_y.abs() <= env.terrain.tile_size[1] / 2)
            )
        upward_force = (projection * normal[..., 2]).clamp_min(0.0)
        support_force = torch.where(
            support[..., None] & (wheel_a | wheel_b), upward_force[..., None], 0.0,
        ).sum(dim=1)
        return wall_force, support_force

    def compute(self, mask, *, exclude=(), standing_mask=None):
        # 通关后仅落地窗口继续计分，其他越障奖励仍使用原任务 mask。
        overrides = {"stairs_down_landing_stability": self.buffers.landing.gate} if self.buffers.landing is not None else None
        if standing_mask is not None:
            mask = mask & ~standing_mask
            if overrides is not None:
                overrides = {name: gate & ~standing_mask for name, gate in overrides.items()}
        return super().compute(mask, exclude=exclude, mask_overrides=overrides)

    def _reward_stairs_down_landing_stability(self):
        """落地窗口内的竖直与 roll/pitch 振荡，配负权重并按 dt 积分。"""
        vertical_velocity = (self.env.base_lin_vel * self.env.projected_gravity).sum(dim=1)
        oscillation = vertical_velocity.square() + self.buffers.landing.angular_weight * self.env.base_ang_vel[:, :2].square().sum(dim=1)
        return self.buffers.landing.gate * oscillation

    def _reward_stairs_up_leg_retraction(self):
        """缩腿新纪录的增量，单位 m；事件奖励不乘 dt，伸缩或退回台阶不重新计奖。"""
        return self.buffers.retraction.delta

    # 上行平台凸台复用上台阶的结算规则，权重和回合累计按地形独立保存。
    _reward_platform_ridge_up_wheel_support = _reward_stairs_up_wheel_support
    _reward_platform_ridge_up_wheel_airborne = _reward_stairs_up_wheel_airborne
    _reward_platform_ridge_up_leg_retraction = _reward_stairs_up_leg_retraction
    _reward_platform_ridge_up_wheel_wall_force = _reward_stairs_up_wheel_wall_force

    def _reward_progress(self):
        return self.buffers.progress_delta

    def _reward_success(self):
        return self.buffers.success_event

    def _reward_failure(self):
        return self.buffers.failure_event

    def _reward_task_lateral(self):
        delta = self.env.base_pos[:, :2] - self.buffers.path_origin_xy
        forward = self.buffers.path_forward_xy
        return (delta[:, 1] * forward[:, 0] - delta[:, 0] * forward[:, 1]).square()

    def _reward_task_heading(self):
        yaw = torch.deg2rad(self.env.base_euler[:, 2])
        forward = torch.stack((yaw.cos(), yaw.sin()), dim=-1)
        return 1.0 - (forward * self.buffers.path_forward_xy).sum(dim=-1).clamp(-1.0, 1.0)
