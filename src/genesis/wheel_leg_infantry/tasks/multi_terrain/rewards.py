"""多地形任务奖励；各 task 在 config.py 中独立选择和设置权重。"""

import torch


class MultiTerrainRewards:
    def _reward_intermittent_stop(self):
        return self.intermittent.stop_event & ~self.terminated_buf

    # 各通关任务共用。
    def _reward_task_progress(self):
        return self.progress * ~self.terminated_buf

    def _reward_task_success(self):
        return self.success * ~self.terminated_buf

    def _reward_stairs_up_support(self):
        return self.stairs_up_buffers.support_event & ~self.terminated_buf

    def _reward_stairs_up_retract(self):
        return self.stairs_up_buffers.retract_progress * ~self.terminated_buf

    def _reward_stairs_up_takeoff(self):
        return self.stairs_up_buffers.takeoff_event & ~self.terminated_buf

    def _reward_stairs_up_landing_distance(self):
        return self.stairs_up_buffers.landing_distance_event * ~self.terminated_buf

    def _reward_stairs_up_wheel_wall_force(self):
        # 失败当拍仍计入碰撞代价；此连续惩罚由 reward_tables 乘 dt。
        return self.stairs_up_buffers.wheel_wall_force

    def _reward_stairs_up_approach_lift(self):
        return self.stairs_up_buffers.approach_lift_progress * ~self.terminated_buf

    def _reward_stairs_up_backswing(self):
        return self.stairs_up_buffers.backswing_progress * ~self.terminated_buf

    def _reward_stairs_up_forward_swing(self):
        return self.stairs_up_buffers.forward_progress * ~self.terminated_buf

    def _reward_stairs_up_extend(self):
        return self.stairs_up_buffers.extend_progress * ~self.terminated_buf

    def _reward_stairs_up_wheel_support(self):
        return self.stairs_up_buffers.wheel_support_event & ~self.terminated_buf

    def _reward_stairs_up_wheel_forward_action(self):
        """双轮落台接地后，在不超过前向速度命令时鼓励轮子正转。"""
        state = self.stairs_up_buffers
        active = (state.task_mask & state.wheels_landed & (self.wheel_contact > 0.5).all(dim=1)
                  & (self.commands[:, 0] > 0) & (self.base_lin_vel[:, 0] <= self.commands[:, 0])
                  & ~state.fallen_off & ~self.terminated_buf)
        wheel = slice(self.num_joints, self.num_joints + self.num_wheels)
        return self.actions[:, wheel].clamp_min(0).mean(dim=1) * active

    def _reward_task_lateral(self):
        _, lateral = self._task_path_coordinates(self.base_pos[:, :2])
        return lateral.square()

    def _reward_task_heading(self):
        yaw = torch.deg2rad(self.base_euler[:, 2])
        forward = torch.stack((torch.cos(yaw), torch.sin(yaw)), dim=-1)
        return 1 - (forward * self.task_forward_xy).sum(dim=-1).clamp(-1, 1)

    def _reward_stairs_down_landing_stability(self):
        """下台阶后的振荡代价，配置负权重；不惩罚水平前进或 yaw 转动。"""
        # 速度和重力方向都在机体系，点积得到世界竖直方向速度（符号不影响平方）。
        state = self.stairs_down_buffers
        vertical_velocity = (self.base_lin_vel * self.projected_gravity).sum(dim=1)
        angular_weight = state.landing_ang_vel_weight[self.terrain_tile_index]
        oscillation = vertical_velocity.square() + angular_weight * self.base_ang_vel[:, :2].square().sum(dim=1)
        return state.landing_gate * oscillation
