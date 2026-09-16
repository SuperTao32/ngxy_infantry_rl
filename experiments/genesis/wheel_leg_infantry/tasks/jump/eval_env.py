"""Jump 交互评估的可选台阶；不改变训练环境或 actor 观测。"""

import math
import time
from datetime import datetime

import genesis as gs
import torch

from .env import JumpEnv


class JumpEvalEnv(JumpEnv):
    def start_motor_logging(self, interval):
        self._motor_log_interval = interval
        self._motor_log_step = 0
        self._motor_log_start = time.perf_counter()

    def _update_task_state(self):
        super()._update_task_state()
        # 在 step 的 reset 分支之前读取；构建和初始 warmup 时尚未启用。
        if not getattr(self, "_motor_log_interval", 0):
            return
        self._motor_log_step += 1
        if self._motor_log_step % self._motor_log_interval:
            return
        self._print_motor_diagnostics()

    def _print_motor_diagnostics(self):
        diag = self.get_pd_diagnostics()
        indices = torch.cat((self.joints_dof_idx, self.wheels_dof_idx))
        # Genesis 根据当前状态重算的限幅控制扭矩快照；不是上一子步的力矩记录，
        # 也不是含接触/约束等作用的 DOF 总内力。
        torque = self.robot.get_dofs_control_force(indices)[0]
        joint_p = diag["joint_kp"] * (diag["joint_target_pos"] - diag["joint_pos"])
        joint_d = diag["joint_kd_damping"]
        wheel_d = diag["wheel_kd"] * (diag["wheel_target_vel"] - diag["wheel_vel"])
        # 一次传回 CPU，避免按电机逐项触发 GPU 同步。
        rows = torch.cat((
            torch.stack((diag["joint_kp"], diag["joint_kd"], diag["joint_force_limit"],
                         diag["joint_target_pos"], diag["joint_pos"], diag["joint_vel"],
                         joint_p, joint_d, torque[:self.num_joints]), dim=1),
            torch.stack((torch.zeros_like(diag["wheel_kd"]), diag["wheel_kd"], diag["wheel_force_limit"],
                         diag["wheel_target_vel"], diag["wheel_vel"], diag["wheel_vel"],
                         torch.zeros_like(wheel_d), wheel_d, torque[self.num_joints:]), dim=1),
        )).detach().cpu().tolist()
        stamp = datetime.now().astimezone().isoformat(timespec="milliseconds")
        elapsed = time.perf_counter() - self._motor_log_start
        lines = [f"[jump motors] time={stamp} elapsed={elapsed:.3f}s "
                 f"sim={self._motor_log_step * self.dt:.3f}s step={self._motor_log_step}"]
        for index, (name, row) in enumerate(zip((*diag["joint_names"], *diag["wheel_names"]), rows)):
            kp, kd, limit, target, actual, velocity, p, d, tau = row
            state = (f"q_target={target:+.4f} q={actual:+.4f} rad qd={velocity:+.4f} rad/s"
                     if index < self.num_joints else f"qd_target={target:+.4f} qd={actual:+.4f} rad/s")
            lines.append(f"  {name}: Kp={kp:.3f} Kd={kd:.3f} {state} "
                         f"P_est={p:+.3f} D_est={d:+.3f} tau_motor={tau:+.3f} "
                         f"|tau|={abs(tau):.3f} limit={limit:.3f} Nm")
        print("\n".join(lines), flush=True)

    def resume_locomotion(self, command_ranges=None):
        """保留落地点、姿态和速度，恢复 teacher 电机参数与命令。"""
        self.set_collect_jump_data(False)
        self.episode_length_buf.zero_()
        self.sample_locomotion_commands(command_ranges)
        return self.get_locomotion_observations()

    def __init__(self, *args, step_terrain=None, **kwargs):
        self.step_terrain = None if step_terrain is None else dict(step_terrain)
        if self.step_terrain is not None:
            for name in ("height", "distance", "length", "width"):
                value = float(self.step_terrain[name])
                if not math.isfinite(value) or value <= 0.0:
                    raise ValueError(f"step {name} must be finite and positive")
                self.step_terrain[name] = value
        super().__init__(*args, **kwargs)

    def _add_terrain(self):
        super()._add_terrain()
        if self.step_terrain is None:
            return
        step = self.step_terrain
        self.step_front_x = float(self.init_base_pos[0].item()) + step["distance"]
        self.step_center_y = float(self.init_base_pos[1].item())
        self.step_entity = self.scene.add_entity(
            gs.morphs.Box(
                pos=(self.step_front_x + step["length"] / 2, self.step_center_y, step["height"] / 2),
                size=(step["length"], step["width"], step["height"]),
                fixed=True,
                collision=True,
            ),
            surface=gs.surfaces.Default(color=(0.45, 0.55, 0.65)),
        )

    def wheels_on_step(self):
        """诊断回合末双轮是否接触台面；侧壁碰撞与仅在台面上方不算落台。"""
        if self.step_terrain is None:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        step = self.step_terrain
        positions = self.robot.get_links_pos(self.wheel_links_idx)
        x, y, bottom_z = positions[:, :, 0], positions[:, :, 1], positions[:, :, 2] - self.wheel_radius
        on_top = (
            (x >= self.step_front_x)
            & (x <= self.step_front_x + step["length"])
            & (torch.abs(y - self.step_center_y) <= step["width"] / 2)
            & (torch.abs(bottom_z - step["height"]) <= 0.02)
            & (self.wheel_contact > 0.5)
        )
        return torch.all(on_top, dim=1) & (self.base_contact < 0.5)
