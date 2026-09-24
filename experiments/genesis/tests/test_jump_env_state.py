"""无需构建 Genesis 场景的 jump 状态生命周期回归检查。"""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import genesis as gs
import torch

from experiments.genesis.wheel_leg_infantry.tasks.jump.config import get_cfgs
from experiments.genesis.wheel_leg_infantry.tasks.jump.env import JumpEnv
from experiments.genesis.wheel_leg_infantry.tasks.jump.phase import phase_encoding


class JumpEnvStateTests(unittest.TestCase):
    def setUp(self):
        dtype_patch = patch.object(gs, "tc_float", torch.float32, create=True)
        dtype_patch.start()
        self.addCleanup(dtype_patch.stop)
        env = JumpEnv.__new__(JumpEnv)
        env.env_cfg, env.obs_cfg, env.reward_cfg, _, _ = get_cfgs()
        env.task_cfg = env.env_cfg["jump_modes"]
        env.num_envs = 3
        env.dt = .02
        env.device = torch.device("cpu")
        env.phase_durations = tuple(env.env_cfg["jump_phase_durations_s"].values())
        env.phase_cycle_s = sum(env.phase_durations)
        env.max_episode_length = round(env.phase_cycle_s / env.dt)
        env.commands = torch.zeros(3, 3)
        env.wheel_contact = torch.ones(3, 2)
        env.base_contact = torch.zeros(3)
        env.projected_gravity = torch.tensor([[0., 0., -1.]]).repeat(3, 1)
        env.terrain_tile_index = torch.arange(3)
        env.episode_length_buf = torch.zeros(3, dtype=torch.long)
        env.base_pos = torch.tensor([[-1., 0., .22], [-1., 4., .22], [-1., 8., .22]])
        env.wheel_radius = .06
        env.wheel_links_idx = [0, 1]
        self.velocity = torch.zeros(3, 3)
        self.wheels = torch.tensor([
            [[-1., -.1, .06], [-1., .1, .06]],
            [[-1., 3.9, .06], [-1., 4.1, .06]],
            [[-1., 7.9, .06], [-1., 8.1, .06]],
        ])
        env.robot = SimpleNamespace(
            get_vel=lambda: self.velocity,
            get_links_pos=lambda _: self.wheels,
        )
        self.solver_error = torch.zeros(3, dtype=torch.bool)
        env.scene = SimpleNamespace(rigid_solver=SimpleNamespace(get_error_envs_mask=lambda: self.solver_error))
        env.extras = {}
        env.collect_jump_data = True
        env._initialize_task_buffers()
        env._reset_task_buffers(None)
        self.env = env

    def step_state(self):
        self.env.episode_length_buf += 1
        self.env._update_jump_state()

    def test_partial_reset_preserves_other_environment_history(self):
        env = self.env
        for buffer in env._jump_episode_buffers():
            buffer.fill_(1)
        env.phase_features.fill_(.5)
        env.jump_reward_state.peak_clearance[:] = torch.tensor([.2, .3, .4])
        env.jump_reward_state.takeoff_rewarded.fill_(True)
        env.terminal_jump_stats = {"score": torch.tensor([10., 20., 30.])}
        selected = torch.tensor([True, False, True])
        env._reset_task_buffers(selected)
        for buffer in env._jump_episode_buffers():
            self.assertFalse(buffer[selected].any())
            self.assertTrue((buffer[1] == 1).all())
        torch.testing.assert_close(env.jump_reward_state.peak_clearance, torch.tensor([0., .3, 0.]))
        torch.testing.assert_close(env.jump_reward_state.takeoff_rewarded, ~selected)
        torch.testing.assert_close(env.terminal_jump_stats["score"], torch.tensor([0., 20., 0.]))
        initial_phase = phase_encoding(torch.zeros(2), env.phase_cycle_s)
        torch.testing.assert_close(env.phase_features[selected], initial_phase)
        torch.testing.assert_close(env.phase_features[1], torch.full((6,), .5))
        torch.testing.assert_close(env.jump_mode_one_hot, torch.eye(3))
        torch.testing.assert_close(env.wheel_clearance_target, torch.tensor([.3, .3, .5]))
        torch.testing.assert_close(env.landing_surface_height, torch.tensor([0., .2, .4]))
        before = [value.clone() for value in env._jump_episode_buffers()]
        env._reset_task_buffers(torch.zeros(3, dtype=torch.bool))
        for actual, expected in zip(env._jump_episode_buffers(), before):
            torch.testing.assert_close(actual, expected)
        env._reset_task_buffers(None)
        self.assertFalse(env.terminal_jump_stats)
        self.assertFalse(any(buffer.any() for buffer in env._jump_episode_buffers()))
        torch.testing.assert_close(env.phase_features, phase_encoding(torch.zeros(3), env.phase_cycle_s))

    def test_takeoff_landing_events_and_rebound_history(self):
        env = self.env
        self.velocity[:, 2] = 1.
        self.step_state()  # 最后一个支撑拍锁存向上速度。
        self.assertFalse(env.has_taken_off.any())
        env.wheel_contact.zero_()
        self.wheels[:, :, 2] += .1
        self.step_state()
        self.assertTrue(env.jump_takeoff_event.all())
        self.assertTrue(env.flight_gate.all())
        torch.testing.assert_close(env.jump_stage, torch.ones(3, dtype=torch.long))
        self.assertFalse(env.jump_reward_state.takeoff_event.any())  # 物理离地先于合格起跳奖励。
        self.step_state()
        self.assertFalse(env.jump_takeoff_event.any())
        self.step_state()
        self.assertTrue(env.jump_reward_state.takeoff_event.all())
        self.velocity[:, 2] = -.5
        self.step_state()
        env.wheel_contact[0, 0] = 1  # 首次单轮触地就结束腾空。
        self.step_state()
        torch.testing.assert_close(env.jump_landing_event, torch.tensor([1., 0., 0.]))
        torch.testing.assert_close(env.impact_vertical_speed, torch.tensor([.5, 0., 0.]))
        torch.testing.assert_close(env.jump_stage, torch.tensor([2, 1, 1]))
        self.assertFalse(env.flight_gate[0])
        self.assertTrue(env.jump_reward_state.height_settlement_event[0])
        self.step_state()
        self.assertFalse(env.jump_landing_event.any())
        self.assertFalse(env.impact_vertical_speed.any())
        env.wheel_contact[0].zero_()
        self.step_state()
        self.assertTrue(env.landing_airborne_gate[0])
        self.assertTrue(env.rebound_seen[0])
        env.wheel_contact[0].fill_(1)
        self.step_state()
        self.assertFalse(env.landing_airborne_gate[0])
        self.assertTrue(env.rebound_seen[0])  # 接触恢复不能撤销二次起跳失败。
        torch.testing.assert_close(env._task_termination(), torch.tensor([True, False, False]))

    def test_solver_errors_are_not_classified_as_normal_task_failure(self):
        env = self.env
        env.jump_invalid[:] = torch.tensor([1., 1., 0.])
        env.rebound_seen[2] = 1
        self.solver_error[1:] = True
        torch.testing.assert_close(env._task_termination(), torch.ones(3, dtype=torch.bool))
        torch.testing.assert_close(env.extras["jump_task_termination"], torch.tensor([True, False, False]))
        self.assertFalse(env.extras["jump_rebound_termination"].any())
        env.collect_jump_data = False
        self.assertFalse(env._task_termination().any())  # teacher 预热不适用 jump 终止规则。


if __name__ == "__main__":
    unittest.main()
