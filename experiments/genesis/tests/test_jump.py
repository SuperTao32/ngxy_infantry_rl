"""模式、采样距离和落台几何的 CPU 检查。"""

from copy import deepcopy
from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from experiments.genesis.wheel_leg_infantry.tasks.jump.config import get_cfgs, validate_configs
from experiments.genesis.wheel_leg_infantry.tasks.jump.geometry import target_wheel_support, trigger_distance
from experiments.genesis.wheel_leg_infantry.tasks.jump.terrain import JumpTerrain


class JumpTests(unittest.TestCase):
    def setUp(self):
        self.env, self.obs, *_ = get_cfgs()
        self.cfg = self.env["jump_modes"]

    def test_observation_contract_and_fixed_modes(self):
        self.assertEqual(self.obs["num_policy_obs"], 48)
        self.assertEqual(self.obs["num_critic_obs"], 67)
        self.assertEqual(self.cfg["step_heights_m"], [0., .2, .4])
        self.assertEqual(self.obs["locomotion_policy_obs_dim"], 33)

    def test_three_phase_schedule_starts_with_takeoff(self):
        from experiments.genesis.wheel_leg_infantry.tasks.jump.config import PHASE_NAMES
        from experiments.genesis.wheel_leg_infantry.tasks.jump.phase import (
            scheduled_phase_index, validate_phase_durations,
        )
        from experiments.genesis.wheel_leg_infantry.tasks.jump.rewards import JumpRewards

        self.assertEqual(PHASE_NAMES, ("takeoff", "flight", "landing"))
        durations = validate_phase_durations(self.env["jump_phase_durations_s"])
        elapsed = torch.tensor([0., .149, .15, .599, .6, 1.4])
        torch.testing.assert_close(
            scheduled_phase_index(elapsed, durations), torch.tensor([0, 0, 1, 1, 2, 2]),
        )
        self.assertAlmostEqual(sum(durations), self.env["episode_length_s"])
        reward_cfg = get_cfgs()[2]
        self.assertFalse(any("crouch" in key for key in reward_cfg["reward_scales"]))
        self.assertFalse(any("crouch" in key for key in dir(JumpRewards)))
        for phase in PHASE_NAMES:
            for value in (0., -1., float("nan"), float("inf")):
                invalid = dict(self.env["jump_phase_durations_s"], **{phase: value})
                with self.subTest(phase=phase, value=value), self.assertRaises(ValueError):
                    validate_phase_durations(invalid)
        with self.assertRaises(ValueError):
            validate_phase_durations(dict(self.env["jump_phase_durations_s"], crouch=0.))

    def test_single_config_and_training_entry_point(self):
        from experiments.genesis.wheel_leg_infantry.tasks import jump
        from experiments.genesis.wheel_leg_infantry.tasks.jump.train import _parse_args

        task_dir = Path(jump.__file__).parent
        self.assertEqual(sorted(p.name for p in task_dir.glob("config*.py")), ["config.py"])
        self.assertFalse((task_dir.parent / "multi_jump").exists())
        args = _parse_args(["--dry-run"])
        self.assertEqual(args.exp_name, "infantry_jump")
        self.assertFalse(hasattr(args, "config"))
        self.assertEqual(self.env["jump_motor_params"]["joint_kp"], 100.)

    def test_height_reference_tracks_base_to_wheel_bottom_distance(self):
        from experiments.genesis.wheel_leg_infantry.core.kinematics import (
            compute_mean_base_to_wheel_bottom_distance,
        )
        from experiments.genesis.wheel_leg_infantry.tasks.jump.rewards import JumpRewards

        base_pos = torch.tensor([[0., 0., .6]])
        wheel_centers = torch.tensor([[[0., -.1, .30], [0., .1, .34]]])
        wheel_radius = .06
        distance = compute_mean_base_to_wheel_bottom_distance(base_pos, wheel_centers, wheel_radius)
        torch.testing.assert_close(distance, torch.tensor([.34]))
        env = SimpleNamespace(
            base_height=torch.tensor([.6]),
            base_to_wheel_bottom_distance=distance,
            commands=torch.tensor([[0., 0., .34]]),
            reward_cfg={"height_reference_tolerance_m": .02, "height_reference_sigma": .04},
            flight_gate=torch.ones(1),
        )
        reward = JumpRewards._reward_height_reference_tracking(env)
        torch.testing.assert_close(reward, torch.ones(1))
        # Translating the whole robot or changing local ground does not change tracking.
        base_pos[:, 2] += .4
        wheel_centers[:, :, 2] += .4
        env.base_height.fill_(.1)
        env.base_to_wheel_bottom_distance = compute_mean_base_to_wheel_bottom_distance(
            base_pos, wheel_centers, wheel_radius,
        )
        torch.testing.assert_close(JumpRewards._reward_height_reference_tracking(env), reward)
        # Retracting both wheels changes the relative distance even at fixed base height.
        wheel_centers[:, :, 2] += .1
        env.base_to_wheel_bottom_distance = compute_mean_base_to_wheel_bottom_distance(
            base_pos, wheel_centers, wheel_radius,
        )
        expected = torch.exp(-torch.tensor([.08 ** 2 / .04]))
        torch.testing.assert_close(JumpRewards._reward_height_reference_tracking(env), expected)
        # The configured 2 cm tolerance leaves a small distance error unpenalized.
        env.base_to_wheel_bottom_distance = torch.tensor([.33])
        torch.testing.assert_close(JumpRewards._reward_height_reference_tracking(env), reward)

    def test_height_reference_keeps_existing_flight_gate(self):
        from experiments.genesis.wheel_leg_infantry.tasks.jump.rewards import JumpRewards

        env = SimpleNamespace(
            base_to_wheel_bottom_distance=torch.tensor([.6, .6]),
            commands=torch.tensor([[0., 0., .6], [0., 0., .6]]),
            reward_cfg={"height_reference_tolerance_m": .02, "height_reference_sigma": .04},
            flight_gate=torch.tensor([0., 1.]),
        )
        torch.testing.assert_close(JumpRewards._reward_height_reference_tracking(env), torch.tensor([0., 1.]))

    def test_distance_depends_on_speed_and_mode(self):
        speed = torch.tensor([0., 1., 1., 1.5, 1.5])
        mode = torch.tensor([0, 1, 2, 1, 2])
        actual = trigger_distance(speed, mode, self.cfg["distance_tables"])
        torch.testing.assert_close(actual, torch.tensor([0., .3, .4, .45, .6]))
        for v in (-.1, 2.1, float("nan")):
            with self.assertRaises(ValueError):
                trigger_distance(torch.tensor([v]), torch.tensor([1]), self.cfg["distance_tables"])

    def test_terrain_geometry_and_mode_assignment(self):
        self.cfg["assignment"] = "cyclic"
        terrain = JumpTerrain(self.cfg)
        terrain.bind_entity(None, device="cpu", dtype=torch.float32)
        centers, modes = terrain.sample_spawn_tiles(torch.arange(6))
        torch.testing.assert_close(modes, torch.tensor([0, 1, 2, 0, 1, 2]))
        torch.testing.assert_close(centers[:, 0], torch.full((6,), -30.))
        torch.testing.assert_close(centers[:, 1], torch.tensor([0., 4., 8., 0., 4., 8.]))
        points = torch.tensor([[1., 0.], [1., 4.], [1., 8.], [-.1, 8.], [7., 8.], [1., 6.]])
        torch.testing.assert_close(terrain.height_at(points), torch.tensor([0., .2, .4, 0., 0., 0.]))

    def test_probabilities_can_select_single_mode(self):
        # 课程允许整数权重（例如 [1, 0, 0]），包括尚未 bind_entity 的 CPU 采样。
        for probabilities in ([1, 0, 0], [0, 1, 0], [0, 0, 2], [0., 0., 1.]):
            with self.subTest(probabilities=probabilities):
                self.cfg["mode_probabilities"] = probabilities
                terrain = JumpTerrain(self.cfg)
                _, modes = terrain.sample_spawn_tiles(torch.arange(20))
                expected_mode = next(i for i, weight in enumerate(probabilities) if weight > 0)
                self.assertTrue(torch.all(modes == expected_mode))
                self.assertEqual(modes.dtype, torch.long)

    def test_landing_rejects_sidewall_ground_air_and_edge(self):
        modes = torch.tensor([0, 1, 2, 1, 1, 1, 1])
        positions = torch.tensor([
            [[0., -.1, .06], [0., .1, .06]],
            [[.3, 3.9, .26], [.3, 4.1, .26]],
            [[.3, 7.9, .46], [.3, 8.1, .46]],
            [[-.01, 3.9, .16], [-.01, 4.1, .16]],  # 立面
            [[-.2, 3.9, .06], [-.2, 4.1, .06]],  # 台阶下方
            [[.3, 3.9, .36], [.3, 4.1, .36]],  # 台面上方
            [[.01, 3.9, .26], [.01, 4.1, .26]],  # 太靠近边缘
        ])
        contacts = torch.ones(7, 2)
        support = target_wheel_support(positions, contacts, modes, self.cfg, .06)
        torch.testing.assert_close(support.all(dim=1), torch.tensor([True, True, True, False, False, False, False]))
        contacts[1, 0] = 0
        self.assertFalse(target_wheel_support(positions, contacts, modes, self.cfg, .06).all(dim=1)[1])

    def test_success_requires_sustained_support_and_real_takeoff(self):
        import genesis as gs
        from experiments.genesis.wheel_leg_infantry.tasks.jump.env import JumpEnv

        env = JumpEnv.__new__(JumpEnv)
        env.task_cfg = self.cfg
        env.collect_jump_data = True
        env.jump_mode = torch.arange(3)
        env.wheel_center_pos = torch.tensor([
            [[0., -.1, .06], [0., .1, .06]],
            [[.3, 3.9, .26], [.3, 4.1, .26]],
            [[.3, 7.9, .46], [.3, 8.1, .46]],
        ])
        env.wheel_contact = torch.ones(3, 2)
        env.wheel_radius = .06
        env.landing_surface_height = torch.tensor([0., .2, .4])
        env.has_taken_off = torch.tensor([False, True, True])
        env.has_landed = torch.ones(3, dtype=torch.bool)
        env.jump_landing_event = torch.zeros(3)
        env.jump_invalid = torch.zeros(3)
        env.jump_landing_gate = torch.ones(3)
        env.rebound_seen = torch.zeros(3)
        env.projected_gravity = torch.tensor([[0., 0., -1.]]).expand(3, -1)
        env.world_vertical_velocity = torch.zeros(3)
        env.stable_landing_steps = torch.zeros(3, dtype=torch.long)
        env.required_stable_steps = 8
        env.jump_reward_state = SimpleNamespace(peak_clearance=torch.full((3,), .5))
        env.wheel_clearance_target = torch.tensor([.3, .3, .5])
        env.task_success = torch.zeros(3, dtype=torch.bool)
        env.success_seen = torch.zeros(3, dtype=torch.bool)
        env.failure_seen = torch.zeros(3, dtype=torch.bool)
        env.task_success_event = torch.zeros(3)
        env.task_failure_event = torch.zeros(3)
        env.episode_length_buf = torch.full((3,), 10)
        env.max_episode_length = 70
        with patch.object(JumpEnv, "_update_jump_state"), patch.object(gs, "tc_float", torch.float32, create=True):
            for _ in range(7):
                env._update_task_state()
                self.assertFalse(env.task_success.any())
            env._update_task_state()
            torch.testing.assert_close(env.task_success, torch.tensor([False, True, True]))
            torch.testing.assert_close(env.task_success_event, torch.tensor([0., 1., 1.]))
            env._update_task_state()
            self.assertFalse(env.task_success_event.any())  # 一次成功只奖励一次。
            env.wheel_contact[2, 0] = 0
            env._update_task_state()
            self.assertFalse(env.task_success[2])  # 丢失支撑后重新累计稳定时间。
            self.assertEqual(env.stable_landing_steps[2], 0)
            env.episode_length_buf.fill_(70)
            env._update_task_state()
            torch.testing.assert_close(env.task_failure_event, torch.tensor([1., 0., 1.]))

    def test_terminal_statistics_ignore_padding_steps(self):
        from experiments.genesis.wheel_leg_infantry.tasks.jump.env import JumpEnv

        env = JumpEnv.__new__(JumpEnv)
        env.collect_jump_data = True
        env.finished_jump = torch.zeros(3, dtype=torch.bool)
        env.terminal_jump_stats = {}
        env.extras = {}
        env.jump_mode = torch.arange(3)
        sums = torch.tensor([-300., 0., 0.])
        env._current_jump_episode_stats = lambda: {
            "reward_death": sums.clone(), "task_success": torch.zeros(3),
        }
        env._reset_idx(torch.tensor([True, False, False]))
        sums[:] = torch.tensor([-600., -300., 0.])
        env._reset_idx(torch.tensor([True, True, False]))
        sums[:] = torch.tensor([-900., -600., 0.])
        stats = env._jump_episode_stats()
        torch.testing.assert_close(stats["reward_death"], torch.tensor([-300., -300., 0.]))
        self.assertEqual(stats["success_step_20cm"].numel(), 1)

    def test_config_rejects_unsafe_or_ambiguous_geometry(self):
        changes = [
            ("clearance_targets_m", [0., .3, .5]),
            ("distance_jitter_m", .3),
            ("lane_spacing_m", 1.),
            ("landing_margin_m", 2.),
            ("mode_names", ["step_20cm", "flat", "step_40cm"]),
        ]
        for key, value in changes:
            cfg = deepcopy(self.env)
            cfg["jump_modes"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_configs(cfg, self.obs)


if __name__ == "__main__":
    unittest.main()
