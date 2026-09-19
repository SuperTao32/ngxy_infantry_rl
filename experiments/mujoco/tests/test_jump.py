"""Jump switching, observation history, and landing regression checks."""

import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.genesis.wheel_leg_infantry.tools.run_utils import load_run_configs
from experiments.mujoco.locomotion.locomotion_env import ROOT
from experiments.mujoco.jump.jump_env import MujocoJumpEnv
from experiments.mujoco.jump.jump_eval import handle_key


class JumpContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        jump = load_run_configs(ROOT / "log_shared/infantry_jump_v6/version_0006")
        source = load_run_configs(ROOT / "log_shared/infantry_locomotion_v3/version_0001")
        cls.env = MujocoJumpEnv(jump, source)

    def setUp(self):
        self.env.command_vx = 0
        self.env.handoff = "landing"
        self.env.reset()

    def test_space_preserves_physics_and_switches_motor_profile(self):
        env = self.env
        env.sim.actions[:] = .2
        env.sim.estimated_velocity[:] = .1
        before = env.sim.data.qpos.copy(), env.sim.data.qvel.copy(), env.sim.data.time
        self.assertEqual(handle_key(env, 32, (-2, 2)), "jump triggered")
        self.assertEqual(env.mode, "jump")
        np.testing.assert_array_equal(env.sim.data.qpos, before[0])
        np.testing.assert_array_equal(env.sim.data.qvel, before[1])
        self.assertEqual(env.sim.data.time, before[2])
        self.assertAlmostEqual(env.sim.estimated_velocity.item(), .1)
        self.assertEqual(env.sim.cfg["joint_kp"], 80)
        self.assertEqual(env.sim.cfg["joint_kd"], 1)
        obs = env.observations()["policy"][0]
        self.assertEqual(tuple(obs.shape), (45,))
        torch.testing.assert_close(obs[27:33], torch.full((6,), .2))
        torch.testing.assert_close(obs[33:39], obs[27:33])
        torch.testing.assert_close(obs[39:], torch.tensor([0., 1., 0., 1., 0., 1.]))

    def test_action_history_and_latency_survive_trigger(self):
        env = self.env
        env.sim.actions[:] = .1
        env.trigger_jump()
        obs = env.step(torch.full((6,), -.2))["policy"][0]
        np.testing.assert_allclose(env.sim.target_wheel, 7., atol=1e-6)
        torch.testing.assert_close(obs[27:33], torch.full((6,), -.2))
        torch.testing.assert_close(obs[33:39], torch.full((6,), .1))
        obs = env.step(torch.full((6,), .3))["policy"][0]
        torch.testing.assert_close(obs[33:39], torch.full((6,), -.2))

    def test_locked_velocity_reference_and_no_queued_space(self):
        env = self.env
        env.command_vx = .5
        env.trigger_jump()
        count = env.jump_count
        env.jump_step = 15  # t=0.30, an exact reference knot.
        handle_key(env, ord("I"), (-2, 2))
        handle_key(env, 32, (-2, 2))
        self.assertEqual(env.jump_count, count)
        self.assertEqual(env.jump_step, 15)
        env.observations()
        torch.testing.assert_close(env.sim.commands, torch.tensor([.5, 0., .2]))

    def test_contact_dropout_does_not_cause_early_handoff(self):
        env = self.env
        env.trigger_jump()
        with patch.object(env.sim, "step"), patch.object(env, "_read_contacts"):
            env.takeoff_contact_vz = .5
            env.wheel_contact[:] = False
            env.clearance = 0.0
            env.step(torch.zeros(6))
            env.wheel_contact[:] = True
            env.step(torch.zeros(6))
            self.assertEqual(env.mode, "jump")
            self.assertFalse(env.has_taken_off)
            env.takeoff_contact_vz = .5
            env.wheel_contact[:] = False
            env.clearance = .02
            env.step(torch.zeros(6))
            self.assertTrue(env.has_taken_off)
            env.wheel_contact[:] = True
            position = env.sim.data.qpos.copy()
            obs = env.step(torch.zeros(6))
        self.assertEqual(env.mode, "locomotion")
        self.assertEqual(obs["policy"].shape[-1], 33)
        self.assertEqual(env.sim.cfg["joint_kp"], 60)
        self.assertEqual(env.sim.cfg["joint_kd"], 3)
        self.assertEqual(env.last_result["reason"], "both_wheels_landed")
        np.testing.assert_array_equal(env.sim.data.qpos, position)
        self.assertTrue(env.trigger_jump())

    def test_horizon_and_reset_restore_locomotion(self):
        env = self.env
        env.trigger_jump()
        env.jump_step = env.horizon - 1
        with patch.object(env.sim, "step"), patch.object(env, "_read_contacts"):
            env.step(torch.zeros(6))
        self.assertEqual(env.last_result["reason"], "horizon")
        self.assertEqual(env.mode, "locomotion")
        env.trigger_jump()
        handle_key(env, ord("R"), (-2, 2))
        self.assertEqual(env.mode, "locomotion")
        self.assertEqual(env.sim.cfg["joint_kp"], 60)
        self.assertEqual(env.sim.data.time, 0)
        torch.testing.assert_close(env.last_actions, torch.zeros(6))


if __name__ == "__main__":
    unittest.main()
