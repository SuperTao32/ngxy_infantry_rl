"""CPU checks: python -m unittest discover -s experiments/mujoco/tests -v."""

import unittest

import mujoco
import numpy as np
import torch

from experiments.genesis.wheel_leg_infantry.tools.run_utils import load_run_configs
from experiments.mujoco.locomotion.locomotion_env import ROOT, MujocoLocomotionEnv


class LocomotionContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.configs = load_run_configs(ROOT / "log_shared/infantry_locomotion_v3/version_0001")
        cls.env = MujocoLocomotionEnv(cls.configs)

    def setUp(self):
        self.env.reset()

    def test_joint_mapping_and_initial_observation(self):
        env = self.env
        self.assertEqual([env.model.joint(n).dofadr[0] for n in env.joint_names], env.jd)
        self.assertNotEqual(env.jd, sorted(env.jd))
        obs = env.observations()["policy"][0]
        self.assertEqual(tuple(obs.shape), (33,))
        torch.testing.assert_close(obs[:7], torch.zeros(7))
        torch.testing.assert_close(obs[7:10], torch.tensor([0., 0., -1.]))
        torch.testing.assert_close(obs[10:13], env.commands * torch.tensor([
            env.scales["lin_vel"], env.scales["ang_vel"], env.scales["base_height"]]))
        torch.testing.assert_close(obs[-6:], torch.zeros(6))

    def test_latency_clipping_and_force_limits(self):
        env = self.env
        env.set_action(torch.full((6,), 100.0))
        np.testing.assert_allclose(env.target_joint, env.default.numpy())
        np.testing.assert_allclose(env.target_wheel, 0)
        previous = env.actions.clone()
        env.set_action(torch.full((6,), -100.0))
        np.testing.assert_allclose(env.target_wheel, previous[4:].numpy() * env.cfg["wheel_vel_scale"])
        self.assertTrue(torch.all(env.actions < 0))
        env.data.qvel[env.jd] = 1000
        env.data.qvel[env.wd] = -1000
        env._forces()
        np.testing.assert_allclose(env.data.qfrc_applied[env.jd], -np.ones(4) * env.cfg["joint_force_limit"])
        np.testing.assert_allclose(env.data.qfrc_applied[env.wd], np.ones(2) * env.cfg["wheel_force_limit"])

    def test_body_gravity_and_gyro_frame(self):
        env = self.env
        env.data.qpos[3:7] = [np.sqrt(.5), 0, np.sqrt(.5), 0]
        env.data.qvel[3:6] = [0.1, 0.2, 0.3]
        mujoco.mj_forward(env.model, env.data)
        env.observations()
        np.testing.assert_allclose(env.gravity, [1, 0, 0], atol=1e-6)
        np.testing.assert_allclose(env.data.sensor("gyro").data, [0.1, .2, .3], atol=1e-6)

    def test_spring_hold_and_clock(self):
        env = self.env
        obs = env.step(torch.zeros(6))
        self.assertAlmostEqual(env.data.time, .02)
        np.testing.assert_allclose(env.spring_force, env.cfg["gas_spring_preload_force"]
                                   + env.cfg["gas_spring_stiffness"] * env.cfg["gas_spring_max_compression"])
        self.assertTrue(torch.isfinite(obs["policy"]).all())
        env.reset()
        self.assertEqual(env.data.time, 0)
        self.assertEqual(env.estimated_velocity.item(), 0)

    def test_diagnostic_velocity_matches_position_difference(self):
        env = self.env
        # Rotate yaw 90 degrees: world +y is robot forward, regardless of inertia axes.
        env.data.qpos[3:7] = [np.sqrt(.5), 0, 0, np.sqrt(.5)]
        env.data.qvel[:6] = [0, .5, 0, 0, 0, .3]
        mujoco.mj_forward(env.model, env.data)
        env.observations()
        diagnostic = env.diagnostics()
        self.assertAlmostEqual(diagnostic["vx"], .5)
        self.assertAlmostEqual(diagnostic["wz"], .3)
        previous = env.data.xpos[env.base_id].copy()
        rotation = env.data.xmat[env.base_id].reshape(3, 3).copy()
        mujoco.mj_integratePos(env.model, env.data.qpos, env.data.qvel, 1e-6)
        mujoco.mj_forward(env.model, env.data)
        finite_difference = rotation.T @ ((env.data.xpos[env.base_id] - previous) / 1e-6)
        self.assertAlmostEqual(diagnostic["vx"], finite_difference[0])

    def test_bad_action_rejected(self):
        for action in (torch.zeros(5), torch.full((6,), float("nan"))):
            with self.assertRaises(ValueError):
                self.env.set_action(action)


if __name__ == "__main__":
    unittest.main()
