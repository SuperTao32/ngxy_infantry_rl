"""Jump 模式概率课程：阶段切换、累计继承及采样时机。"""

from copy import deepcopy
import unittest

import torch

from experiments.genesis.wheel_leg_infantry.core.curriculum import CurriculumManager
from experiments.genesis.wheel_leg_infantry.tasks.jump.config import get_cfgs, validate_configs
from experiments.genesis.wheel_leg_infantry.tasks.jump.env import JumpEnv
from experiments.genesis.wheel_leg_infantry.tasks.jump.terrain import JumpTerrain


class JumpCurriculumTests(unittest.TestCase):
    def test_initial_probabilities_come_from_first_stage_without_aliasing(self):
        env_cfg, _, _, _, curriculum_cfg = get_cfgs()
        initial = curriculum_cfg["stages"][0]["targets"]["terrain"]["mode_probabilities"]
        runtime = env_cfg["jump_modes"]["mode_probabilities"]
        self.assertEqual(runtime, initial)
        self.assertIsNot(runtime, initial)
        runtime[0] = 10.
        self.assertNotEqual(runtime, initial)

    def test_stage_changes_affect_next_sampling_and_inherit_when_omitted(self):
        env = JumpEnv.__new__(JumpEnv)
        env.task_cfg = get_cfgs()[0]["jump_modes"]
        env.jump_mode = torch.tensor([0, 1, 2])
        env.jump_mode_one_hot = torch.eye(3)
        terrain = JumpTerrain(env.task_cfg)
        curriculum_cfg = {
            "enabled": True,
            "stages": [
                {"name": "flat", "start_iteration": 0,
                 "targets": {"terrain": {"mode_probabilities": [1, 0, 0]}}},
                {"name": "step", "start_iteration": 500,
                 "targets": {"terrain": {"mode_probabilities": [0, 0, 1]}}},
                {"name": "inherit", "start_iteration": 1000, "targets": {}},
            ],
        }
        original_cfg = deepcopy(curriculum_cfg)
        curriculum = CurriculumManager(curriculum_cfg)
        curriculum.register_target("terrain", env._apply_terrain_curriculum)
        for iteration, expected_mode in ((0, 0), (499, 0), (500, 2), (1000, 2), (0, 0)):
            with self.subTest(iteration=iteration):
                curriculum.update(iteration)
                _, sampled = terrain.sample_spawn_tiles(torch.arange(30))
                self.assertTrue(torch.all(sampled == expected_mode))
                # Updating weights must not change the task already assigned to an active rollout.
                torch.testing.assert_close(env.jump_mode, torch.tensor([0, 1, 2]))
                torch.testing.assert_close(env.jump_mode_one_hot, torch.eye(3))
        self.assertEqual(curriculum_cfg, original_cfg)

    def test_invalid_stage_probabilities_fail_validation_and_do_not_mutate_runtime(self):
        env_cfg, obs_cfg, _, _, curriculum_cfg = get_cfgs()
        env = JumpEnv.__new__(JumpEnv)
        env.task_cfg = env_cfg["jump_modes"]
        original = list(env.task_cfg["mode_probabilities"])
        for values in ([0., 0., 0.], [-1., 1., 1.], [1., 0.], [1., float("nan"), 0.], [float("inf"), 0., 1.]):
            with self.subTest(values=values):
                curriculum_cfg["stages"][1]["targets"]["terrain"]["mode_probabilities"] = values
                with self.assertRaises(ValueError):
                    validate_configs(env_cfg, obs_cfg, curriculum_cfg)
                with self.assertRaises(ValueError):
                    env._apply_terrain_curriculum({"mode_probabilities": values})
                self.assertEqual(env.task_cfg["mode_probabilities"], original)

    def test_disabled_curriculum_preserves_eval_override_and_cyclic_ignores_weights(self):
        env = JumpEnv.__new__(JumpEnv)
        env.task_cfg = get_cfgs()[0]["jump_modes"]
        env.task_cfg["mode_probabilities"] = [0., 1., 0.]
        curriculum = CurriculumManager({"enabled": False, "stages": []})
        curriculum.register_target("terrain", env._apply_terrain_curriculum)
        self.assertFalse(curriculum.update(1000))
        terrain = JumpTerrain(env.task_cfg)
        _, sampled = terrain.sample_spawn_tiles(torch.arange(6))
        torch.testing.assert_close(sampled, torch.ones(6, dtype=torch.long))
        env.task_cfg["assignment"] = "cyclic"
        _, sampled = terrain.sample_spawn_tiles(torch.arange(6))
        torch.testing.assert_close(sampled, torch.tensor([0, 1, 2, 0, 1, 2]))


if __name__ == "__main__":
    unittest.main()
