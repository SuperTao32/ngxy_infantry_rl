"""PPO runner：保存环境课程及组件奖励配置。"""

from copy import deepcopy
import gc
import os
import time

import torch
from rsl_rl.utils import check_nan

from .tools.run_utils import load_runner_class
from .terrains.stages import active_terrain_types, terrain_weights_at_iteration


class PolicyRunner(load_runner_class()):
    def __init__(self, env, train_cfg, *args, workers=None, terrain_env_factory=None, **kwargs):
        self.workers = workers
        self.terrain_env_factory = terrain_env_factory
        self._terrain_cross_state = {}
        super().__init__(env, deepcopy(train_cfg), *args, **kwargs)

    def _environment_state(self):
        state = self.env.training_state_dict()
        # 暂时移出场景的地形仍保留其独立奖励配置，之后重新启用或续训时恢复。
        self._terrain_cross_state.update(state.get("terrain_cross", {}))
        state["terrain_cross"] = deepcopy(self._terrain_cross_state)
        return state

    def _rebuild_terrain_scene(self, iteration):
        weights = terrain_weights_at_iteration(self.env.terrain.config, self.env.curriculum_cfg, iteration)
        if weights is None or active_terrain_types(weights) == self.env.terrain.built_types:
            return False
        state = self._environment_state()
        state["iteration"] = iteration
        if self.logger.writer is not None:
            self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))
        if self.is_distributed:
            torch.distributed.barrier()
        print(f"[terrain-build] iteration={iteration}; types={','.join(sorted(active_terrain_types(weights)))}", flush=True)
        # 在旧场景释放后创建新场景，避免两套物理环境同时驻留显存。
        self.env.scene.destroy()
        self.env = None
        gc.collect()
        if str(self.device).startswith("cuda"):
            torch.cuda.empty_cache()
        self.env = self.terrain_env_factory(iteration)
        self.logger.env_cfg = self.env.cfg
        with torch.inference_mode():
            self.env.load_training_state_dict(state)
            self.alg.actor.reset()
            self.alg.critic.reset()
            for name in ("cur_reward_sum", "cur_episode_length", "cur_ereward_sum", "cur_ireward_sum"):
                value = getattr(self.logger, name, None)
                if value is not None:
                    value.zero_()
        if self.is_distributed:
            torch.distributed.barrier()
        return True

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        if self.terrain_env_factory is None:
            return super().learn(num_learning_iterations, init_at_random_ep_len)
        # 沿用 RSL-RL 的采样、更新和日志顺序；仅在完整 PPO 更新之间插入场景切换。
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
        obs = self.env.get_observations().to(self.device)
        self.alg.train_mode()
        if self.is_distributed:
            self.alg.broadcast_parameters()
        self.logger.init_logging_writer()
        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations
        for it in range(start_it, total_it):
            if self._rebuild_terrain_scene(it):
                obs = self.env.get_observations().to(self.device)
            start = time.time()
            with torch.inference_mode():
                for _ in range(self.cfg["num_steps_per_env"]):
                    actions = self.alg.act(obs)
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    if self.cfg.get("check_for_nan", True):
                        check_nan(obs, rewards, dones)
                    obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))
                    self.alg.process_env_step(obs, rewards, dones, extras)
                    intrinsic_rewards = self.alg.intrinsic_rewards if self.cfg["algorithm"]["rnd_cfg"] else None
                    self.logger.process_env_step(rewards, dones, extras, intrinsic_rewards)
                stop = time.time()
                collect_time = stop - start
                start = stop
                self.alg.compute_returns(obs)
            loss_dict = self.alg.update()
            learn_time = time.time() - start
            self.current_learning_iteration = it
            self.logger.log(
                it=it, start_it=start_it, total_it=total_it, collect_time=collect_time, learn_time=learn_time,
                loss_dict=loss_dict, learning_rate=self.alg.learning_rate, action_std=self.alg.get_policy().output_std,
                rnd_weight=self.alg.rnd.weight if self.cfg["algorithm"]["rnd_cfg"] else None,
            )
            if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))
        if self.logger.writer is not None:
            self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))
            self.logger.stop_logging_writer()

    def _configure_multi_gpu(self):
        workers = self.workers
        if workers is None or not workers.distributed:
            return super()._configure_multi_gpu()
        self.is_distributed = True
        self.gpu_global_rank, self.gpu_local_rank = workers.rank, workers.local_rank
        self.gpu_world_size = workers.world_size
        self.cfg["multi_gpu"] = dict(global_rank=workers.rank, local_rank=workers.local_rank, world_size=workers.world_size)

    def save(self, path, infos=None):
        infos = {**(infos or {}), "infantry_env": self._environment_state()}
        return super().save(path, infos)

    def load(self, path, load_cfg=None, strict=True, map_location=None, *, restore_reward_scales=True):
        # RSL-RL 在 rollout 的 inference_mode 中替换归一化缓存；重载前将其实体化为普通 Tensor。
        for model in (self.alg.actor, self.alg.critic):
            for module in model.modules():
                for name, buffer in module.named_buffers(recurse=False):
                    if buffer.is_inference():
                        setattr(module, name, buffer.clone())
        infos = super().load(path, load_cfg=load_cfg, strict=strict, map_location=map_location) or {}
        # 环境和 GRU 的运行时缓存也可能来自上一次 rollout；优化器加载仍在普通模式执行。
        with torch.inference_mode():
            if load_cfg is None or load_cfg.get("iteration", True):
                state = infos.get("infantry_env")
                if state is None:
                    raise ValueError("Checkpoint has no v2 environment state; use --load-weight to start a new run")
                self._terrain_cross_state = deepcopy(state.get("terrain_cross", {})) if restore_reward_scales else {}
                self.env.load_training_state_dict(state, restore_reward_scales=restore_reward_scales)
            self.alg.actor.reset()
            self.alg.critic.reset()
        return infos
