"""统一 PPO runner：一个 actor/critic，保存所有任务的课程状态。"""

from copy import deepcopy
from pathlib import Path

import torch

from .tools.run_utils import load_runner_class


class PolicyRunner(load_runner_class()):
    def __init__(self, env, train_cfg, *args, workers=None, allow_new_tasks=False, **kwargs):
        self.workers = workers
        self.allow_new_tasks = allow_new_tasks
        train_cfg = deepcopy(train_cfg)
        # 旧 checkpoint 的 PPO/RNN 配置沿用同一模型参数结构，训练时补上轨迹过滤。
        if train_cfg["algorithm"]["class_name"] == "PPO":
            train_cfg["algorithm"]["class_name"] = "src.genesis.wheel_leg_infantry.trajectory_ppo:TrajectoryPPO"
        for role in ("actor", "critic"):
            if train_cfg[role]["class_name"] == "RNNModel":
                train_cfg[role]["class_name"] = "src.genesis.wheel_leg_infantry.trajectory_ppo:TrajectoryRNNModel"
        super().__init__(env, train_cfg, *args, **kwargs)
        if callable(getattr(env, "rebuild_pending_terrain", None)):
            if torch.device(self.device) != torch.device(env.device):
                raise ValueError("Deferred terrain loading requires the policy and environment on the same device")
            self.alg.on_rollout_end = self._rebuild_terrain

    def _rebuild_terrain(self):
        iteration = getattr(self.env, "_scene_rebuild_iteration", None)
        if iteration is not None and self.logger.writer is not None:
            # The update just finished; save it before a potentially expensive
            # scene build so interruption can resume directly into the new stage.
            self.current_learning_iteration = iteration - 1
            self.save(str(Path(self.logger.log_dir) / f"model_{self.current_learning_iteration}.pt"))
        # New simulator tensors must be ordinary tensors even when learn() was
        # called from an inference context. Model/optimizer objects stay intact.
        with torch.inference_mode(False), torch.no_grad():
            if self.env.rebuild_pending_terrain():
                self.alg.actor.reset()
                self.alg.critic.reset()
                if self.logger.writer is not None:
                    self.logger.writer.add_scalar(
                        "Perf/terrain_rebuild_time", self.env.terrain_rebuild_seconds, self.env.training_iteration,
                    )

    def _configure_multi_gpu(self):
        workers = self.workers
        if workers is None or not workers.distributed:
            return super()._configure_multi_gpu()
        self.is_distributed = True
        self.gpu_global_rank, self.gpu_local_rank = workers.rank, workers.local_rank
        self.gpu_world_size = workers.world_size
        self.cfg["multi_gpu"] = dict(global_rank=workers.rank, local_rank=workers.local_rank,
                                     world_size=workers.world_size)

    def save(self, path, infos=None):
        infos = {**(infos or {}), "infantry_tasks": self.env.training_state_dict()}
        return super().save(path, infos)

    def load(self, path, load_cfg=None, strict=True, map_location=None):
        infos = super().load(path, load_cfg=load_cfg, strict=strict, map_location=map_location) or {}
        if load_cfg is None or load_cfg.get("iteration", True):
            state = infos.get("infantry_tasks")
            if state is None:
                raise ValueError("Checkpoint has no unified task state; use a unified GRU training checkpoint")
            self.env.load_training_state_dict(state, allow_new_tasks=self.allow_new_tasks)
        # A resumed run starts fresh simulator episodes and recurrent trajectories.
        self.alg.actor.reset()
        self.alg.critic.reset()
        return infos
