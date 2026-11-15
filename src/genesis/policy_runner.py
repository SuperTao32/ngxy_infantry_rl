"""PPO runner：保存环境课程及组件奖励配置。"""

from copy import deepcopy

import torch

from .tools.run_utils import load_runner_class


class PolicyRunner(load_runner_class()):
    def __init__(self, env, train_cfg, *args, workers=None, **kwargs):
        self.workers = workers
        super().__init__(env, deepcopy(train_cfg), *args, **kwargs)

    def _configure_multi_gpu(self):
        workers = self.workers
        if workers is None or not workers.distributed:
            return super()._configure_multi_gpu()
        self.is_distributed = True
        self.gpu_global_rank, self.gpu_local_rank = workers.rank, workers.local_rank
        self.gpu_world_size = workers.world_size
        self.cfg["multi_gpu"] = dict(global_rank=workers.rank, local_rank=workers.local_rank, world_size=workers.world_size)

    def save(self, path, infos=None):
        infos = {**(infos or {}), "infantry_env": self.env.training_state_dict()}
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
                self.env.load_training_state_dict(state, restore_reward_scales=restore_reward_scales)
            self.alg.actor.reset()
            self.alg.critic.reset()
        return infos
