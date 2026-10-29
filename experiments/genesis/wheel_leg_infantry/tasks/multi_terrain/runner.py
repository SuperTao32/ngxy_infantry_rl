"""复用 RSL-RL checkpoint 的 infos 保存各任务成功率窗口与等级。"""

from .config import TASK_NAMES


def _expand_legacy_critic_input(module, state_dict, prefix, *args):
    """为相同基础观测布局的平地 critic 补齐地形编码。"""
    key = prefix + "mlp.0.weight"
    weight = state_dict.get(key)
    target = module.mlp[0].weight
    if (weight is not None and weight.ndim == 2 and weight.shape[0] == target.shape[0]
            and target.shape[1] - weight.shape[1] == len(TASK_NAMES)):
        shared = min(weight.shape[1], target.shape[1])
        expanded = weight.new_zeros(target.shape)
        expanded[:, :shared] = weight[:, :shared]
        state_dict[key] = expanded
        # 支持启用观测归一化的快照；旧通道统计保持不变，新增通道使用初始统计。
        for name, buffer in module.named_buffers():
            saved = state_dict.get(prefix + name)
            if name.startswith("obs_normalizer.") and saved is not None and saved.ndim and saved.shape[-1] == weight.shape[1]:
                expanded_buffer = buffer.clone()
                expanded_buffer[..., :shared] = saved[..., :shared]
                state_dict[prefix + name] = expanded_buffer
        print(f"[train] migrated legacy critic input: {weight.shape[1]} -> {target.shape[1]}")


def task_runner_class(base, workers=None):
    class TaskRunner(base):
        def _configure_multi_gpu(self):
            if workers is None or not workers.distributed:
                return super()._configure_multi_gpu()
            # Startup already initialized NCCL to coordinate resume and run paths.
            # Unlike the upstream runner, each worker sees only its assigned GPU.
            if self.device != "cuda:0":
                raise ValueError(f"Runner device {self.device} does not match the isolated GPU cuda:0")
            self.is_distributed = True
            self.gpu_global_rank = workers.rank
            self.gpu_local_rank = workers.local_rank
            self.gpu_world_size = workers.world_size
            self.cfg["multi_gpu"] = dict(global_rank=workers.rank, local_rank=workers.local_rank,
                                         world_size=workers.world_size)

        def save(self, path, infos=None):
            infos = dict(infos or {})
            infos["terrain_curriculum"] = self.env.terrain_state_dict()
            if hasattr(self.env, "parking_state_dict"):
                infos["parking_curriculum"] = self.env.parking_state_dict()
            return super().save(path, infos)

        def load(self, path, load_cfg=None, strict=True, map_location=None):
            hook = None
            # 仅迁移权重；完整续训必须严格匹配模型和优化器维度。
            if (load_cfg is not None and load_cfg.get("critic", True)
                    and not load_cfg.get("optimizer", True) and not load_cfg.get("iteration", True)
                    and self.env.obs_cfg.get("critic_task_one_hot", False)):
                hook = self.alg._raw_critic.register_load_state_dict_pre_hook(_expand_legacy_critic_input)
            try:
                infos = super().load(path, load_cfg=load_cfg, strict=strict, map_location=map_location)
            finally:
                if hook is not None:
                    hook.remove()
            # --load-weights 明确重启课程；完整恢复时在 set_training_iteration/reset 应用。
            if load_cfg is None or load_cfg.get("iteration", True):
                self.env._pending_terrain_state = (infos or {}).get("terrain_curriculum")
                self.env._pending_parking_state = (infos or {}).get("parking_curriculum")
            return infos

    return TaskRunner
