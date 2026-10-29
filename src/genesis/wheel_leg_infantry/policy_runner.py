"""统一 PPO runner：一个 actor/critic，保存所有任务的课程状态。"""

from .tools.run_utils import load_runner_class


class PolicyRunner(load_runner_class()):
    def __init__(self, *args, workers=None, **kwargs):
        self.workers = workers
        super().__init__(*args, **kwargs)

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
            self.env.load_training_state_dict(state)
        # A resumed run starts fresh simulator episodes and recurrent trajectories.
        self.alg.actor.reset()
        self.alg.critic.reset()
        return infos
