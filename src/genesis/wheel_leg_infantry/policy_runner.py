"""统一 PPO runner：一个 actor/critic，保存所有任务的课程状态。"""

from copy import deepcopy

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
