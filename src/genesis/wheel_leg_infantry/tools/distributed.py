"""Single-node torchrun setup shared by simulation and RSL-RL."""

from contextlib import contextmanager
from dataclasses import dataclass
import os

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class TrainingWorkers:
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1

    @property
    def distributed(self):
        return self.world_size > 1

    def primary(self, function, *args, **kwargs):
        """Run filesystem decisions once and broadcast their result or failure."""
        if not self.distributed:
            return function(*args, **kwargs)
        result = [None, None]
        if self.rank == 0:
            try:
                result[0] = function(*args, **kwargs)
            except Exception as exc:
                result[1] = f"{type(exc).__name__}: {exc}"
        dist.broadcast_object_list(result, src=0)
        if result[1] is not None:
            raise RuntimeError(f"Primary training worker failed: {result[1]}")
        return result[0]

    def metadata(self, num_envs, seed):
        return dict(world_size=self.world_size, num_envs_per_gpu=num_envs,
                    total_num_envs=num_envs * self.world_size,
                    worker_seeds=[seed + rank for rank in range(self.world_size)])


@contextmanager
def training_workers():
    workers = TrainingWorkers(int(os.getenv("RANK", "0")), int(os.getenv("LOCAL_RANK", "0")),
                              int(os.getenv("WORLD_SIZE", "1")))
    if not workers.distributed:
        yield workers
        return
    if int(os.getenv("LOCAL_WORLD_SIZE", str(workers.world_size))) != workers.world_size:
        raise ValueError("wheel_leg_infantry currently supports single-node torchrun only")
    if not 0 <= workers.rank < workers.world_size or workers.local_rank != workers.rank:
        raise ValueError("Invalid single-node torchrun rank configuration")
    visible = os.getenv("CUDA_VISIBLE_DEVICES")
    if visible is not None and workers.world_size > len(visible.split(",")):
        raise ValueError("torchrun worker count exceeds CUDA_VISIBLE_DEVICES")
    physical_gpu = visible.split(",")[workers.local_rank].strip() if visible is not None else str(workers.local_rank)
    if not physical_gpu.isdecimal():
        raise ValueError("Use numeric GPU indices in CUDA_VISIBLE_DEVICES for Genesis multi-GPU training")
    # Pin before ANY CUDA calls. Genesis/Quadrants use the physical device ID;
    # PyTorch and the runner address the sole visible device as cuda:0.
    os.environ["CUDA_VISIBLE_DEVICES"] = physical_gpu
    os.environ["QD_VISIBLE_DEVICE"] = physical_gpu
    if torch.cuda.device_count() != 1:
        raise ValueError(f"GPU {physical_gpu} is unavailable to training worker {workers.rank}")
    torch.cuda.set_device(0)
    dist.init_process_group(backend="nccl", device_id=torch.device("cuda", 0))
    try:
        yield workers
    finally:
        # No barrier here: a failed rank must not strand its peers during teardown.
        dist.destroy_process_group()
