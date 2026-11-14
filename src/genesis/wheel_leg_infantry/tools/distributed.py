"""Single-node torchrun setup shared by simulation and RSL-RL."""

from contextlib import contextmanager
from dataclasses import dataclass
import os
import signal
import sys

import torch
import torch.distributed as dist


def launch_training(num_gpus, argv, *, cpu=False):
    """Replace the CLI with torchrun; existing workers continue inline."""
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    if cpu and (world_size > 1 or (num_gpus is not None and num_gpus > 1)):
        raise ValueError("--cpu cannot be used with multi-GPU training")
    if "LOCAL_RANK" in os.environ or world_size > 1:
        if num_gpus is not None and num_gpus != world_size:
            raise ValueError(f"--num-gpus {num_gpus} does not match torchrun WORLD_SIZE={world_size}")
        return False
    if num_gpus is None or num_gpus == 1:
        return False
    available = torch.cuda.device_count()
    if num_gpus > available:
        raise ValueError(f"--num-gpus {num_gpus} requires {num_gpus} visible CUDA GPUs, but found {available}; check CUDA_VISIBLE_DEVICES")
    if not dist.is_nccl_available():
        raise ValueError("Multi-GPU training requires a PyTorch build with NCCL support")
    print(f"[train] launching {num_gpus} GPU workers", flush=True)
    # Keep the externally managed PID as the torchrun agent. Its signal handlers
    # forward termination to workers and enforce a graceful-shutdown deadline,
    # including workers that torchrun starts in separate process groups.
    os.execv(sys.executable, [
        sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
        f"--nproc-per-node={num_gpus}", "--module", "src.genesis.wheel_leg_infantry.train", *argv,
    ])
    # A successful exec never returns; this also permits a mocked exec in tests.
    return True


@dataclass(frozen=True)
class TrainingWorkers:
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1

    @property
    def distributed(self):
        return self.world_size > 1

    def close(self, *, synchronize=False):
        """Stop NCCL watchdogs before Genesis tears down its CUDA runtime."""
        if self.distributed and dist.is_initialized():
            try:
                # Only a completed run may wait for peers (including rank 0's
                # final checkpoint). Failure/interruption cleanup must not wait.
                if synchronize:
                    dist.barrier()
            finally:
                dist.destroy_process_group()

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
    devices = [device.strip() for device in visible.split(",")] if visible is not None else [str(rank) for rank in range(workers.world_size)]
    if workers.world_size > len(devices):
        raise ValueError("torchrun worker count exceeds CUDA_VISIBLE_DEVICES")
    if not all(device.isdecimal() for device in devices[:workers.world_size]):
        raise ValueError("Use numeric GPU indices in CUDA_VISIBLE_DEVICES for Genesis multi-GPU training")
    if len(set(devices[:workers.world_size])) != workers.world_size:
        raise ValueError("Each training worker requires a distinct GPU in CUDA_VISIBLE_DEVICES")
    physical_gpu = devices[workers.local_rank]
    # Pin before ANY CUDA calls. Genesis/Quadrants use the physical device ID;
    # PyTorch and the runner address the sole visible device as cuda:0.
    os.environ["CUDA_VISIBLE_DEVICES"] = physical_gpu
    os.environ["QD_VISIBLE_DEVICE"] = physical_gpu
    if torch.cuda.device_count() != 1:
        raise ValueError(f"GPU {physical_gpu} is unavailable to training worker {workers.rank}")
    torch.cuda.set_device(0)
    dist.init_process_group(backend="nccl", device_id=torch.device("cuda", 0))
    # torchrun terminates workers with SIGTERM on interruption or peer failure.
    # Let Python unwind the training/Genesis contexts instead of exiting directly.
    def terminate(signum, frame):
        raise SystemExit(128 + signum)

    previous_terminate = signal.signal(signal.SIGTERM, terminate)
    try:
        yield workers
    finally:
        try:
            workers.close()
        finally:
            signal.signal(signal.SIGTERM, previous_terminate)
