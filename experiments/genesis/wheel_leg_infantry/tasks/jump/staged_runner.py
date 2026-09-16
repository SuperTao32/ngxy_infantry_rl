"""Jump 的分阶段 PPO 训练流程。

每轮先由冻结的 locomotion teacher 预热，再采集一个完整 jump 周期并更新 PPO。
死亡/落地交接这一拍仍参与学习，之后的补齐步不参与优势归一化或 minibatch 采样。
落地后是否立即交回 teacher，由 handoff_on_landing 配置决定。
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from copy import copy, deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch


@dataclass(frozen=True)
class WarmupStats:
    """预热完成时的控制步数与连续稳定环境比例。"""

    steps: int
    stable_fraction: float


# ============ Teacher 准备与预热：不采集 PPO transition ============


def validate_warmup_cfg(warmup_cfg: Mapping) -> None:
    """检查预热时长、连续稳定条件、通过比例以及各项跟踪容差。"""
    min_steps = int(warmup_cfg["min_steps"])
    max_steps = int(warmup_cfg["max_steps"])
    stable_steps = int(warmup_cfg["stable_steps"])
    stable_fraction = float(warmup_cfg["stable_fraction"])
    if min_steps <= 0 or max_steps < min_steps:
        raise ValueError("warmup min_steps must be positive and cannot exceed max_steps")
    if stable_steps <= 0 or stable_steps > max_steps:
        raise ValueError("warmup stable_steps must be in [1, max_steps]")
    if not 0.0 < stable_fraction <= 1.0:
        raise ValueError("warmup stable_fraction must be in (0, 1]")
    for name in (
        "forward_velocity_tolerance",
        "yaw_rate_tolerance",
        "base_height_tolerance",
        "vertical_velocity_tolerance",
        "tilt_tolerance_deg",
    ):
        if float(warmup_cfg[name]) <= 0.0:
            raise ValueError(f"warmup {name} must be positive")


def build_frozen_locomotion_actor(
    source_configs: Mapping,
    checkpoint_path: str | Path,
    example_observations,
    num_actions: int,
    device,
):
    """在同一个 JumpEnv 上构造只读 locomotion actor，不再创建第二个 Genesis 场景。"""
    from rsl_rl.utils import resolve_callable

    # 复制源 actor 配置，避免 pop(class_name) 修改保存的训练配置。
    train_cfg = source_configs["train_cfg"]
    actor_cfg = deepcopy(train_cfg["actor"])
    actor_class = resolve_callable(actor_cfg.pop("class_name"))
    obs_groups = deepcopy(train_cfg.get("obs_groups", {"actor": ["policy"]}))
    if obs_groups.get("actor") != ["policy"]:
        raise ValueError(
            "locomotion checkpoint actor must consume exactly obs_groups['actor']=['policy']"
        )
    actor = actor_class(
        example_observations,
        obs_groups,
        "actor",
        num_actions,
        **actor_cfg,
    ).to(device)
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device)
    if "actor_state_dict" not in checkpoint:
        raise KeyError(f"checkpoint has no actor_state_dict: {checkpoint_path}")
    actor.load_state_dict(checkpoint["actor_state_dict"], strict=True)
    # teacher 只做推理：不更新参数，也不收集梯度。
    actor.eval()
    actor.requires_grad_(False)
    return actor


@torch.inference_mode()
def stabilize_with_locomotion(env, locomotion_actor, warmup_cfg: Mapping) -> WarmupStats:
    """执行不进 PPO storage 的 teacher 步，直到命令跟踪连续稳定。"""
    validate_warmup_cfg(warmup_cfg)
    min_steps = int(warmup_cfg["min_steps"])
    max_steps = int(warmup_cfg["max_steps"])
    stable_steps = int(warmup_cfg["stable_steps"])
    required_fraction = float(warmup_cfg["stable_fraction"])

    # 预热可以长于 jump 周期，临时延长 timeout，避免预热被 jump 时限打断。
    original_max_episode_length = env.max_episode_length
    env.max_episode_length = max(original_max_episode_length, max_steps + 1)
    try:
        observations = env.prepare_locomotion_warmup(warmup_cfg.get("command_ranges"))
        consecutive_stable = torch.zeros(
            env.num_envs, dtype=torch.long, device=env.device
        )
        last_ready_fraction = 0.0
        last_diagnostics = {}
        for step in range(1, max_steps + 1):
            # 直接运行 teacher，而不是 algorithm.act/process_env_step：不进入 PPO storage。
            actions = locomotion_actor(observations.to(env.device))
            _, _, dones, _ = env.step(actions.to(env.device))
            observations = env.get_locomotion_observations()
            stable, last_diagnostics = env.get_locomotion_stability(warmup_cfg)
            # 每个环境独立计数；一拍失稳或 reset 就中断其连续稳定记录。
            consecutive_stable = torch.where(
                stable & ~dones,
                consecutive_stable + 1,
                torch.zeros_like(consecutive_stable),
            )
            ready = consecutive_stable >= stable_steps
            last_ready_fraction = float(ready.float().mean().item())
            if step >= min_steps and last_ready_fraction >= required_fraction:
                return WarmupStats(steps=step, stable_fraction=last_ready_fraction)

        maxima = {
            name: float(value.max().item()) for name, value in last_diagnostics.items()
        }
        raise RuntimeError(
            "locomotion warmup did not reach the required stable state; "
            f"ready_fraction={last_ready_fraction:.4f}, required={required_fraction:.4f}, "
            f"max_errors={maxima}. No jump PPO data was collected."
        )
    finally:
        # 成功返回和失败抛异常都恢复 jump 的原始 episode 时限。
        env.max_episode_length = original_max_episode_length


# ============ PPO 数据过滤：保留终止拍，排除终止后的补齐步 ============


@contextmanager
def valid_jump_batches(algorithm, valid_steps):
    """临时过滤 storage 的 minibatch；退出 with 时恢复原生成器。

    valid_steps 的形状为 [rollout 时间步, 并行环境数]。
    保留原 storage 布局，只改变优势归一化和训练时的抽样范围。
    """
    if bool(valid_steps.all()):
        yield
        return
    storage = algorithm.storage
    # 与 storage 一样按 [时间, 环境] 展平，得到有效 transition 的一维索引。
    indices = valid_steps.flatten().nonzero(as_tuple=False).flatten().to(storage.device)
    # compute_returns 在 inference_mode 中创建 advantages，须在相同模式下改写。
    with torch.inference_mode():
        advantages = storage.returns - storage.values
        selected = advantages.flatten(0, 1)[indices]
        if not algorithm.normalize_advantage_per_mini_batch:
            selected = (selected - selected.mean()) / (selected.std(unbiased=False) + 1e-8)
        storage.advantages.zero_()
        storage.advantages.flatten(0, 1)[indices] = selected

    def generator(num_mini_batches, num_epochs=8):
        fields = {
            name: getattr(storage, source).flatten(0, 1)
            for name, source in (
                ("observations", "observations"),
                ("actions", "actions"),
                ("values", "values"),
                ("returns", "returns"),
                ("advantages", "advantages"),
                ("old_actions_log_prob", "actions_log_prob"),
            )
        }
        params = tuple(p.flatten(0, 1) for p in storage.distribution_params)
        for _ in range(num_epochs):
            shuffled = indices[torch.randperm(indices.numel(), device=indices.device)]
            # 极少有效样本时允许重复抽样，避免空 minibatch。
            if shuffled.numel() < num_mini_batches:
                shuffled = shuffled.repeat((num_mini_batches + shuffled.numel() - 1) // shuffled.numel())
            for batch_idx in torch.tensor_split(shuffled, num_mini_batches):
                yield storage.Batch(
                    **{name: value[batch_idx] for name, value in fields.items()},
                    old_distribution_params=tuple(p[batch_idx] for p in params),
                )

    # 即使 PPO update 抛异常，也必须恢复生成器，不能污染下一轮采集。
    original = storage.mini_batch_generator
    storage.mini_batch_generator = generator
    try:
        yield
    finally:
        storage.mini_batch_generator = original


def update_jump_transitions(algorithm, valid):
    """将交接前的有效 transition 打包到临时 storage，再执行 PPO 更新。"""
    original = algorithm.storage
    packed = copy(original)
    valid = valid.to(original.device)
    count = int(valid.sum().item())
    if count < algorithm.num_mini_batches:
        raise ValueError("too few jump transitions for PPO mini-batches")
    # compute_returns 已在完整时间序列上执行。这里仅打包训练样本，不重新计算 GAE。
    # [时间, 环境, ...] -> [有效样本, ...] -> [1, 有效样本, ...]。
    packed.num_transitions_per_env, packed.num_envs = 1, count
    packed.observations = original.observations[valid].unsqueeze(0)
    for name in ("actions", "values", "returns", "actions_log_prob", "rewards", "dones"):
        setattr(packed, name, getattr(original, name)[valid].unsqueeze(0))
    packed.distribution_params = tuple(p[valid].unsqueeze(0) for p in original.distribution_params)
    packed.advantages = packed.returns - packed.values
    if not algorithm.normalize_advantage_per_mini_batch:
        packed.advantages = (packed.advantages - packed.advantages.mean()) / (
            packed.advantages.std(unbiased=count > 1) + 1e-8
        )
    # 浅拷贝保留 storage 的接口；原始 storage 留给下一轮固定长度采集。
    algorithm.storage = packed
    try:
        return algorithm.update()
    finally:
        algorithm.storage = original
        original.clear()


# ============ 训练入口：每轮预热、采集、更新与保存 ============


def learn_staged_jump(
    runner,
    locomotion_actor,
    num_learning_iterations: int,
    warmup_cfg: Mapping,
) -> None:
    """执行预热、jump 采集、有效数据 PPO 更新、日志与 checkpoint 保存。"""
    from rsl_rl.utils import check_nan

    # 1. 检查采集契约：一个 rollout 必须恰好覆盖一个 jump 周期。
    env = runner.env
    handoff = getattr(env, "env_cfg", {}).get("handoff_on_landing", False)
    if handoff and (runner.alg.actor.is_recurrent or runner.alg.critic.is_recurrent):
        raise ValueError("landing handoff currently requires feedforward actor and critic")
    expected_horizon = int(round(env.phase_cycle_s / env.dt))
    rollout_horizon = int(runner.cfg["num_steps_per_env"])
    if rollout_horizon != expected_horizon:
        raise ValueError(
            f"jump PPO horizon must equal one phase cycle: got {rollout_horizon}, expected {expected_horizon}"
        )

    # 2. 准备模型、分布式参数和日志 writer。
    runner.alg.train_mode()
    if any(getattr(getattr(runner.alg, name, None), "is_recurrent", False) for name in ("actor", "critic")):
        raise ValueError("jump termination masking requires feedforward actor and critic")
    locomotion_actor.eval()
    if runner.is_distributed:
        print(f"Synchronizing parameters for rank {runner.gpu_global_rank}...")
        runner.alg.broadcast_parameters()
    runner.logger.init_logging_writer()

    start_it = runner.current_learning_iteration
    total_it = start_it + num_learning_iterations
    for it in range(start_it, total_it):
        start = time.time()
        with torch.inference_mode():
            # 3. teacher 预热后直接切换 jump，不再次 reset 机器人物理状态。
            warmup_stats = stabilize_with_locomotion(env, locomotion_actor, warmup_cfg)
            observations = env.begin_jump_rollout().to(runner.device)
            active = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
            # active：这一拍开始前仍在采集 jump 的环境。
            # valid_steps：整轮哪些 transition 可参与 PPO，形状为 [时间, 环境]。
            valid_steps = torch.zeros(
                (rollout_horizon, env.num_envs), dtype=torch.bool, device=env.device
            )

            # 4. 固定长度采集；各环境可以提前结束，但仍补齐 storage 的时间维度。
            for rollout_step in range(rollout_horizon):
                # 在 step 前记录，因此导致死亡/落地的动作仍是有效训练样本。
                valid_steps[rollout_step] = active
                actions = runner.alg.act(observations)
                if handoff and torch.any(~active):
                    # 已结束环境使用 teacher 动作；未结束环境继续使用 jump 动作。
                    teacher_actions = locomotion_actor(env.get_locomotion_observations().to(runner.device))
                    actions = torch.where(active.to(actions.device).unsqueeze(-1), actions, teacher_actions)
                observations, rewards, dones, extras = env.step(actions.to(env.device))
                if runner.cfg.get("check_for_nan", True):
                    check_nan(observations, rewards, dones)
                # 允许 rebound 导致提前死亡，其他意外终止中止本轮 PPO 更新。
                rebound_death = extras.get("jump_rebound_termination", torch.zeros_like(dones))
                unexpected = dones & active & ~rebound_death
                if torch.any(unexpected):
                    failed = int(torch.count_nonzero(unexpected).item())
                    raise RuntimeError(
                        f"{failed} environments terminated before the end of the jump horizon "
                        f"at rollout step {rollout_step + 1}/{rollout_horizon}; PPO update aborted"
                    )

                if handoff:
                    # 交接只切换策略/电机参数，不 reset 物理状态；落地拍视为 jump 终点。
                    landed = env.handoff_landed_environments() & active
                    dones = dones | landed
                    observations = env.get_observations()
                # 已结束环境的补齐步奖励清零并保持 done，防止 GAE 跨越技能终点。
                rewards = torch.where(active, rewards, 0.0)
                dones = dones | ~active
                active = active & ~dones

                is_final_step = rollout_step == rollout_horizon - 1
                if is_final_step:
                    # 周期结束是技能自然终点，不是 timeout，不额外 bootstrap。
                    dones = torch.ones_like(dones)
                    extras = env.finish_jump_rollout(warmup_steps=warmup_stats.steps)

                observations = observations.to(runner.device)
                rewards = rewards.to(runner.device)
                dones = dones.to(runner.device)
                runner.alg.process_env_step(observations, rewards, dones, extras)
                intrinsic_rewards = runner.alg.intrinsic_rewards if runner.alg.rnd else None
                # 补齐步虽然持续 done，但日志只结算仍有效的结束拍，避免重复计数。
                log_dones = dones & valid_steps[rollout_step]
                runner.logger.process_env_step(rewards, log_dones, extras, intrinsic_rewards)

            stop = time.time()
            collect_time = stop - start
            start = stop
            # 先在完整时间序列上计算回报，再过滤 minibatch；不能先打包再计算 GAE。
            runner.alg.compute_returns(observations)

        # 5. 只用有效 jump transition 更新策略；预热和终点后的补齐步不参与学习。
        if handoff:
            loss_dict = update_jump_transitions(runner.alg, valid_steps)
        else:
            with valid_jump_batches(runner.alg, valid_steps):
                loss_dict = runner.alg.update()
        stop = time.time()
        learn_time = stop - start
        runner.current_learning_iteration = it
        # 6. 记录本轮统计；collect_time 包含 teacher 预热与 jump 采集。
        runner.logger.log(
            it=it,
            start_it=start_it,
            total_it=total_it,
            collect_time=collect_time,
            learn_time=learn_time,
            loss_dict=loss_dict,
            learning_rate=runner.alg.learning_rate,
            action_std=runner.alg.get_policy().output_std,
            rnd_weight=runner.alg.rnd.weight if runner.alg.rnd else None,
        )
        if runner.logger.writer is not None and it % runner.cfg["save_interval"] == 0:
            runner.save(os.path.join(runner.logger.log_dir, f"model_{it}.pt"))

    # 训练结束额外保存一次最终 checkpoint，再关闭日志 writer。
    if runner.logger.writer is not None:
        runner.save(
            os.path.join(
                runner.logger.log_dir,
                f"model_{runner.current_learning_iteration}.pt",
            )
        )
        runner.logger.stop_logging_writer()
