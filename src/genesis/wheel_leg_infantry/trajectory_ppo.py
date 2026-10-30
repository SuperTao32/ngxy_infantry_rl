"""只用可训练控制步更新 PPO，保留异步环境的连续跳跃轨迹。"""

import torch
from tensordict import TensorDict

from rsl_rl.algorithms import PPO
from rsl_rl.models import RNNModel
from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.storage import RolloutStorage


class TrajectoryRNNModel(RNNModel):
    def get_latent(self, obs, masks=None, hidden_state=None):
        if masks is None:
            return super().get_latent(obs, masks, hidden_state)
        latent = MLPModel.get_latent(self, obs)
        latent, _ = self.rnn.rnn(latent, hidden_state)
        # 不等长轨迹不能使用框架按固定环境步数 reshape 的 unpad。
        # 保留 [1, 有效样本数, hidden]，与 PPO 的 [time, batch] 数据布局一致。
        return latent.transpose(0, 1)[masks.T].unsqueeze(0)


class TrajectoryStorage(RolloutStorage):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_mask = torch.zeros(self.num_transitions_per_env, self.num_envs, dtype=torch.bool, device=self.device)

    def add_transition(self, transition):
        self.train_mask[self.step].copy_(transition.train_mask)
        super().add_transition(transition)

    def trajectory_bounds(self):
        valid = self.train_mask
        starts, ends = valid.clone(), valid.clone()
        starts[1:] &= ~valid[:-1] | self.dones[:-1, :, 0].bool()
        ends[:-1] &= ~valid[1:] | self.dones[:-1, :, 0].bool()
        # 环境优先排序，保证起点和终点逐条配对。
        return starts.T.nonzero(), ends.T.nonzero()

    def _batch(self, times, envs, observations, *, hidden_states=(None, None), masks=None):
        def take(value):
            return value[times, envs].unsqueeze(0)

        return self.Batch(
            observations=observations,
            actions=take(self.actions), values=take(self.values),
            advantages=take(self.advantages), returns=take(self.returns),
            old_actions_log_prob=take(self.actions_log_prob),
            old_distribution_params=tuple(take(value) for value in self.distribution_params),
            hidden_states=hidden_states, masks=masks,
        )

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        times, envs = self.train_mask.nonzero(as_tuple=True)
        for _ in range(num_epochs):
            for indices in torch.randperm(len(times), device=self.device).tensor_split(num_mini_batches):
                t, e = times[indices], envs[indices]
                yield self._batch(t, e, self.observations[t, e].unsqueeze(0))

    def recurrent_mini_batch_generator(self, num_mini_batches, num_epochs=8):
        starts, ends = self.trajectory_bounds()
        lengths = ends[:, 1] - starts[:, 1] + 1
        for _ in range(num_epochs):
            for indices in torch.randperm(len(starts), device=self.device).tensor_split(num_mini_batches):
                first = starts[indices]
                length = lengths[indices]
                time = torch.arange(int(length.max()), device=self.device)[:, None]
                masks = time < length[None, :]
                times = (time + first[None, :, 1]).clamp_max(self.num_transitions_per_env - 1)
                envs = first[None, :, 0].expand_as(times)
                observations = TensorDict(
                    {key: value[times, envs].masked_fill(~masks[..., None], 0.)
                     for key, value in self.observations.items()},
                    batch_size=masks.shape, device=self.device,
                )

                def hidden(saved):
                    if saved is None:
                        return None
                    states = [value[first[:, 1], :, first[:, 0], :].transpose(0, 1).contiguous() for value in saved]
                    return states[0] if len(states) == 1 else tuple(states)

                # 目标与模型的输出均按轨迹优先顺序展平，不含助跑或 padding。
                yield self._batch(
                    times.T[masks.T], envs.T[masks.T], observations,
                    hidden_states=(hidden(self.saved_hidden_state_a), hidden(self.saved_hidden_state_c)), masks=masks,
                )


class TrajectoryPPO(PPO):
    def __init__(self, actor, critic, storage, **kwargs):
        storage = TrajectoryStorage(
            storage.training_type, storage.num_envs, storage.num_transitions_per_env,
            storage.observations[0], storage.actions_shape, storage.device,
        )
        super().__init__(actor, critic, storage, **kwargs)

    def process_env_step(self, obs, rewards, dones, extras):
        # 此 mask 在 env.step 开始时快照，对应产生动作的旧观测，不能用返回的新信号判定。
        valid = extras.get("ppo_train_mask", torch.ones_like(dones, dtype=torch.bool)).to(self.device).bool()
        self.transition.train_mask = valid
        if valid.any():
            train_obs = self.transition.observations[valid]
            self.actor.update_normalization(train_obs)
            self.critic.update_normalization(train_obs)
            if self.rnd:
                self.rnd.update_normalization(train_obs)

        self.transition.rewards = torch.where(valid, rewards, 0.).clone()
        self.transition.dones = dones
        if self.rnd:
            self.intrinsic_rewards = torch.zeros_like(rewards)
            if valid.any():
                self.intrinsic_rewards[valid] = self.rnd.get_intrinsic_reward(self.transition.observations[valid])
            self.transition.rewards += self.intrinsic_rewards
        if "time_outs" in extras:
            self.transition.rewards += self.gamma * self.transition.values.squeeze(-1) * extras["time_outs"].to(self.device) * valid
        self.storage.add_transition(self.transition)
        self.transition.clear()
        # 助跑到 jump 的信号更新不制造 done；起始隐状态保留，助跑不参与反向传播。
        self.actor.reset(dones)
        self.critic.reset(dones)

    def compute_returns(self, obs):
        st = self.storage
        # GAE 不能穿过被排除的区间，即使调用方没有为信号边界发出 done。
        st.dones[:-1, :, 0] |= (~st.train_mask[1:]).to(st.dones.dtype)
        st.dones[:, :, 0] |= (~st.train_mask).to(st.dones.dtype)
        per_batch = self.normalize_advantage_per_mini_batch
        self.normalize_advantage_per_mini_batch = True
        try:
            super().compute_returns(obs)
        finally:
            self.normalize_advantage_per_mini_batch = per_batch
        valid_advantages = st.advantages[st.train_mask]
        if not per_batch and valid_advantages.numel():
            std = valid_advantages.std() if valid_advantages.numel() > 1 else 0.
            st.advantages[st.train_mask] = (valid_advantages - valid_advantages.mean()) / (std + 1e-8)
        st.advantages[~st.train_mask] = 0.
        st.returns[~st.train_mask] = 0.

    def update(self):
        recurrent = self.actor.is_recurrent or self.critic.is_recurrent
        count = len(self.storage.trajectory_bounds()[0]) if recurrent else int(self.storage.train_mask.sum())
        # 各 rank 使用相同批次数；某个 rank 没有有效样本时共同跳过，避免分布式死锁。
        if self.is_multi_gpu:
            available = torch.tensor(count, device=self.device)
            torch.distributed.all_reduce(available, op=torch.distributed.ReduceOp.MIN)
            count = int(available)
        if not count:
            self.storage.clear()
            return {"value": 0., "surrogate": 0., "entropy": 0.}
        configured_batches = self.num_mini_batches
        self.num_mini_batches = min(configured_batches, count)
        try:
            return super().update()
        finally:
            self.num_mini_batches = configured_batches
