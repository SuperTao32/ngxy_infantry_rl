"""完整快照复位、动静态恢复及起身状态；仿真与动作执行由统一环境负责。"""

from copy import deepcopy

from ..base import Task
from .rewards import RecoveryRewards
from .buffers import RecoveryBuffers
from .reset_sampling import RecoveryResetSampler
from .reset_pose import resolve_reset_selection
from ...utils.reset import ensure_reset_clearance


class RecoveryTask(RecoveryRewards, Task):
    EVENT_REWARDS = Task.EVENT_REWARDS | {"stand_up_speed"}
    EPISODE_METRICS = ("height_gate", "attitude_gate", "base_contact")

    def _register_task_curriculum(self):
        self.curriculum.register_target("recovery_reset", self._apply_recovery_reset)

    def _apply_recovery_reset(self, values):
        unknown = set(values) - {"difficulty_range", "dynamic_probability"}
        if unknown:
            raise ValueError(f"Unknown recovery_reset curriculum options: {sorted(unknown)}")
        config = self.env_cfg.get("recovery_reset")
        if config is None:
            raise ValueError("recovery_reset curriculum requires a reset pose pool")
        updated = {**config, **deepcopy(values)}
        sampler = self.__dict__.get("reset_sampler")
        if sampler is None:
            # 课程的首阶段早于 bind_buffers，先校验并存储筛选配置。
            resolve_reset_selection(updated)
        else:
            sampler.apply_selection(updated)
        self.env_cfg["recovery_reset"] = updated

    def _initialize_task_buffers(self):
        self.stand_up = RecoveryBuffers(self.env_cfg["stand_up"], self.num_envs, self.dt, self.device)
        self.stand_up.active &= self.mask
        reset_cfg = self.env_cfg.get("recovery_reset")
        self.reset_sampler = (
            RecoveryResetSampler(reset_cfg, self.robot, self.num_envs, self.device, self.env_cfg["default_joint_pos"])
            if reset_cfg is not None
            else None
        )
        if self.reset_sampler is not None:
            sampler = self.reset_sampler
            print(f"[recovery] poses={len(sampler.available_pose_ids)}/{len(sampler.pose_pool)}, "
                  f"difficulty_range={reset_cfg.get('difficulty_range')}, dynamic_probability={sampler.dynamic_probability:g}")

    def _sample_base_reset_state(self, env_ids):
        if self.reset_sampler is None:
            state = super()._sample_base_reset_state(env_ids)
            ensure_reset_clearance(self.robot, env_ids, state)
            return state
        state = self.reset_sampler.sample(env_ids, self.terrain)
        self.reset_sampler.ensure_clearance(self.robot, env_ids, state)
        return state

    def _reset_task_buffers(self, mask):
        self.stand_up.reset(mask)

    def _on_env_ids_changed(self):
        if self.stand_up is not None:
            self.stand_up.active &= self.mask
            self.stand_up.was_active &= self.mask
            self.stand_up.just_completed &= self.mask
        if self.__dict__.get("reset_sampler") is not None:
            self.reset_sampler.is_dynamic[~self.mask] = False
            self.reset_sampler.pose_ids[~self.mask] = -1
            self.reset_sampler.landed[~self.mask] = True

    def _resample_commands(self, mask):
        super()._resample_commands(mask)
        self.stand_up.enforce_commands(self.commands)

    def _update_task_state(self):
        if self.reset_sampler is not None:
            self.reset_sampler.update_landing(self.wheel_contact, self.base_contact)
        self.stand_up.update(self)
        self.stand_up.enforce_commands(self.commands)

    def _task_success(self):
        return self.stand_up.just_completed

    def episode_metrics(self, mask):
        success = ~self.stand_up.active
        # 同一拍站稳与物理失败并发时，统计与环境的失败优先终止规则一致。
        terminated = getattr(self.env, "terminated_buf", None)
        if terminated is not None:
            success = success & ~terminated
        success = success.float()
        duration = self.stand_up.elapsed_steps * self.dt
        metrics = {"success": success[mask], "recovery_duration_s": duration[mask]}
        if self.__dict__.get("reset_sampler") is not None:
            metrics.update(self.reset_sampler.episode_metrics(mask, success, duration))
        return metrics
