# uv相关命令
- uv sync --locked
- uv run --locked python <脚本>
- uv add <包>
- uv remove <包>
- uv lock --upgrade-package <包>
- UV_HTTP_TIMEOUT=600 
- UV_HTTP_RETRIES=10 
- UV_CONCURRENT_DOWNLOADS=2 

# Tensorboard
- uv run --locked tensorboard --logdir logs

# Infantry jump 续训

- 从最新有效 jump 版本的最新 checkpoint 继续：
  `uv run --locked python -m experiments.genesis.wheel_leg_infantry.tasks.jump.train -e infantry_jump_v4 --resume --max-iterations 4001`
- 指定来源：追加 `--resume 3 --checkpoint 1500`（替换上面的 `--resume`），从第 1501 轮训练到第 4000 轮。
- 默认 `--resume-config saved` 使用来源版本保存的配置；要使用当前 jump 配置，追加 `--resume-config current --config 25`（也可选 45，模型结构须兼容）。
- 恢复模型、优化器、迭代号和课程进度，每次写入新的 `version_NNNN`；`--max-iterations` 是总目标轮数。
- 续训仍使用来源 jump 记录的 locomotion checkpoint 做每轮预热，需要保留对应的 `cfgs.pkl` 和模型。日志迁移后可用 `--locomotion-log-root` 指定其新根目录；续训时不会按 `--locomotion-exp-name/version/ckpt` 更换 teacher。
- `--dry-run` 仅检查当前 `--config` 的配置契约，不检查续训 checkpoint。


# 单次跳跃防翻转与反弹

- Jump 姿态误差改为 `2 * (1 + projected_gravity_z)`，正立为 0，倒立为 4；平衡、空中姿态和落地姿态奖励都区分正反。
- 两套配置的 `jump_max_tilt_deg=45`：任一时刻超出倾角或机身接触，本回合高度失效，停止腾空收益，并持续施加 `jump_invalid=-200` 惩罚。已发放的历史奖励不会追溯撤回。
- 首次任意轮或机身接触即结算本次跳跃；后续再腾空不累计高度和腾空奖励。首次落地后双轮离地超过 2 cm 时施加 `landing_airborne=-100` 的持续惩罚（按 dt 积分）。
- 观察 `Episode/jump_invalid`、`Episode/rebound_seen` 和 `Episode/reward_landing_airborne`。`peak_wheel_clearance_m` 为有效跳跃成绩，失稳回合归零，不再直接等同于视觉上的最高轮高。
- 从旧 07/08 续训时使用 `--resume-config current` 和对应 `--config 25` / `--config 45`，才能启用新增惩罚；运行中的进程不会热加载修改。
