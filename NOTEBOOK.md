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

# Infantry jump 统一任务

- 唯一配置：`experiments/genesis/wheel_leg_infantry/tasks/jump/config.py`。
- 三维 one-hot：平地 `[1,0,0]`、20 cm 台阶 `[0,1,0]`、40 cm 台阶 `[0,0,1]`。
- `env_cfg["jump_modes"]` 配置模式比例、轮底峰值目标、参考轨迹和速度—起跳距离表；env 在 teacher 稳速后设置起跳位置，无需 ToF。
- 配置检查：`uv run --locked python -m experiments.genesis.wheel_leg_infantry.tasks.jump.train --dry-run`。
- 重新训练：`uv run --locked python -m experiments.genesis.wheel_leg_infantry.tasks.jump.train -e infantry_jump --locomotion-log-root log_shared -B 1024 --max-iterations 2001`。
- 自动评估：`uv run --locked python -m experiments.genesis.wheel_leg_infantry.tasks.jump.eval -e infantry_jump --mode all --speed 1.0 -B 3`。

## 当前任务续训

- `uv run --locked python -m experiments.genesis.wheel_leg_infantry.tasks.jump.train -e infantry_jump --resume --max-iterations 4001`。
- 指定来源：追加 `--resume 3 --checkpoint 1500`（替换上面的 `--resume`）。
- 默认 `--resume-config saved` 使用来源版本保存的配置；要使用当前唯一配置，追加 `--resume-config current`。
- 恢复模型、优化器、迭代号和课程进度，每次写入新的 `version_NNNN`；`--max-iterations` 是总目标轮数。
- 续训使用该 run 记录的 locomotion checkpoint 做每轮预热；日志迁移后可用 `--locomotion-log-root` 指定其新根目录。

## 跳跃成功与失败

- 姿态误差使用 `2 * (1 + projected_gravity_z)`；当前 `jump_max_tilt_deg=20`。超限、机身接触、撞立面或台阶下方落地均使该次跳跃失败。
- 首次落地后再次双轮离地判定为反弹失败。终止后的补齐步不参与 PPO 或回合统计。
- 台阶成功要求真实起跳、双轮在台面内部接触并连续落稳；平地还要求达到轮底峰值目标。
- 分别观察 `Episode/success_flat`、`Episode/success_step_20cm`、`Episode/success_step_40cm`。
- 完整使用说明见 `experiments/genesis/wheel_leg_infantry/tasks/jump/README.md`。
