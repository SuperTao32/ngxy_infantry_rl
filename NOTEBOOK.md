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

# Infantry 单策略多任务

- 配置与结构：`experiments/genesis/wheel_leg_infantry/README.md`。
- 统一训练：`uv run --locked python -m experiments.genesis.wheel_leg_infantry.train -e infantry -B 8192 --max-iterations 10000`。
- 统一评估：`uv run --locked python -m experiments.genesis.wheel_leg_infantry.eval -e infantry --tasks jump -B 64`。
- 续训：`uv run --locked python -m experiments.genesis.wheel_leg_infantry.train -e infantry --resume --max-iterations 20000`。
- 四个任务共用一个环境和一个 GRU + MLP policy；任务配置分别位于 `tasks/locomotion`、`tasks/recovery`、`tasks/jump`、`tasks/multi_terrain`。
