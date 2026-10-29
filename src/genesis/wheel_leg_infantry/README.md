# Wheel Leg Infantry：单策略多任务框架

四个任务在一个 `WheelLegInfantryEnv`、一个 Genesis Scene 和一份 PPO rollout 中训练同一 actor/critic。默认按环境数量等分 `locomotion`、`recovery`、`jump`、`multi_terrain`；各任务的回合可以独立结束，GRU 按各环境的 done 重置。

## 入口

从仓库根目录执行：

```bash
# 默认混合四个任务，训练一个 policy
uv run --locked python -m src.genesis.wheel_leg_infantry.train \
  -e infantry -B 8192 --max-iterations 10000

# 调整混合比例；任务内部的配置仍在各自目录
uv run --locked python -m src.genesis.wheel_leg_infantry.train \
  -e infantry --task-weights locomotion=4 recovery=1 jump=1 multi_terrain=2

# 使用保存配置、优化器和任务课程续训；总目标轮数语义
uv run --locked python -m src.genesis.wheel_leg_infantry.train \
  -e infantry --resume --max-iterations 20000

# 同一 checkpoint 评估全部任务，或用 --tasks 聚焦其中一个任务
uv run --locked python -m src.genesis.wheel_leg_infantry.eval \
  -e infantry --tasks jump -B 64 --steps 3000
```

训练和评估共用 `--tasks`、`--log-root`、`--seed`、`--cpu` 及随机化开关。`--tasks` 支持多个名称；训练默认全部开启。每个开启任务需要正权重且至少一个环境；multi_terrain 的内部子任务还需足够的环境配额。默认评估开启 viewer，`--no-vis` 关闭；`--output results.json` 保存汇总。

多卡入口仍是同一个模块，每卡环境数由 `-B` 指定：

```bash
uv run --locked torchrun --standalone --nproc_per_node=2 \
  --module src.genesis.wheel_leg_infantry.train -e infantry -B 4096
```

每次训练创建 `logs/<experiment>/version_NNNN`，保存配置与 `model_N.pt`。checkpoint 的 `infos.infantry_tasks` 保存多地形难度和停车课程状态；完整续训使用相同的任务集合。此框架的观测和网络已变化，旧任务的 MLP/teacher checkpoint 需要另外迁移。

## 代码分工

| 路径 | 职责 |
| --- | --- |
| `config.py` | 统一机器人、观测、GRU/PPO 默认值、任务权重与课程组装；保存时仅存任务环境覆盖项 |
| `env.py` | 统一场景、控制、传感器、任务调度与重置；`_update_observations()` 显式列出全部观测 |
| `train.py` / `eval.py` | 统一训练、恢复与评估 |
| `policy_runner.py` | 一份 PPO actor/critic，保存和恢复任务课程 |
| `tasks/task_config.py` | 共享命令采样、reset 状态采样与配置更新 |
| `tasks/base.py` / `manager.py` | 任务协议、固定 batch 分组、任务注册和 hook 调度 |
| `tasks/<name>/config.py` | 各任务独立声明的命令、奖励参数、重置、终止和课程覆盖项 |
| `tasks/<name>/task.py` | 任务行为；不创建 Scene、runner 或其他策略 |
| `tasks/<name>/buffers.py` / `rewards.py` | 专用状态和奖励；无额外状态的 locomotion 直接复用公共实现 |
| `terrains/training.py` | 在同一 Scene 中隔离各任务区域，转换局部/世界坐标 |
| `terrains/` | 共享地形几何、参数、布局和高度查询 |
| `tasks/jump/terrain.py` / `tasks/multi_terrain/terrain.py` | 各任务的通道和地形分配 |
| `sensors/tof.py` / `sensors/base_contact.py` | 训练运行时的 ToF 测距与机身分区接触统计 |
| `utils/` | 共用的运动学、张量采样和门控函数 |
| `tools/` | 分布式启动、运行记录、模型展示、viewer 和制动诊断辅助 |

各任务 `config.py::get_cfgs()` 返回 `env_cfg`、`reward_cfg`、`command_cfg`、`curriculum_cfg` 四个具名覆盖项，不读取其他任务配置。机器人、动作、观测、公共奖励形状和 PPO 默认值统一在根目录 `config.py` 声明；`TASK_WEIGHTS` 控制四任务比例。任务构造时合并共享默认值，任务中的共享机器人/动作覆盖会在配置组装时报错。

全局随机化由根目录 `get_curriculum_cfg()` 控制，保留原有 0、4500、7000 轮的强度变化；各任务课程仅管理自身参数。

任务组件读取共享物理状态，写公共 buffer 时使用自己的 `mask`。奖励只累加本任务的行，所有行随后进入同一 RSL-RL rollout 和 PPO 更新。任务内部可以保留独立课程，但不各自建立训练循环。

`env.py::_reset_idx()` 将 `None` 转换成全量 mask，后续全量和局部 reset 共用机器人、物理状态、传感器、任务 buffer 和命令的重置路径。`None` 用于初始化或手动 reset，不生成结束回合日志；显式布尔 mask 会先保存结束回合统计，再清零对应行。

## 网络与观测

网络参照 `experiments/beni_rl/rl_cfg.py`：actor 与 critic 都使用观测归一化 → GRU（512 hidden、1 层）→ MLP（512/256/128、ELU）。actor 使用 GaussianDistribution，初始标准差 1.0；PPO 默认学习率 1e-3、rollout 长度 24。

所有任务共用六维动作：四个腿关节位置目标和两个轮速目标，共享缩放、裁剪和一拍动作延迟。控制步为 0.01 s，物理子步维持原实现。

actor 包含 IMU、重力方向、关节/轮测量、6D actions、六维命令、ToF 与六维跳跃阶段编码。命令依次为缩放后的前向速度、偏航速度、高度目标，以及未缩放的三维 jump 模式 one-hot；非 jump 环境的模式为零。critic 再追加速度、接触等仿真真值，以及 baselink 原点的世界坐标 z（不扣除地形高度）。默认 ToF 配置下 actor 为 51D、critic 为 75D；选择任务子集不会改变网络输入维度。

jump 直接从配置的起跳位置和前向初速度开始回合，沿用阶段状态、轨迹和落地奖励，正常参与每个 PPO rollout。其起跳初速度、恢复姿态、各任务比例与奖励目前保留为框架初值，后续在各任务配置中调整。
