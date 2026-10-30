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

# 选择地形，连续操纵同一台机器人
uv run --locked python -m src.genesis.wheel_leg_infantry.eval \
  -e infantry --terrain plane
```

训练默认开启全部任务，`--tasks` 支持多个名称。每个开启任务需要正权重且至少一个环境；multi_terrain 的内部子任务还需足够的环境配额。

**eval 是连续交互仿真**，默认开启 viewer、控制一台机器人，直到关闭窗口。无需选择 tasks、设置任务配额或添加键盘开关。通过 `--terrain`（也支持 `-terrain`）加载已有地形；`--difficulty` 选择对应地形已有级别，`--terrain-size LENGTH WIDTH` 调整场地尺寸：

```bash
uv run --locked python -m src.genesis.wheel_leg_infantry.eval \
  -e infantry -terrain stairs_up --difficulty 0
```

可用地形：`plane`、`stairs`、`stairs_up`、`platform_ridge_down`、`platform_ridge_up`、`trapezoidal_wave`、`square_wave`、`slope_up`、`slope_down`。

点击 viewer 窗口后操纵：

| 按键 | 命令 |
| --- | --- |
| ↑ / ↓ | 前向速度增减 0.5 m/s |
| ← / → | 偏航速度增减 0.5 rad/s |
| PageUp / PageDown | 目标高度增减 0.05 m |
| J | 在当前位置启动一次跳跃命令 |
| 1 / 2 / 3 | 选择平地 / 20 cm / 40 cm 跳跃编码，随后按 J |
| R | 随机选择一个已配置的姿态复位并起身，清空 GRU 和速度目标，站稳后恢复普通控制 |
| L | 取消当前行为，返回普通 command 控制 |
| 空格 | 取消当前行为并清零速度、转向 |
| Backspace | 显式复位机器人到场地出生位置，并清空 GRU |
| Enter | 相机对准机器人 |

跳跃沿用保存配置的模式 one-hot、六维时间编码、目标 base 到轮底距离轨迹和电机参数，按键只启动控制信号，不改机器人位置或实际速度。跳跃计时结束后恢复普通命令；起身期间保持零速度与起身高度，方向/高度按键暂时锁定。跳跃及起身完成后继续运行，保留连续的策略历史。倒地、接触、动作完成、回合时长和地形边界均不会自动复位；R 和 Backspace 会显式复位机器人、传感器、动作与 GRU，数值求解器报错则停止仿真。

R 的姿态池由 `tasks/recovery/config.py::get_recovery_reset_cfgs()` 提供，默认通过已有的 `get_reset_pose_cfg()` 解析 `reset_pos.json`。新增姿态时将快照路径追加到 `RECOVERY_RESET_POSES` 即可，也可使用 `"default"` 站立姿态。eval 每次按 R 等概率随机选择一个配置（可能重复，受 `--seed` 控制），共用完整环境 reset 路径并适配当前地形高度；Backspace 仍使用正常站立出生配置。仿真不直接读取姿态文件；也可向 `InteractiveEnv` 注入其他来源的数值姿态配置列表。此姿态池不改变训练使用的 `RESET_POSE`。

命令范围取 checkpoint 对应的 locomotion 课程；速度区间包含零，支持从静止逐步调速。优先使用 checkpoint 保存的跳跃/起身参数，缺失时使用当前默认配置；动作效果取决于模型实际训练能力，尤其是当前姿态下按键触发跳跃与训练时直接从起跳状态开始的差异。

`--steps N` 可选地限制仿真步数；`--no-vis --steps N` 用于无窗口运行，`--output results.json` 保存运行步数、模拟时长与地形配置。训练和评估共用 `--log-root`、`--seed`、`--cpu` 及随机化开关。

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
| `train.py` / `eval.py` | 多任务训练/恢复与连续交互仿真 |
| `tools/interactive_env.py` / `interactive_viewer.py` | 部署命令、固定场景、连续仿真与键盘诊断 |
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
