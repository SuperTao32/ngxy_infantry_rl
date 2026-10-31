# Wheel Leg Infantry：单策略多任务框架

四个任务在一个 `WheelLegInfantryEnv`、一个 Genesis Scene 和一份 PPO rollout 中训练同一 actor/critic。默认权重为 `locomotion:recovery:jump:multi_terrain = 2:0:1:0`；各任务的回合可以独立结束，GRU 按各环境的 done 重置。

## 入口

从仓库根目录执行：

```bash
# 默认按 2:1 混合平地行走与跳跃任务，训练一个 policy
uv run --locked python -m src.genesis.wheel_leg_infantry.train \
  -e infantry -B 8192 --max-iterations 10000

# 调整混合比例；任务内部的配置仍在各自目录
uv run --locked python -m src.genesis.wheel_leg_infantry.train \
  -e infantry --task-weights locomotion=4 recovery=1 jump=1 multi_terrain=2

# 使用保存配置、优化器和任务课程续训；总目标轮数语义
uv run --locked python -m src.genesis.wheel_leg_infantry.train \
  -e infantry --resume --max-iterations 20000

# 只加载权重，按当前配置开启新实验；优化器和课程从头开始
uv run --locked python -m src.genesis.wheel_leg_infantry.train \
  -e infantry_finetune --load-weight logs/infantry/version_0000/model_3000.pt \
  --tasks locomotion jump --task-weights locomotion=2 jump=1 --max-iterations 10000

# 使用当前配置续训单任务实验
uv run --locked python -m src.genesis.wheel_leg_infantry.train \
  -e locomotion_only --tasks locomotion --resume --resume-config current

# 沿用 locomotion 策略新增 recovery，按当前全局课程分配任务
uv run --locked python -m src.genesis.wheel_leg_infantry.train \
  -e locomotion_only --tasks locomotion recovery --resume --resume-config current

# 选择地形，连续操纵同一台机器人
uv run --locked python -m src.genesis.wheel_leg_infantry.eval \
  -e infantry --terrain plane
```

训练默认构建全部任务，零权重任务暂不采样，可由后续课程启用。`--tasks` 支持多个名称并限制可分配的任务集合；若只选择初始权重为零的任务，需要同时用 `--task-weights` 设置正权重。每个正权重任务至少分配一个环境，所有权重不能同时为零；multi_terrain 的正配额还需容纳其内部课程。`--task-weights` 覆盖初始比例，后续仍遵循配置中的阶段分配。

`--load-weight CHECKPOINT` 加载 actor/critic 的完整模型状态，包括 GRU 权重、动作分布参数及观测归一化统计；使用当前配置创建新版本，优化器、全局迭代数、任务课程和 GRU 隐状态重新初始化。可自由选择任务集合及权重，网络结构与观测/动作维度须兼容源 checkpoint；不需要源目录中的 `cfgs.pkl`。此模式的 `--max-iterations` 为从第 0 轮开始的训练轮数，不能与 `--resume`、`--checkpoint` 或 `--resume-config current` 同用，来源路径记录到 `run_metadata.json`。

续训默认使用 `--resume-config saved`，保留已保存的任务集合与权重，不接受 `--tasks` / `--task-weights`。使用 `--resume-config current` 时可通过这些参数选择当前配置中的任务和初始权重，并在保留 checkpoint 全部任务的基础上新增任务。已有任务保留课程进度及专属状态，新增任务使用当前配置初始化，独立课程从 0 开始；策略、优化器和全局迭代数继续恢复，网络结构需兼容 checkpoint。全局课程立即应用恢复轮次所属阶段：例如从 `model_3000.pt` 恢复，从 3001 轮继续；若该阶段权重为 locomotion=1、recovery=0.5，环境约按 2:1 分配。新增任务在实际分到环境后才推进独立课程。`--max-iterations` 表示总目标轮数，须大于 checkpoint 的已完成轮数。

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

跳跃沿用保存配置的模式 one-hot、六维时间编码和目标 base 到轮底距离轨迹，电机参数始终来自共享根配置；按键只启动控制信号，不改机器人位置或实际速度。跳跃计时结束后恢复普通命令；起身期间保持零速度与起身高度，方向/高度按键暂时锁定。跳跃及起身完成后继续运行，保留连续的策略历史。倒地、接触、动作完成、回合时长和地形边界均不会自动复位；R 和 Backspace 会显式复位机器人、传感器、动作与 GRU，数值求解器报错则停止仿真。

R 的姿态池由 `tasks/recovery/config.py::get_recovery_reset_cfgs()` 提供，默认通过已有的 `get_reset_pose_cfg()` 解析 `reset_pos.json`。新增姿态时将快照路径追加到 `RECOVERY_RESET_POSES` 即可，也可使用 `"default"` 站立姿态。eval 每次按 R 等概率随机选择一个配置（可能重复，受 `--seed` 控制），共用完整环境 reset 路径并适配当前地形高度；Backspace 仍使用正常站立出生配置。仿真不直接读取姿态文件；也可向 `InteractiveEnv` 注入其他来源的数值姿态配置列表。

recovery 训练使用独立的数值姿态池：默认只加载 `tasks/recovery/config.py::RESET_POSE_DIFFICULTIES` 中列出的快照，不扫描采集目录。课程可按难度筛选，过滤侧躺终态，每个环境每次 reset 独立采样，默认 80% 静态起身、20% 低难度动态释放。`train --recovery-poses PATH ...` 可显式指定采集目录/JSON，`--recovery-dynamic-probability 0` 可单测静态。采样参数、分组成功率与续训行为见 [Recovery 说明](tasks/recovery/README.md)。

命令范围取 checkpoint 对应的 locomotion 课程；速度区间包含零，支持从静止逐步调速。优先使用 checkpoint 保存的跳跃/起身参数，缺失时使用当前默认配置；动作效果取决于模型实际训练能力。jump 训练先在空旷区域实际加速，达速并站稳后移动至指定起跳位置，再输入跳跃信息；交互中的 J 则在当前位置直接触发。

`--steps N` 可选地限制仿真步数；`--no-vis --steps N` 用于无窗口运行，`--output results.json` 保存运行步数、模拟时长与地形配置。训练和评估共用 `--log-root`、`--seed`、`--cpu` 及随机化开关。

多卡入口仍是同一个模块，每卡环境数由 `-B` 指定：

```bash
uv run --locked torchrun --standalone --nproc_per_node=2 \
  --module src.genesis.wheel_leg_infantry.train -e infantry -B 4096
```

每次训练创建 `logs/<experiment>/version_NNNN`，保存配置与 `model_N.pt`。checkpoint 的 `infos.infantry_tasks` 保存每个任务的采样进度、多地形难度和停车课程状态；完整续训使用相同的任务集合，可改变环境数或 GPU 数，已有进度保持不变。旧统一框架 checkpoint 缺少采样计数时，以原全局课程轮次初始化并打印提示。此框架的观测和网络已变化，旧任务的 MLP/teacher checkpoint 需要另外迁移。

## 代码分工

| 路径 | 职责 |
| --- | --- |
| `config.py` | 统一机器人、观测、GRU/PPO 默认值、任务权重与课程组装；保存时仅存任务环境覆盖项 |
| `env.py` | 统一场景、控制、传感器、任务调度与重置；`_update_observations()` 显式列出全部观测 |
| `train.py` / `eval.py` | 多任务训练/恢复与连续交互仿真 |
| `tools/interactive_env.py` / `interactive_viewer.py` | 部署命令、固定场景、连续仿真与键盘诊断 |
| `policy_runner.py` / `trajectory_ppo.py` | 一份 PPO actor/critic，过滤助跑样本，保存和恢复任务课程 |
| `tasks/task_config.py` | 共享命令采样、reset 状态采样与配置更新 |
| `tasks/base.py` / `manager.py` | 任务协议、动态环境分配、采样进度和 hook 调度 |
| `tasks/<name>/config.py` | 各任务独立声明的命令、奖励参数、重置、终止和课程覆盖项 |
| `tasks/<name>/task.py` | 任务行为；不创建 Scene、runner 或其他策略 |
| `tasks/<name>/buffers.py` / `rewards.py` | 专用状态和奖励；无额外状态的 locomotion 直接复用公共实现 |
| `terrains/training.py` | 在同一 Scene 中隔离各任务区域，转换局部/世界坐标 |
| `terrains/` | 共享地形几何、参数、布局和高度查询 |
| `tasks/jump/terrain.py` / `tasks/multi_terrain/terrain.py` | 各任务的通道和地形分配 |
| `sensors/tof.py` / `sensors/base_contact.py` | 训练运行时的 ToF 测距与机身分区接触统计 |
| `utils/` | 共用的运动学、张量采样和门控函数 |
| `tools/` | 分布式启动、运行记录、模型展示、viewer 和制动诊断辅助 |

各任务 `config.py::get_cfgs()` 返回 `env_cfg`、`reward_cfg`、`command_cfg`、`curriculum_cfg` 四个具名覆盖项，不读取其他任务配置。机器人、动作、观测、公共奖励形状和 PPO 默认值统一在根目录 `config.py` 声明；`TASK_WEIGHTS` 控制初始比例。任务构造时合并共享默认值，任务中的共享机器人/动作覆盖会在配置组装时报错。

kp、kd、力矩限幅及动作缩放由所有任务共用，任务 reset、课程和交互模式切换均不写入电机参数；只有共享域随机化在物理 reset 时按统一规则调整各环境。

根目录 `get_curriculum_cfg()` 按全局 PPO 轮次控制随机化和四任务的环境权重，保留原有 0、4500、7000 轮的随机化强度变化。在任一阶段的 `targets` 添加 `task_weights` 即可分阶段调配环境，未填写的权重继承上一阶段。例如在第 4500 轮阶段加入：

```python
"task_weights": {"locomotion": 2.0, "recovery": 1.0, "jump": 1.0, "multi_terrain": 2.0},
```

也可向 `stages` 插入独立的权重阶段；`start_iteration` 必须递增。所有阶段配额在场景创建前校验。任务区域预先构建，rollout 结束后只迁移超出新配额的环境；其余回合继续运行。迁移环境通过统一 reset 清空状态，返回 done 以重置 GRU，并切断旧任务到新任务的价值 bootstrap；迁移本身不计作任务成功或失败。

各任务自己的 `start_iteration` 表示累计样本折算出的任务课程轮数：

```text
任务课程轮数 = 累计环境控制步数 / (reference_num_envs × reference_steps_per_iteration)
每个控制步累计 = 该任务当前环境数 × GPU 数
```

环境基准由根配置的 `TASK_CURRICULUM_REFERENCE_ENVS` 声明，当前为每任务 8192 个环境；参考步数默认使用 `train_cfg["num_steps_per_env"]`（当前为 24），无需另外维护步数常量。各任务的 `curriculum_cfg` 仍可显式设置 `reference_num_envs` / `reference_steps_per_iteration`。例如单卡 `-B 8192` 等分两个任务时，每个任务每全局轮推进 0.5 轮课程；分到 8192 个环境时推进一轮，零配额时暂停。首次分到环境前不累计样本，暂停后重新分配会继续已有进度；小数样本进度与折算基准跨 checkpoint 保留，续训不重新解释历史进度。多地形内部的按成功率升级课程仍由真实完成的回合驱动。

任务组件读取共享物理状态，写公共 buffer 时使用自己的 `mask`。奖励只累加本任务的行，训练共用一份 RSL-RL rollout 和 PPO。jump 助跑行通过动作执行前快照的 `ppo_train_mask` 排除；训练批次只包含其他任务及 jump 信号更新后的连续轨迹，助跑不参与奖励、GAE、观测归一化或跳跃课程样本计数。整批无有效样本时跳过优化器更新。任务内部可以保留独立课程，但不各自建立训练循环。

`env.py::_reset_idx()` 将 `None` 转换成全量 mask，后续全量和局部 reset 共用机器人、物理状态、传感器、任务 buffer 和命令的重置路径。`None` 用于初始化或手动 reset，不生成结束回合日志；显式布尔 mask 会先保存结束回合统计，再清零对应行。

TensorBoard 以 `locomotion/`、`recovery/`、`jump/`、`multi_terrain/` 为独立顶层，不再将任务统计集中到 `Episode/`。每个任务内按以下路径分类：

| 路径 | 内容 |
| --- | --- |
| `<task>/rewards/<name>` | 各奖励分量的回合累计值 |
| `<task>/metrics/<name>` | 成功率、接触率、高度、起身用时等任务指标 |
| `<task>/episode/duration_s` | 回合时长 |
| `<task>/curriculum/stage`、`iteration`、`sampled_env_steps` | 任务课程阶段、折算进度及累计样本数 |
| `<task>/allocation/num_envs`、`env_fraction` | 当前环境配额及占比 |
| `multi_terrain/terrain/<terrain>/...` | 多地形内部各地形的成功率、难度及停车等统计 |
| `shared/curriculum/stage` | 共享随机化课程阶段 |

回合项只使用该任务本次结束的回合，保留逐回合样本供 logger 求均值；奖励只输出当前权重非零或本回合已有累计值的分量。环境配额及任务课程进度每轮记录，便于检查延迟启动和配额变化。PPO 的 `Loss/`、`Policy/`、`Perf/`、`Train/` 保留框架分组。新标签在下次启动或续训后生效，已有 event 文件中的旧标签仍保留。评估从 checkpoint 的各任务采样进度冻结对应课程。

逐步诊断由任务的 `EPISODE_METRICS` 声明：locomotion 和 multi_terrain 记录高度、姿态、速度跟踪 gate 与机身接触率；recovery 记录高度/姿态 gate、接触率、起身成功率和起身用时；jump 记录接触率、成功率、机身峰值高度和轮子峰值离地高度，不输出恒为 1 的 gate。`alive_gate` 与接触率互补，不重复记录。multi_terrain 的地形子任务统计也只在对应子任务有回合结束时输出。

## 网络与观测

网络参照 `experiments/beni_rl/rl_cfg.py`：actor 与 critic 都使用观测归一化 → GRU（512 hidden、1 层）→ MLP（512/256/128、ELU）。actor 使用 GaussianDistribution，初始标准差 1.0；PPO 默认学习率 1e-3、rollout 长度 24。

所有任务共用六维动作：四个腿关节位置目标和两个轮速目标，共享缩放、裁剪和一拍动作延迟。控制步为 0.01 s，物理子步维持原实现。

actor 包含 IMU、重力方向、关节/轮测量、6D actions、六维命令、ToF 与六维跳跃阶段编码。命令依次为缩放后的前向速度、偏航速度、高度目标，以及未缩放的三维 jump 模式 one-hot；非 jump 环境的模式为零。critic 再追加速度、接触等仿真真值，以及 baselink 原点的世界坐标 z（不扣除地形高度）。默认 ToF 配置下 actor 为 51D、critic 为 75D；选择任务子集不会改变网络输入维度。

jump 从空旷跑道静止出生，使用同一策略推理加速；连续达速并站稳后保留物理状态移到指定起跳距离，再发送跳跃编码并开始独立计时。助跑不发奖励、不参与 PPO 更新；跳跃编码更新后产生的轨迹进入 PPO，GRU 保留切换时的隐状态，训练不回放助跑过程。加速容差/超时、起跳距离表、恢复姿态、各任务比例与奖励在对应配置中调整。
