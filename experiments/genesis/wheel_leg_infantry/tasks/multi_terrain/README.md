# 任务驱动多地形训练

独立入口 `tasks.multi_terrain`，一套策略同时训练七种通关任务和保留用的 locomotion。
机器人、控制器、IMU/ToF 观测及三维命令接口复用 locomotion；不额外向 actor 输入任务 ID 或终点真值。
相同观测配置的 locomotion 权重可通过 `--load-weights` 迁移。

```bash
.venv/bin/python -m experiments.genesis.wheel_leg_infantry.tasks.multi_terrain.train \
  -e multi_terrain_v1 -B 64 --max_iterations 25001

# 迁移已有策略；优化器和所有任务课程从头开始。
.venv/bin/python -m experiments.genesis.wheel_leg_infantry.tasks.multi_terrain.train \
  -e multi_terrain_v1 -B 64 --load-weights logs/locomotion_v4/version_0000/model_10000.pt

# 保存/恢复各任务等级、成功率窗口及 epoch，同时恢复 PPO 状态。
.venv/bin/python -m experiments.genesis.wheel_leg_infantry.tasks.multi_terrain.train \
  -e multi_terrain_v1 --resume --max_iterations 20001
```

训练使用本目录的 `config.py`，`--load-weights` 同样使用该配置。
`--resume` 默认恢复运行时保存的配置；如需续训时使用当前 `config.py`，添加 `--resume-config current`。

训练需要可用的 CUDA/Genesis 环境。默认每张 GPU 4096 个环境；所有任务的课程几何预先构建为独立区域，
增大环境数量前应先测显存与吞吐。示例 checkpoint 路径需替换为实际文件。

当前 Genesis ToF Raycaster 会为每个环境分配整个场景的三角面 BVH，即使地形使用
`batch_fixed_verts=False`，该 BVH 也不会随静态顶点一起共享。以 127196 个碰撞面为例，
4096 个环境仅 AABB、BVH 节点和主要排序缓存就约需 63 GiB，尚未包含物理仿真和 PPO。
24 GiB GPU 请先使用 `-B 64`；不要直接沿用平地训练的 4096。构建时若出现字段 shape
异常或显存不足，先降低 `-B` 验证；增加任务难度等级或网格精度也会增加面数与显存。

## 双 GPU 训练

在项目根目录使用 `torchrun` 启动两个进程；`-B` 始终表示**每张卡**的环境数：

```bash
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node=2 \
  --module experiments.genesis.wheel_leg_infantry.tasks.multi_terrain.train \
  -e multi_terrain_2gpu -B 4096 --max_iterations 4001

# 同一个实验续训；两卡从同一个 checkpoint 恢复。
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node=2 \
  --module experiments.genesis.wheel_leg_infantry.tasks.multi_terrain.train \
  -e multi_terrain_2gpu -B 4096 --resume --max_iterations 8001
```

每个进程独立仿真 4096 个环境，RSL-RL 同步 PPO 参数和梯度，总环境数 8192。
每轮采样总量为 `4096 × 2 × num_steps_per_env`；默认 24 步即 196608 条 transition。
首次验证建议使用 `-B 16 --max_iterations 2`，再根据每卡显存增加环境数；两卡显存不会合并供单个场景使用。
支持单机多卡，启动时 `CUDA_VISIBLE_DEVICES` 使用数字 GPU 编号（可指定顺序）。
入口按 `LOCAL_RANK` 为每个进程只保留一张卡，并设置相同物理编号的 `QD_VISIBLE_DEVICE`；
各进程内的设备均为 `cuda:0`，项目的 Runner 适配层处理此编号与分布式 rank 的区别。

仅 rank 0 创建版本目录、保存配置、日志和 checkpoint；所有进程共享该目录，种子为 `--seed + rank`。
`run_metadata.json` 记录 `world_size`、`num_envs_per_gpu`、`total_num_envs` 和 `worker_seeds`。
`--vis` 仅在 rank 0 显示。每卡任务配比相同，各自按本地回合推进成功率课程；
TensorBoard 的任务指标和 checkpoint 的课程窗口来自 rank 0，续训时所有卡从该课程快照恢复。
`--load-weights` 在所有卡加载同一份权重，并重新开始优化器与课程。

## 默认任务与配比

新训练默认启用 `obs_cfg["critic_task_one_hot"]`，仅在 critic 观测末尾追加 8 维任务 one-hot，actor 输入不变。
编码顺序固定为 `locomotion, stairs_down, stairs_up, slope_up, slope_down, square_wave, platform_ridge_down, platform_ridge_up`；
禁用部分任务或单任务评估时仍使用相同的 8 维编码，课程重新分配任务后随环境 reset 更新。
旧 run 的保存配置没有该字段时，续训和评估保留旧观测维度。
通过 `--load-weights` 迁移旧默认 MLP 权重到新训练时，critic 第一层新增的 8 列权重置零，保留初始 value。
旧 checkpoint 要启用此功能应使用 `--load-weights` 开始新 run；完整 `--resume` 要求模型和优化器维度一致。

配置入口：`config.py` 的 `get_cfgs()`，最终保存在 `env_cfg["multi_terrain"]["tasks"]`。

| 任务 | 最终目标 | 默认课程 | 权重 |
| --- | --- | --- | --- |
| `locomotion` | 平地速度、高度跟踪、姿态保持与静止采样 | 保留经典最终命令与奖励，默认不升降级 | 2 |
| `stairs_down` | 下一级台阶 | 高度 15 → 20 → 25 cm | 1 |
| `stairs_up` | 上一级 20 cm 台阶 | 高度 15 → 17 → 20 cm | 1，已启用 |
| `slope_up` | 上一个独立坡面 | 坡角等级：5 → 10 → 15 → 20 → 23 → 25°，高差 25 cm | 1 |
| `slope_down` | 下一个独立坡面 | 与上坡相同，但独立统计成功率 | 1 |
| `square_wave` | 登上一级小高度台阶 | `square_wave` 高度 2 → 2.5 → 3 → 3.5 → 4 → 4.5 → 5 cm | 1 |
| `platform_ridge_down` | 沿 +x 越过凸台，经第一平台下到地面 | 第二平台与凸台高差 1 → 2 → 3 → 4 → 5 cm | 100 轮前为 0，之后为 1 |
| `platform_ridge_up` | 沿 +x 从地面登上第一平台、越过凸台，落到第二平台 | 凸台高度 15 → 20 → 25 → 30 → 35 cm，平台高度同步增加 | 100 轮前为 0，之后为 1 |

几何和难度唯一来源是 `terrains` 注册模块的 `PARAMETERS` 与 `LEVELS`。
上述表格描述当前默认值；修改模块难度表会作用于新训练，已有训练使用保存的 `level_catalog` 快照。
上台阶从 15 cm 开始，按该任务的独立成功率课程推进到 17 cm、20 cm。

任务层不再重建单级楼梯或小高度上台阶。
台阶、坡面模块提供 `traversal_end`，任务终点沿通行方向位于其后 `landing_length` 处；台阶和坡面默认 0.8 m。
下台阶默认末沿 x=0.8 m，终点 x=1.6 m，地块长 16 m，回合上限 10 s。
`square_wave` 保留原入口名，现为单级上台阶：从低平地出生，台阶沿 x=0.5 m，高平台延伸至地块末端，终点 x=1.3 m，回合上限 10 s。
独立坡面的参数按注册等级读取，水平坡长由 `height / tan(angle)` 决定。

`enabled=False` 或 `weight=0` 关闭某任务，其他任务按权重重新归一化。
环境按最大余数法分组，同一全局阶段内不重抽任务。全局阶段改变任务权重时，重新计算目标配额；
每个环境在下一次 reset 时才采用新任务，正在运行的回合保留原来的奖励、命令和成功率归属。
所有环境完成一次 reset 后达到新配比；稳定阶段内，回合长短不改变环境占比。
由于环境数量为整数，比例存在最多一个环境的取整误差；任何阶段权重大于 0 的任务分不到环境时直接报错。
训练必须保留 `locomotion.enabled=True` 且 `weight>0`。

## 间断通行模式

七种通关任务都可按**回合比例**混入间断通行，默认各为 30%；剩余回合保持原通关方式。
`locomotion` 继续使用原来的静止采样。每个 task 的配置独立，例如：

```python
intermittent=_intermittent_cfg(
    probability=0.3,                         # 0 关闭，1 全部采用；每次 reset 独立抽样
    approach_speed_m_s=0.4,                   # 接近阶段速度上限
    pre_stop_distance_range_m=[0.2, 0.4],     # 第一障碍沿前，触发停车的轮心距离
    post_stop_distance_range_m=[0.35, 0.6],   # 最后障碍沿后，触发停车的轮心距离
    hold_duration_range_s=[0.5, 1.5],         # 连续停稳时长，本回合两次停车共用
    stop_timeout_s=4.0,                       # 每次停车的制动 + 停稳总时限
    max_braking_distance_m=0.8,               # 收到零指令到首次停稳的最大位移
    stop_position_tolerance_m=0.15,           # 首次停稳后相对固定位置的最大漂移
    stop_lin_vel_m_s=0.08,                    # 世界水平速度的停稳门限
    stop_ang_vel_rad_s=0.15,                  # 三轴角速度模长的停稳门限
    edge_margin_m=0.02,                       # 边缘前轮子外缘的最小余量
    restart_distance_m=0.3,                   # 第二次停车后必须重新前进的距离
    extra_episode_time_s=6.0,                 # 为慢速接近/再次起步增加的运动时间
),
```

`platform_ridge_up` 的末端平台仅长 1.2 m，默认沿后停车触发距离为 0.25～0.35 m、
最大制动位移为 0.2 m、重新起步距离为 0.2 m；其他任务使用上面的通用默认值。
配置会检查停车及重新起步是否留有足够地面空间。

也可在 `curriculum.overrides[level]["intermittent"]` 中按难度覆盖比例或参数。
比例是回合抽样概率，不是严格的环境配额或训练时间占比；间断回合通常更长。
参数默认值位于 `intermittent.py`，`config.py` 的 `_intermittent_cfg()` 复制完整配置到每个任务快照。

一个被选中的回合必须顺序完成：

1. **接近**：保持本回合机身高度，前进指令取任务采样速度与 `approach_speed_m_s` 的较小值。
2. **边缘前停留**：任一轮心进入抽样的沿前距离后，线速度和偏航速度设为零；双轮必须留在第一障碍沿前。
3. **通过地形**：连续停稳达标后恢复本回合采样的正向速度。
4. **通过后停车**：双轮越过最后障碍沿、满足轮子余量且接地，并到达抽样的沿后距离，再给零速度指令。
5. **重新起步**：再次连续停稳后恢复行驶；实际前进至少 `restart_distance_m` 且满足原通关条件，才结束回合。

台阶/坡面入口使用真实几何起点；平台凸台下行入口是凸台前沿，上行入口是第一平台前沿。
组合地形在整个障碍之前和之后各停一次，不要求在每个中间边缘停车。
沿前距离取更靠近入口的轮心，沿后距离取更靠近末沿的轮心，支持斜向接近。
距离是**发出停车指令的位置**，实际停车还会产生制动位移。

停稳需要同时满足速度门限、双轮接地、机身未触地和位置约束；任一条件不满足会清零连续停稳计时。
边缘前越沿、停车位移越界或停车超时判为失败；两次停车不能靠等待时间或提前到终点跳过。
制动完成后固定停留位置，后续振荡不会重新选取锚点，避免靠缓慢漂移不断移动停车位置。

停车期间暂停停滞计时，屏蔽前进/通关及上台阶动作引导奖励，保留平衡、碰撞和零速跟踪约束。
各 task 的 `rewards` 新增 `standing_drift=-3.0` 与 `intermittent_stop=3.0`；后者每次连续停稳达标只发放一次。
普通通关回合的命令与停滞判定保持原样。间断回合的正向行驶命令在 reset 时抽样后保持，避免周期重采样覆盖停车指令。
回合时限为原时限 + `extra_episode_time_s` + 已实际消耗的停车时间；每次停车仍受独立的 `stop_timeout_s` 限制。

TensorBoard 在 `task/<任务名>/` 下分别记录 `ordinary_success`、`intermittent_success`、
`intermittent_fraction`、`intermittent_pre_stop_completed`、`intermittent_post_stop_completed`、
`intermittent_stop_failure` 和 `intermittent_max_stop_drift_m`。最后一项不包含首次停稳之前的制动距离。
任务成功率课程统计两种模式的完整回合结果。

状态机使用训练地形真值安排命令，不增加 actor/critic 观测维度，不要求部署端提供阶段编号。
缺少 `intermittent` 的旧配置默认关闭。**旧 run 直接 `--resume` 会继续使用旧保存配置**；
启用新模式需使用 `--resume --resume-config current`，或用 `--load-weights` 新建训练。

回归检查（不启动训练）：

```bash
.venv/bin/python -m unittest discover -s tests/genesis -p test_intermittent.py -v
# 另含八个环境、全部任务最低难度的 CPU 短仿真；关闭 ToF 和随机化。
NGXY_INTERMITTENT_SMOKE=1 .venv/bin/python -m unittest discover -s tests/genesis -p test_intermittent.py -v
```

## 独立课程、奖励与命令

每个任务的主要字段：

- `enabled`、`weight`：是否启用、环境数量权重。
- `commands`：前进速度、角速度、高度范围及静止概率。
- `resampling_time_s`：本任务自己的命令重采样周期。
- `rewards`：本任务完整奖励权重表，未指定的奖励为零。
- `terrain`：注册地形的 `preset`、地块尺寸等公共参数；高度等难度参数由 `LEVELS` 提供。
- `traversal`：障碍末沿后的落地距离。
- `termination`：最大时长、姿态与接触容忍时间、赛道宽度、停滞超时。
- `success`：终点允许姿态、轮子越过障碍的余量与落地高度误差。
- `curriculum`：本任务初始/最低/最高难度、成功率窗口和升降级阈值。

每个 task 在 `config.py` 中独立集中配置，例如小高度上台阶：

```python
terrain=dict(preset="square_wave", tile_size=[16.0, 6.0]),
traversal=dict(landing_length=0.8),
curriculum=dict(
    enabled=True,
    difficulty=0,           # 初始 LEVELS 等级
    min_difficulty=0,
    max_difficulty=None,    # 当前快照中的最高等级；也可指定整数上限
    window_episodes=100,
    promote_threshold=0.8,
    demote_threshold=0.3,
),
```

每项默认统计至少 100 个同等级、同 epoch 的结束回合，成功率 ≥0.8 升一级、≤0.3 降一级。
升降级由本目录 `curriculum.py` 的 `TaskCurriculum` 统一实现，各 task 独立设置窗口和阈值。
只在该环境下一次 reset 应用新难度；旧等级仍在运行的回合保留原来的地形、命令和奖励，
不污染新课程窗口。关闭课程后固定在 `difficulty`。例如直接练小高度上台阶最高级，设 `difficulty=6`。

若某些难度需要调整命令或奖励，可在该 task 的 `curriculum` 中添加：

```python
overrides={
    1: dict(commands=dict(lin_vel_range=[0.25, 0.45]), rewards=dict(task_progress=7.0)),
},
```

这些行为覆盖项按难度累计继承，支持 `commands`、`rewards`、`termination`、`success`、
`resampling_time_s`、`tracking_success`、`landing_stability` 和 `ascent`。不允许覆盖几何；几何只在 `terrains` 中维护。

上楼梯各等级的机身目标高度在 `tasks["stairs_up"]["curriculum"]["overrides"]` 的
`commands.base_height_range` 中分别配置，对应 `terrains/stairs_up.py` 的 `LEVELS` 等级。
例如 `0: dict(commands=dict(base_height_range=[0.26, 0.26]))` 表示 0 级固定为 0.26 m；
`[0.24, 0.28]` 表示在该范围内随机采样。当前各级均保留 0.36 m，可按需要分别修改。
这里的高度是机身原点相对脚下地面的目标高度，不是台阶高度或世界坐标 z。
若全局阶段的 `task_command_ranges.stairs_up` 也指定了 `base_height_range`，则以全局阶段覆盖为准。

locomotion 固定使用 plane 的 0 级，完整时长回合内的线速度、角速度、高度 RMSE 达标记为成功，
阈值在 `tracking_success` 中。全局 `curriculum_cfg` 由 `config.py` 的 `_get_curriculum_cfg()` 返回，控制动力学随机化、传感器噪声、双轮独立打滑强度、任务配额和各地形命令范围。
打滑使用独立目标 `"wheel_slip": {"strength": 0.4}` 和开关 `--wheel-slip/--no-wheel-slip`；
默认课程与动力学随机化使用相同强度，各目标可独立调整。
配置、时间单位和摩擦恢复规则见 [双轮独立随机打滑](../locomotion/README.md#双轮独立随机打滑)。
以下示例将两种凸台延后到 15000 轮启用；当前默认配置在 100 轮启用两种凸台，具体以 `config.py` 为准：

```python
def _get_curriculum_cfg():
    """按阶段配置随机化、传感器噪声与任务权重；15000 轮起加入凸台任务。"""
    return {
        "enabled": True,
        "stages": [
            {
                "name": "clean",
                "start_iteration": 0,
                "targets": {
                    "domain_rand": {"strength": 0.0},
                    "sensor_noise": {"strength": 0.0},
                    "task_weights": {
                        "platform_ridge_down": 0.0,
                        "platform_ridge_up": 0.0,
                    },
                },
            },
            {
                "name": "randomized",
                "start_iteration": 1000,
                "targets": {
                    "domain_rand": {"strength": 0.4},
                    "sensor_noise": {"strength": 0.4},
                },
            },
            {
                "name": "full",
                "start_iteration": 1500,
                "targets": {
                    "domain_rand": {"strength": 1.0},
                    "sensor_noise": {"strength": 1.0},
                },
            },
            {
                "name": "platform_ridge_down",
                "start_iteration": 15000,
                "targets": {
                    # 保留已有任务及其权重，新增凸台任务；随机化和噪声继承 full 阶段。
                    "task_weights": {
                        "platform_ridge_down": 1.0,
                        "platform_ridge_up": 1.0,
                    },
                },
            },
        ],
    }
```

在任意阶段的 `targets` 中添加 `task_command_ranges`，按任务名覆盖 command 范围，例如：

```python
"task_command_ranges": {
    "locomotion": {
        "lin_vel_range": [-1.0, 1.0],
        "ang_vel_range": [-2.0, 2.0],
        "base_height_range": [0.22, 0.30],
        "standing_probability": 0.3,
    },
    "stairs_down": {"lin_vel_range": [0.5, 1.0]},
    "platform_ridge_down": {"lin_vel_range": [0.8, 1.5]},
},
```

任务名为 `locomotion`、`stairs_down`、`stairs_up`、`slope_up`、`slope_down`、
`square_wave`、`platform_ridge_down`、`platform_ridge_up`，与 `tasks` 中的名称一致。支持上例中的四个字段。
全局阶段覆盖优先于该任务各难度的 `curriculum.overrides.commands`，对该任务所有难度生效；
未覆盖字段继续使用各难度配置。后续阶段只填写要改变的任务/字段，其他项累计继承，
空字典表示没有新变化。范围必须有限且下限不大于上限，机身高度为正，通关任务线速度下限为正。
已禁用任务的覆盖不会重新启用任务。

阶段切换时更新采样范围，已有 command 保持到下一次按任务周期重采样或回合 reset；
其他任务不受影响。第 0 阶段在首次采样前生效，续训按恢复轮次重建累计覆盖。
运行时覆盖不修改原始任务和难度快照。关闭全局课程则使用任务各难度原始 commands。

后续阶段继承此前的目标值；凸台阶段保持随机化和噪声强度 1.0，其他任务保留各自权重。
两种凸台各自的几何及 5 个难度在初始化时预建，前期不分配环境；启用后各自从 0 级开始独立统计成功率并升降级。
`enabled=False` 或任务初始 `weight=0` 会禁用该任务，全局 `task_weights` 不会重新启用未构建的任务。
关闭全局课程时使用任务原始权重（凸台为 1），因此会从训练开始就分配凸台环境。
训练入口默认上限为 25001 轮，给 15000 轮后新增的任务留出训练时间。

`platform_ridge_down` 从 x=0 的第二平台沿 +x 出发；越过 35 cm 凸台、20 cm 第一平台后下到地面，
继续前进 0.8 m 到终点。`platform_ridge_up` 从 x=0 的地面沿 +x 登上第一平台、越过凸台，
在第二平台上继续前进 0.8 m 到终点。两者独立配置奖励、命令和成功率课程。
地形几何仍沿 +x 构建；斜向训练时，小车沿每次出生 yaw 跨越固定朝向的障碍。

reset 的位置和姿态范围仍由 `config.py` 末尾的 `env.update(...)` 统一配置：
`base_init_pos_range` 依次为 x/y/z 的米制范围（z 相对当地地面），
`base_init_rpy_offset_range_deg` 依次为 roll/pitch/yaw 的角度范围。
两种凸台均保留这些随机量，不再强制平移或旋转小车。
默认 yaw 为 `[-30, 30]` 度，兼顾正向和左右斜向跨越；roll/pitch 保持为 0。

所有通关任务在 reset 时保存本回合的实际出生位置与 yaw，直到该环境再次 reset 才更新：

- 航向奖励相对出生 yaw 计算；横向奖励和走廊限制相对出生点沿该 yaw 的直线计算。
- 进度投影到出生航向，沿该方向到达障碍末沿后再前进 `landing_length` 到终点。
- 双轮越沿仍按真实地形的 x 边缘判断，并检查接地和姿态，不能靠横移或从侧面绕行通关。
- 台阶专用的接触、支撑、伸腿和后摆奖励也使用同一条斜向走廊。
- 地块实际安全边界仍有效。增大 yaw 时若斜线路径超出地块，应扩大 `terrain.tile_size` 的 y 尺寸。

通关任务出生 yaw 必须朝向障碍，与地形通行方向的夹角小于 90 度；平地任务不受此限制。

`--resume` 默认读保存配置及难度表快照；变更任务集合、难度表或权重应新建训练并 `--load-weights`。
旧的 `geometry` / `curriculum.stages` 格式及其课程状态不再兼容，旧策略可以通过 `--load-weights` 迁移。

## 通关奖励与结束条件

七种通关任务禁止非零 `alive` 奖励；locomotion 保留经典奖励。
通关任务默认使用：

- `task_progress=5`：每前进到本回合新的最远位置奖励 5 × 新增米数，最多计算到终点。
- `task_success=20`：成功终止一次奖励 20，不乘 `dt`。
- 横向偏离、朝向偏离、动作变化、姿态、关节限位、接触和失败惩罚。

进度奖励也不再乘 `dt`；来回倒车不能重复获得进度。其他连续奖励按 `dt` 积分，
`death` 是一次性失败惩罚。默认通关任务不启用持续速度/高度跟踪正奖励。

### 上台阶三阶段奖励

`stairs_up` 在现有水平进度奖励之外增加靠近时抬升、搭接前后摆、稳定支撑后缩腿的增量奖励，以及稳定搭台事件奖励。
参数集中在 `config.py` 的 `tasks["stairs_up"]["ascent"]`。

上台阶的水平进度以左右轮心平均位置为准，reset 时也以该位置记录起点；
轮子不动而 baselink 向前倾倒不会产生前进奖励，也不会重置停滞计时。
历史最远进度和 `progress_m` 使用同一参考，往返不重复领奖，进度最多计到终点。
其余通关任务仍以 baselink 水平位置计算进度，最终成功条件保持原有机身、双轮及姿态要求。

`stairs_up_approach_lift=2`：机体原点在第一级台阶前 `approach_distance_m=0.35 m` 内，
双轮仍在下方地面接触、底盘没有接触且尚未获得搭接奖励、roll/pitch 满足成功姿态门限时，
对底盘下底面向“眼前台阶顶面高度 + `approach_clearance_m=0.01 m`”的新增抬升发奖。
下底面使用 baselink 实际碰撞体的最低点，包含碰撞体自身偏移和机身 roll/pitch 的影响，
不使用视觉网格或固定机体原点高度。15 / 17 / 20 cm 台阶对应下底面目标为 16 / 18 / 21 cm
（相对下方地面）；达到余量后继续抬高不再增加奖励。每拍奖励为：

`2 × max(0, min(当前下底面高度, 台阶顶面高度 + 余量) − 此前最高下底面高度) / max(台阶顶面高度 + 余量 − 出生下底面高度, 1e-6)`。

此前最高下底面高度在 reset 时初始化为出生下底面高度（截断到目标高度），随后每拍更新。
奖励不乘 `dt`，每回合累计最多 +2；保持高度、反复升降、离开再进入奖励区都不能重复领奖。
区域外或腾空获得的高度也会更新最高值，不能在进入奖励区或落地时补领奖励。
出生时已达到目标高度则不发该项；同拍失败时不发正奖励。
越沿余量和 +2 系数是初始训练设置，可在 `ascent` 和 `rewards` 中分别调整。

`stairs_up_backswing=2`：进入前摆/落台阶段之前，鼓励左右腿向后摆。
底盘首次接触台面之前，要求在台阶沿前后 `approach_distance_m` 范围内、双轮仍位于立面后方且机身没有接触。
底盘首次真实触台之后，不再限制接近距离、机身接触或轮子是否越沿，稳定搭接后仍可继续获得后摆奖励。
前后两段都要求机身及双轮处于赛道范围内，且未超过失败姿态门限。
`approach_distance_m` 是非负的范围宽度，不能用负数表示沿后位置；设为 0 不影响触台后的后摆奖励。
使用实测关节角计算的虚拟腿角，当前模型正值为后摆，`backswing_angle_deg=20` 为初始目标，
低于当前 ±30° 动作摆角限幅。无需等待抬升完成，也不要求双轮一直接地，允许腿/轮抵住立面时后摆。

每条腿的进度为 `clamp(腿角 / 目标后摆角, 0, 1)`，只对超过本回合该腿历史最高进度的增量发奖，
两腿增量取平均后乘 2，不乘 `dt`。左右腿各最多贡献 +1，每回合合计最多 +2；
历史最高进度只在上述有效区域和阶段内更新；区域外的后摆不消耗奖励额度，进入有效区域后
可获得超过此前有效区域历史峰值的进度奖励。保持姿态、反复前后摆、离开再进入都不能重复领奖，同拍失败不发奖励。
出生已有的后摆进度不计奖。首次真实台面接触（向上力 >0）即启用触台后的窗口，
无需等待搭接奖励的 5 N 支撑力、搭接深度或 0.10 s 稳定时间；仅碰侧壁不会启用该窗口。
进入前摆/落台阶段或双轮直接落台当拍立即关闭后摆奖励，并锁存至 reset，之后失去接触或轮子降低也不会重新启用。
此项只提供奖励引导，不直接修改动作，也不要求最终成功必须完成后摆。

`stairs_up_retract` 复用稳定搭接门限：baselink 接触点须深入台面至少 `support_margin_m`，
这些接触点的向上支撑力合计须达到 `support_force_n`，并连续稳定支撑 `support_duration_s`。
当前配置为 10 cm、5 N、0.10 s，同时保留支撑窗口的竖直速度和角速度门限。
首次满足稳定支撑时记录左右腿长，随后仅在继续满足稳定支撑且双轮尚未落台时，
对相对该基准的新增缩短量发奖；无需轮子越沿或水平前进。
每腿进度为 `clamp((首次稳定支撑时腿长 − 当前腿长) / retract_shortening_m, 0, 1)`；
只奖励超过该腿历史最高进度的增量，两腿取平均后乘奖励权重，不乘 `dt`。
当前 `retract_shortening_m=0.05 m`、权重为 20，因此左右腿各缩短 5 cm 时累计最多 +20；
保持腿长、反复伸缩不能重复领奖，首次稳定支撑之前的缩短量不补奖。
支撑中断后暂停奖励，重新连续满足窗口时才恢复；暂停期间的缩短量也不补领。

机身当前已接触台面、双轮轮底均达到台面高度（允许 `wheel_bottom_tolerance_m` 容差），
且双轮尚未落台时，进入落台阶段并永久关闭该回合缩腿奖励；不要求先拿到稳定搭接奖励。
此时同时启用前摆、伸腿引导，不要求前摆到目标角度后才伸腿，也不直接修改控制动作：

- `stairs_up_forward_swing=5`：从进入落台阶段时的腿角开始，鼓励向负腿角目标靠近的新增前摆，两腿进度取平均；
  `forward_swing_angle_deg=20` 仅表示奖励饱和角度，不是伸腿或成功判定门槛。
- `stairs_up_extend=5`：以进入落台阶段时的腿长为基准，对每腿新增伸长量发奖，
  `landing_extension_m=0.05 m` 时进度饱和。逐轮要求轮心已越沿、轮底不低于台面容差，避免向立面伸腿也拿奖励。
- `stairs_up_wheel_support=10`：落台阶段双轮首次同时落在上台面，且各自有真实的向上台面接触力时发一次奖励；
  单轮、侧壁接触或仅腾空越沿不算支撑。

前摆、伸腿奖励不乘 `dt`，仅奖励超过历史峰值的增量，各项每回合最多发其配置权重。
进入该阶段前已有的前摆和伸长量不补奖，反复伸缩或前后摆不能重复领奖。
同拍失败不发正奖励，部分 reset 仅清空相应环境的阶段、基准和峰值。
这三项奖励均为零或未配置时不启用新阶段，旧存档保持原有收腿切换行为。

双轮第一次同时落在顶部台面，当拍后永久关闭该回合前摆与伸腿引导，不等稳定站立或最终成功。
判定要求双轮都有接触、轮心进入台面至少 `wheel_edge_margin_m=0.01 m`、位于赛道范围内，
且轮底距顶部台面不超过 `wheel_bottom_tolerance_m=0.005 m`；单轮落台、腾空越沿均不触发。
双轮落台后，任一轮底低于台面超过 `fall_height_tolerance_m=0.03 m` 就当拍按失败终止，
无需等待落到下方地面或停滞超时；向上弹跳不会触发，也不会重启缩腿奖励。
即使没有 baselink 搭台、直接双轮登台，跌台终止也生效。

`stairs_up_wheel_forward_action=1`：双轮已经落上台面且当前双轮接地时，若机体系前向速度
`base_lin_vel[:, 0] <= commands[:, 0]` 且前向命令为正，持续奖励轮子向前转的 action。
逐轮取裁剪后 action 的正值部分，再取双轮均值；倒转和零 action 不发奖，不使用实际轮速。
每拍奖励为 `权重 × mean(max(wheel_action, 0)) × dt`；超过 command、腾空、跌台或同拍失败时为 0。
直接双轮登台也能启用，无需先领取机身搭台或落台支撑事件奖励。
该项使用现有 `wheels_landed` 状态，reset 后需重新落台，日志记录为 `reward_stairs_up_wheel_forward_action`。

`stairs_up_takeoff=10`：本回合确认双轮都曾接触台下地面后，双轮轮底第一次同时高出台下地面
`wheel_bottom_tolerance_m=0.005 m` 就发放一次 +10，不乘 `dt`，不要求 baselink 已搭台。
单轮抬起或只有接触信号消失不发奖；轮子离地后抵住立面仍可触发。
空中出生不会直接领奖，需先实际双轮接触台下地面。离地状态锁存至 reset，之后任一轮
再次接触台下地面就当拍按失败终止，无需先登上台面；落到上台面不会触发此失败。
同拍失败时不发离地正奖励。

`stairs_up_landing_distance=10`：任一轮第一次真实接触顶部台面时，按接触点沿 +x 距台阶入口边缘
的水平距离（m）乘权重发奖，不乘 `dt`。例如首次落点深入台面 10 cm 得 +1，30 cm 得 +3；
距离越大奖励越大，不作额外饱和截断，落点必须位于平台赛道范围内。
使用 Genesis 真实轮子接触点，要求向上受力、接触高度距台面 ≤15 mm、法向竖直分量绝对值 ≥0.9；
baselink 搭台、侧壁和台下地面接触不算。单轮先落台就立即发放，不等另一轮或稳定站立；
同拍多个轮/接触点触台时取离入口边缘最近的一个距离。
即使第一次恰好触到边缘得到 0，也会锁存本回合机会；之后滚动前进、另一轮落台或弹起重落均不补奖。
同拍失败时不发该正奖励，部分 reset 仅清空所选环境的落点记录。

以下两个事件同样不乘 `dt`，每回合各最多发放一次：

- `stairs_up_support=10`：baselink 的真实接触点进入顶部台面至少 10 cm，接触点高度距台面不超过
  15 mm，接触法向竖直分量绝对值 ≥0.9，向上支撑力合计 ≥5 N，并连续稳定支撑 0.10 s。
  侧壁、低处地面、腿/轮接触和瞬间撞击不计入。
- `task_success=20`：保留到达上平台终点、双轮越沿并接地、底盘脱离接触、roll/pitch 各不超过 10°
  的条件；底盘离台面高度还需 ≥0.20 m，并连续稳定站立 0.20 s。直接登上台阶也可以成功，
  不强制先领取搭台或缩腿奖励。

支撑和站立窗口均要求竖直速度 ≤0.3 m/s、roll/pitch 角速度合范数 ≤1.5 rad/s；
水平前进速度不受此稳定门限限制。窗口中断会重新计时，已发奖励不会因失去接触或倒车重发。
部分环境 reset 只清空对应回合的状态；同拍失败时不发上述正奖励。

`stairs_up_wheel_wall_force=-0.05`：持续惩罚轮子撞台阶竖直立面的法向接触力。
逐接触点取 `abs(轮子受力 · 接触法向)`，将双轮所有有效立面接触的力相加（N），
每拍奖励为 `-0.05 × 法向力总和 × dt`；例如总力 100 N 持续 0.1 s，累计扣 0.5。
只作用于 `stairs_up`，不要求已搭台或已离地，失败当拍仍计入惩罚。

立面判定要求接触属于轮子、位于台阶起始沿至末沿的 x 范围及台下地面至顶部台面的高度范围，
位置容差为 `wall_surface_tolerance_m=0.015 m`，且处于任务走廊内。
接触法向需近似水平：`abs(normal_z) <= wall_normal_z_max=0.3`。
台面/地面的竖直支撑接触、机身/腿部接触和沿立面的切向摩擦不计入；
离开接触后惩罚立即清零，部分 reset 仅清空所选环境的力状态。

TensorBoard 新增 `task/stairs_up/support_reached`、`task/stairs_up/retract_reached`、
`task/stairs_up/takeoff_reached`、`task/stairs_up/first_landing_reached` 回合到达率，
以及 `task/stairs_up/first_landing_distance_m` 首次轮子触台距离（未触台回合记为 0），
其中 `retract_reached` 表示落台前两腿都曾达到目标缩短量，
以及 `reward_stairs_up_support`、`reward_stairs_up_retract`、`reward_stairs_up_approach_lift`、
`reward_stairs_up_backswing`、`reward_stairs_up_takeoff`、`reward_stairs_up_landing_distance`、
`reward_stairs_up_wheel_wall_force` 奖励统计。
落台引导还记录 `task/stairs_up/landing_phase_reached`、`forward_swing_progress`、
`extension_progress` 和 `wheel_support_reached`，后三项同样使用 `task/stairs_up/` 前缀。
对应奖励统计为 `reward_stairs_up_forward_swing`、`reward_stairs_up_extend` 和 `reward_stairs_up_wheel_support`。

使用当前任务配比开启新训练（保留其他已启用任务及 locomotion）：

```bash
.venv/bin/python -m experiments.genesis.wheel_leg_infantry.tasks.multi_terrain.train \
  -e stairsup_v1 -B 64 --max_iterations 25001 \
  --load-weights logs/locomotion_v4/version_0002/model_10000.pt
```

该命令读取当前代码中的奖励和 15 → 17 → 20 cm 课程。`--resume` 默认读取旧运行保存的配置，
不会自动启用新增奖励；迁移已有多地形策略时也应使用 `--load-weights` 指向所选 checkpoint。

```bash
.venv/bin/python -m unittest tests.genesis.test_stairs_up_rewards -v
NGXY_STAIRS_UP_SMOKE=1 .venv/bin/python -m unittest \
  tests.genesis.test_stairs_up_rewards.StairsUpSimulationTests -v
```

下台阶额外配置 `stairs_down_landing_stability=-2.0`，专用参数集中在该 task 内：

```python
landing_stability=dict(duration_s=0.3, ang_vel_weight=0.25),
```

任一轮首次越过台阶末沿并接触下方地面后，启动 0.3 s 的落地窗口，每回合只启动一次。
窗口内每拍奖励为 `-2 × dt × (竖直速度² + 0.25 × (横滚角速度² + 俯仰角速度²))`。
竖直速度由机体系速度沿重力方向投影得到，水平前进速度不计入此项。
上平台、空中不会触发；触发后再次弹跳仍继续计入，回合提前结束时窗口随之结束。
该项衡量触地后的振荡，不直接测量撞击瞬间的峰值冲击力，也不持续发放静止正奖励。

任务专用奖励函数集中在 `rewards.py`。任务专用 buffer 集中在 `buffers.py`，
每个任务各有一个类，按固定参数表、回合状态分组，并在同一类中实现 `create()` 和 `reset()`。
目前 `LocomotionBuffers` 保存跟踪误差，`StairsDownBuffers` 保存落地触发状态、剩余步数和窗口开关，
`StairsUpBuffers` 保存支撑/站立计时、首次触台腿长、缩腿最高进度、台下接地/离地、双轮首次落台及跌台状态。
`env.py` 按相同任务顺序调用创建、更新和重置；部分重置只清空选中环境，不清空其他回合或固定参数表。

到终点且双轮已越过最后障碍、落地接触正常、无机身接触、姿态合格时，结束并发成功奖励；
`stairs_up` 还需完成上述 0.20 s 稳定站立窗口。
小高度上台阶需到达高平台上的终点后才会成功。停滞或驶出任务走廊按失败处理；最大时长按截断处理。
失败优先于成功，成功不发 death，也不作为 timeout bootstrap。
同一拍既超时又失败时同样不进行 timeout bootstrap。

TensorBoard 的 `task/<name>/...` 分别记录环境占比、当前难度（`difficulty`）、课程成功率窗口、
实际回合成功/失败、时长、通关进度，以及 locomotion 三项跟踪误差。

## 单任务评估与测试

```bash
# 默认评估所选任务的最终难度，可通过 --difficulty 0 查看初始等级（--stage 为兼容别名）。
.venv/bin/python -m experiments.genesis.wheel_leg_infantry.tasks.multi_terrain.eval \
  -e multi_terrain_v1 --task square_wave

.venv/bin/python -m experiments.genesis.wheel_leg_infantry.tasks.multi_terrain.eval \
  -e multi_terrain_v1 --task platform_ridge_down --difficulty 0

# 自动命令、自动终点 reset，打印成功数；无需 GUI。
.venv/bin/python -m experiments.genesis.wheel_leg_infantry.tasks.multi_terrain.eval \
  -e multi_terrain_v1 --task slope_down --headless --steps 3000

# 全部回合采用间断通行：障碍前停车、通行、障碍后停车，再重新起步。
.venv/bin/python -m experiments.genesis.wheel_leg_infantry.tasks.multi_terrain.eval \
  -e multi_terrain_v1 --task stairs_up --intermittent

# 全部回合采用普通通行；不传开关则沿用存档中该难度的间断回合比例。
.venv/bin/python -m experiments.genesis.wheel_leg_infantry.tasks.multi_terrain.eval \
  -e multi_terrain_v1 --task stairs_up --no-intermittent

# 无需 GPU 的几何、配置、课程、奖励、命令及重置测试。
.venv/bin/python -m unittest tests.genesis.test_multi_terrain -v

# 可选：真实 Genesis CPU 构建、步进和状态恢复测试。
NGXY_MULTI_TERRAIN_SMOKE=1 .venv/bin/python -m unittest \
  tests.genesis.test_multi_terrain.SimulationTests -v
```

显式开关优先于 `curriculum.overrides` 中的间断比例。旧存档没有 `intermittent` 配置时，
`--intermittent` 使用当前对应地形的停车参数；已有停车参数保持存档值。
`locomotion` 不支持 `--intermittent`。评估输出总成功数、普通/间断各自成功数，
间断回合额外输出两次停稳是否完成、停车失败和最大停稳漂移距离。

课程和奖励是初始训练配置，各地形最终等级的成功率需要训练后验证。
