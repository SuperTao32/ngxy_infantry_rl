# Genesis 环境框架

`env.py` 是唯一环境，保留 locomotion 的场景、控制、物理状态、传感器、
观测、命令采样、终止、奖励和课程。不再按任务分配并行环境，`src` 不依赖 `src_old`。

## 组件职责

| 所有者 | 接口 | 自有状态 |
| --- | --- | --- |
| env | `reset()`、`step(actions)`、`_resample_commands(mask)` | 物理状态、locomotion 奖励与回合统计 |
| recovery | `sample_reset_pose(...)`、`reset(env, mask)`、`update(env)` | 姿态模板/采样编号、`RecoveryBuffers`、`RecoveryRewards` |
| jump | `on_resample(env, mask, previous_vx)`、`update(env)`、`apply_commands(commands)` | `JumpBuffers`、`JumpRewards` |
| terrain_cross | `reset(env, mask)`、`update(env)`、`compute_rewards()` | 各地形的 `TerrainCrossBuffers`、`TerrainCrossRewards` 与分组指标 |

`configuration.py` 提供环境的 locomotion 命令/重置配置方法，`rewards.py`
提供 locomotion 奖励项。`reward_buffer.py` 只复用结算和累计逻辑；env、recovery、
jump、terrain_cross 各自创建实例，不共享奖励 buffer 或回合累计。

### Terrain cross 越障

`tasks/terrain_cross/` 与 jump/recovery 一样分为 `config.py`、`buffers.py`、`rewards.py`、`task.py`。
`env_cfg.terrain_cross` 配置越障判定和各类地形的奖励；平地继续执行普通 locomotion。
`env.terrain_cross.buffers["stairs"]` 等字典项分别保存该类地形的逐环境状态，
包括分配、开始/活动、起点/终点、最远归一化进度、连续支撑、耗时和最终结果。
同类地形共用复位、命令和奖励配置，各地块保留自己的几何终点；结果同时按地形及“地形＋等级”统计。

只有平地按 recovery 概率选择姿态池，并使用普通命令与 jump。
特殊地形使用默认关节姿态，roll/pitch 和初速度归零，按各地形的 `reset` 配置出生：

```python
terrain = configs["env_cfg"]["terrain_cross"]["terrains"]["stairs"]
terrain["reset"] = {
    "distance_range_m": [0.8, 1.2],  # 基座到障碍起点沿 x 的距离，m。
    "yaw_range_deg": [-5.0, 5.0],    # 相对 +x 朝向，度。
    "base_height_m": 0.22,          # 相对出生点地面的基座高度。
}
terrain["commands"] = {
    "lin_vel_range": [0.5, 1.0],
    "ang_vel_range": [0.0, 0.0],
    "base_height_range": [0.22, 0.28],
}
```

出生点 `x = 地块中心 x + 障碍起点 x - 采样距离`，y 位于地块中线；默认姿态叠加采样 yaw。
这些参数覆盖普通课程的复位扰动；即使 recovery 概率为 1，特殊地形仍保持默认姿态。
所有特殊地形只在 reset 时采样一次独立命令，整回合跳过周期重采样，优先于普通课程和 jump，
键盘命令或即时跳跃也不能覆盖。成功后仍保持地形命令，直到物理回合结束。
特殊地形仅在物理 reset 时按 `commands.standing_probability` 抽取停车回合；
默认 `None` 表示跟随全局 `standing_probability` 课程，也可以为某类地形单独配置概率。
ToF 启用时，只有停车回合按前向射线安装位置、出生高度与 yaw 自动计算出生距离：
光心在边缘前、假想平地落点在边缘后，两侧保留 `reset.standing_tof_margin_m`（默认 2 cm）。
默认安装和出生高度下基座距边缘约 0.30–0.35 m，普通越障仍使用 `distance_range_m` 保留助跑距离。
如果安装、量程或 yaw 使该距离区间不存在，会报错；关闭 ToF 时沿用普通出生范围。
停车回合线速度和偏航速度命令为 0，高度命令只采样一次，
整个回合跳过周期命令重采样；到下次物理 reset 才重新决定停车或越障。
停车回合只使用基础奖励，不计入越障尝试、失败或难度课程；回合日志记录 `terrain_cross/<地形>/parking`。
课程可通过 `targets.terrain_cross.terrains.<地形名>.reset/commands` 覆盖这些范围，
均在下次 reset 生效。各地形共享现有 locomotion 基础奖励。
速度跟踪门控由各地形的 `tracking_gate` 独立配置：当前默认高度误差 5 cm 内全开、
8 cm 时降至保底；倾角 5° 内全开、30° 时降至保底，`floor=0.30`。
平地仍使用全局 `tracking_gate` 课程，收紧平地容差不会影响特殊地形；
特殊地形通关后继续使用该地形门控，直到物理 reset。
课程可通过 `targets.terrain_cross.terrains.<地形名>.tracking_gate` 局部覆盖，下一控制步生效。
双轮都越过几何终点加 `finish_distance_m`（默认 0.3 m），在所属地块内连续双轮支撑
`success_hold_s`（默认 0.2 s），机身无接触且倾角不超过 `max_tilt_deg`（默认 30°）即完成。
梯形波的几何终点为经过一次下坡、低平台、上坡后的下一个高平台起点。
成功后继续当前物理回合；未完成就终止或 timeout 则计失败。每次越障只结算一次结果。

各地形奖励叠加在当前 locomotion/jump 的奖励上，默认权重为：

```python
configs["env_cfg"]["terrain_cross"]["terrains"]["stairs"]["reward_scales"] = {
    "progress": 5.0,   # 只奖励新增最远进度，整段累计最多 5；往返不能重复刷分。
    "success": 50.0,   # 完成时结算一次。
    "failure": -10.0,  # 未完成就结束时结算一次。
    "task_lateral": -1.0,  # 相对出生航向参考线的横向距离平方。
    "task_heading": -0.5,  # 航向误差 1 - cos(yaw_error)。
}
```

前三项按增量/事件结算，不乘 dt；横向与航向惩罚仅在越障活动期间按 dt 积分。
参考线在 reset 时按出生位置和 yaw 固定，部分 reset 不影响其他环境。
速度跟踪、平衡、动作平滑和碰撞等继续使用现有 locomotion 奖励，避免重复配置计分。
可在 `TerrainCrossRewards` 中新增特殊奖励函数，
或在某类地形的 `reward_scales` 中显式配置 env 的共享奖励项。
课程可通过 `targets.terrain_cross.terrains.<地形名>.reward_scales` 覆盖对应权重。
每轮 PPO 输出 `terrain_cross/<地形名>/attempts`、`completed`、`success_rate`、`failure_rate` 等，
回合日志另记录该地形的奖励累计、最大进度和耗时；成功率只统计已结束的尝试。
checkpoint 保存各地形奖励权重、当前难度和未满判定窗口的累计样本，加载后物理回合和运行 buffer 按统一 reset 流程重建。

### Reset 恢复

1. env 按当前课程的 reset 范围采样位置、姿态扰动和速度，再按 recovery 概率决定是否使用 JSON 模板。
2. env 将模板与扰动组合、适配当前地形，校正穿地后写入初始物理状态和传感器状态。
3. env 清空 jump，调用 `recovery.reset(env, mask)`。
4. 命中姿态池的环境进入 recovery，覆盖命令为零速度和起身高度；未命中时直接进入 locomotion。
5. 站稳完成的这一步结算 recovery 奖励；下一步恢复 locomotion 命令，不重置物理状态或策略记忆。

恢复期间允许倾倒与机身接触，恢复超时或求解器错误仍终止回合。
recovery 的 `config.py` 中，`RESET_POSE_FILES` 指定 `reset_pos/` 内的文件清单。
创建配置时将 JSON 读入 `recovery.reset_pos.pose_pool`，存档后 reset 只采样数值模板，
不再读取文件。每个待 reset 的环境独立进行概率采样，命中后再从池中均匀选取模板。
未命中时使用 env 默认朝向、关节位置及原始 `reset_ranges`，采样编号为 `-1`，也不进行快照穿地校正。
只更新本次 reset 环境的采样编号。

`recovery.reset_pos.probability` 默认 `0`，课程通过 `targets.recovery.probability` 调整。
`0` 全部默认复位并直接进入 locomotion，`1` 全部从姿态池复位并执行恢复。

命中姿态池时，模板与 `reset_ranges` 按以下规则组合：

| 数据 | 组合规则 |
| --- | --- |
| base 位置 | `JSON 位置 + base_init_pos_range 采样值 - reference_base_pos`，再叠加地形中心/高度 |
| base 朝向 | JSON 四元数上叠加 `base_init_rpy_offset_range_deg` 的局部旋转 |
| base 速度 | 从 `base_init_lin_vel_range` / `base_init_ang_vel_range` 采样 |
| 关节位置 | 完整还原 JSON 中的电机、被动关节、轮子和弹簧位置 |

默认 `reference_base_pos=[0, 0, 0.22]`。例如 JSON 的 z 为 `0.1456`：
范围 `[0.22, 0.22]` 保留该高度；范围 `[0.22, 0.24]` 在该高度上增加 `0~0.02 m`。
姿态范围全零时保留 JSON 朝向。位置和姿态扰动后如碰撞体穿地，只向上修正高度，
`clearance_m=None` 可关闭此校正。

课程切换后，下次 reset 自动使用最新概率和范围。概率为 0 或将 `reset_pos` 设为 `None` 时，
初始姿态继续由 env 的 `base_init_*`、`default_joint_pos` 和 `reset_joint_pos` 决定。

### Jump 命令输入

1. env 在控制步内判断命令重采样时间，先保存上一条命令的 `vx`，再采样新的 locomotion 命令。
2. `jump.on_resample` 消费请求或按概率决定是否跳跃。命中时缓存本次新命令，沿用上一条命令的 `vx`，将 yaw 设为 0、高度列换成 jump 轨迹；未命中时直接使用新命令。
3. jump 自行推进计时、模式 one-hot、相位编码、高度参考和接触事件，期间保持该 `vx` 目标。
4. jump 输入窗口结束后编码归零，整个跳跃周期结束后应用本次缓存的新 locomotion 命令，不额外采样。

`jump.request(mask, mode=0)` 只排队，下一次命令重采样时触发。
`jump.trigger(mask, mode=0)` 立即开始跳跃并缓存当前 locomotion 命令，恢复或跳跃中的行忽略请求。
reset 不触发 jump，恢复或跳跃执行中不接受新的命令采样覆盖。
常规命令重采样仍按回合内的固定周期触发。若 jump 结束恰好遇到采样边界，本拍先应用缓存命令。

jump buffer 保留起跳/落地历史、连续与累计腾空计数、奖励峰值及追回状态。
`rebound_seen` 记录首次落地后再次双轮离地，直到本次 jump 重置，供二次起跳相关奖励读取。
轮心位置、当前成功结果和冲击速度按需计算；模式与相位观测直接由当前状态生成。
跳跃命令只缓存旧 `vx`，同时保留完整的新 locomotion 命令供 jump 结束后使用。

每个控制步只结算实际执行阶段的奖励：recovery、jump 或 locomotion。
组件完成的最后一步仍归该组件，奖励先结算，再发布下一步命令。
全量与局部 reset 均按 mask 清理各自的 buffer 和奖励累计。

### 旧奖励初步适配

recovery 沿用旧版起身奖励的形状和默认权重，保留新框架已有的六个简称：
`height`、`balance`、`leg_angle`、`success`、`speed`、`failure`。
补回轮底离地距离、翻倒时的轮支撑力、速度跟踪、腿部对称、关节速度、
动作平滑、轮子离地和机身接触奖励。完成起身后的静止跟踪由 locomotion 负责。
当前没有旧版动态释放标记，因此不接入“动态释放至首次落地期间免除离地惩罚”的分支。

jump 先迁入旧版平地跳的奖励公式、默认权重和参数，权重位于
`tasks/jump/config.py:get_reward_cfg()`：

| 阶段 | 奖励与状态 |
| --- | --- |
| 起跳 | 支撑力、向上速度、最后支撑拍的起跳速度、连续腾空资格事件 |
| 腾空 | 首次合格腾空峰值、增量高度进度、缺高/超高、离地间隙、姿态/腿角和空转轮速 |
| 落地 | 首次任意轮/机身触地前的竖直速度、三轴加速度、软着陆和落地稳定奖励 |
| 结果 | 合格腾空达到高度门槛的一次性成功、失效或窗口结束未成功时的一次性失败 |

高度以每次触发 jump 时的当地地面为固定基准，base 目标为 `0.45/0.63 m`；
轮底及 base 成功门槛沿用旧版平地跳。达到高度门槛即可记录成功，不等待落地或连续站稳；
软着陆和落地稳定奖励仍独立结算。落地后站高和二次腾空不能补成绩。
`JumpRewardState` 保存连续腾空资格与峰值；倾角超限或 env 物理终止会撤销已认可的峰值，
追回已发高度进度和成功奖励，并补齐已结算的缺高惩罚。成功奖励只撤回一次，
最终统计按跳跃窗口结束或回合终止时的有效结果计算。过倾阈值由 `jump_max_tilt_deg` 配置。

结果事件在 env 判断本拍终止后、reset 前结算。它们用于奖励，保持现有 env 终止规则；
窗口结束后仍返回 locomotion，奖励中的成功或失败不额外 reset。
台阶几何、助跑、落台判定和按地形分组的权重暂未迁入。

共享物理奖励可以复用 env 的计算函数；每个组件仍拥有独立的权重、输出和回合累计。
jump 权重支持标量或按 `JUMP_20/JUMP_40` 排列的列表。
奖励形状按物理量配置：`height_sigma`、`tilt_sigma`、`lin_vel_sigma`、`vertical_vel_sigma`、
`ang_vel_sigma` 和 `flight_leg_angle_limit_deg`。飞行/落地共用姿态尺度，yaw 跟踪/落地角速度
共用角速度尺度；sigma 是指数分母，单位与对应误差平方一致。未启用的收腿奖励及其参数已移除。
缩放沿用旧版：recovery 的成功、速度和失败事件不乘 `dt`；jump 只有 `death` 不乘 `dt`，
其余项（包括起跳、成功和失败事件）均乘 `dt`，避免迁移后事件奖励放大 100 倍。

## 配置与入口

训练默认预建各等级地形并按成功率调级。在 `config.py` 的 `env_cfg.terrain` 中配置：

```python
"terrain": {
    **default_terrain_cfg("mixture"),
    "curriculum": {
        "enabled": True,
        "min_completed": 100,
        "promote_threshold": 0.95,
        "demote_threshold": 0.1,
    },
    "tiles": [
        {"preset": "plane", "difficulty": 0, "weight": 0.5},
        {"preset": "stairs", "difficulty": 0, "weight": 0.2},
        {"preset": "stairs_up", "difficulty": 0, "weight": 0.1},
        {"preset": "slope_up", "difficulty": 0, "weight": 0.2},
    ],
}
```

地块共用 `tile_size`，沿 y 排列，行进方向保持 +x；场景启动时一次创建所有 `LEVELS`。
tile 的 `difficulty` 为初始等级。每个待 reset 环境先按权重选类型，再选择该类型的当前目标等级；
零权重类型不采样，权重无需总和为 1，增加等级数不改变类型比例。
出生位置叠加地块中心和当地地面高度；高度查询也供 reward、recovery 和 ToF 使用。
接近所属地块边界时按 timeout 复位，局部 reset 只更新选中环境。
拼盘中的 plane 使用有限 Box，单地形 `default_terrain_cfg("plane")` 保持无限 Plane。
每轮 PPO 结束时，各地形独立累计当前等级的已完成越障结果；满 100 次后，按
`env_cfg.terrain.curriculum` 中的成功率阈值升降级，其余保持不变；判定后清空该窗口，等级限制在对应 `LEVELS` 内。
尚未结束的尝试不计失败，未通关的终止/timeout 计失败。旧等级回合完成后仍按原等级记录指标，
不进入新等级的判定窗口。升降级只修改后续 reset 的出生地块，运行中的机器人和碰撞几何保持原样。
多 GPU 在调级前汇总完成/成功数，保持相同目标等级；eval 的单地形 `--difficulty` 保持固定。
TensorBoard 的 `terrain_cross/<地形>/level_<等级>/...` 记录分级结果，
`terrain_curriculum/<地形>/level` 和 `window_completed` 记录目标等级与当前窗口样本数。
设置 `curriculum.enabled=False` 可只创建指定 `difficulty` 的固定地形；自动课程每种类型配置一项即可。
需要覆盖某一块的几何参数时，在该 tile 中添加 `parameters` 字典，其优先于 `difficulty`。
难度表随配置存档；当前配置要求显式 `preset`，不再迁移 `flat` 或忽略旧字段。

总课程通过 `targets.terrain_weights` 控制类型占比，`tiles[].weight` 作为未启用总课程时的默认权重。
默认日程如下，数值为每次 reset 的目标采样概率，运行中的回合不改变地形：

| PPO 轮数 | plane | stairs | stairs_up | slope_up |
| --- | --- | --- | --- | --- |
| 0～3999 | 100% | 0 | 0 | 0 |
| 4000～5999 | 80% | 10% | 5% | 5% |
| 6000 起 | 50% | 20% | 10% | 20% |

```python
"targets": {
    "terrain_weights": {"plane": 0.8, "stairs": 0.1, "stairs_up": 0.05, "slope_up": 0.05},
}
```

首次设置 `{"plane": 1.0}` 即屏蔽所有特殊地形；后续阶段遵循课程的累计覆盖语义，
关闭已开启的类型需显式写 0。零权重地形仍预建，可在后期启用；重新启用时保留已学习的难度等级。
课程切换只影响后续 reset，当前各类环境数量会随回合结束逐步接近目标比例。
续训按保存轮次恢复阶段占比，单地形 eval 保持 `--terrain` 指定地形。
TensorBoard 的 `terrain_sampling/<地形>/probability` 记录当前目标比例。

两个组件默认概率均为 0，不使用命令行开关或单独的 `enabled` 字段。
在 `config.py` 的课程阶段 `targets` 中配置：

```python
"targets": {
    "recovery": {"probability": 0.2},  # reset 时 20% 从姿态池复位并恢复
    "jump": {"probability": 0.1},      # 命令重采样时 10% 插入跳跃
    "reset_ranges": {...},
}
```

未在后续阶段覆盖的概率继承上一阶段。设回 0 只关闭后续触发，当前恢复/跳跃继续完成。
初始化、续训和评估会应用对应轮次的课程概率，组件在启动时预先创建，无需重建场景。
`jump.request(...)` 是显式请求接口，不受自动采样概率限制。

指定姿态文件的子集：

```python
from src.genesis.tasks.recovery.config import get_reset_pos_cfg

configs["env_cfg"]["recovery"]["reset_pos"] = get_reset_pos_cfg([
    "pose_leg_bb.json", "pose_pitch90.json",
])
```

```bash
python -m src.genesis.train
python -m src.genesis.eval --exp-name wheel_leg_infantry
```

eval 的可视化窗口支持方向键控制速度/转向、PgUp/PgDn 调整高度、Space 停止，
`1/2` 选择 `JUMP_20/JUMP_40`（默认 20 cm），`J` 立即跳跃。
跳跃期间保留起跳时的前向速度，结束后恢复键盘命令；恢复或跳跃期间重复按 `J` 不排队。
平地的 `R` 从 checkpoint 保存的姿态池随机复位并执行 recovery，`Backspace` 跳过姿态池使用普通复位，
两者均清空策略记忆，且不会修改后续自动复位的课程概率。未配置姿态池时 `R` 输出提示。
完成起身后自动回到 locomotion；特殊地形两种复位都使用其默认姿态和出生配置。`Enter` 重新聚焦机器人。

新配置标记为 `wheel_leg_infantry_v2`，checkpoint 的环境状态保存在
`infos.infantry_env`。旧配置的任务调度不在新入口恢复；可用 `--load-weight`
将结构兼容的 actor/critic 权重载入新训练。

`--resume-config saved` 恢复 checkpoint 的奖励权重，再应用对应课程；
`--resume-config current` 使用当前配置和课程的权重，不被 checkpoint 中的旧权重覆盖。
当前 ToF 在场景启动时创建，actor/critic 从启动起固定保留测距通道，课程开启前输入全零。
在需要启用的课程阶段 `targets` 中配置 `"tof": {"enabled": True}`；后续阶段继承此设置，
也可设回 False 重新屏蔽。没有启用课程或没有显式开启时，ToF 输入保持零。
测距和历史按 `update_hz` 持续更新，开启后立即提供当前历史，不改变观测维度或重置回合。
默认 4 个传感器、1 帧历史，actor 为 40 维；旧 36 维模型需要适配输入维度后才能加载。

每轮 PPO 输出 recovery 的尝试/完成次数和成功率，以及按模式划分的 jump 成功率、
失败率和失效率。成功率分母只包含本轮结束的尝试，仍在执行的尝试不计失败。
`termination/*` 分别记录倾倒、机身/腿部接触、求解器错误、恢复超时、时间及边界截断；
物理失败原因可能同时命中。各组件独立持有计数，不影响奖励 buffer 或回合累计。

训练前验证（在项目根目录运行，CPU 短训练只临时修改内存中的验证配置）：

```bash
.venv/bin/python -m unittest discover -s tests/genesis -p test_training_readiness.py -v
.venv/bin/python -m unittest discover -s tests/genesis -p test_terrain_sampling.py -v
NGXY_TERRAIN_SMOKE=1 .venv/bin/python -m unittest discover -s tests/genesis -p test_terrain_sampling.py -v
.venv/bin/python -m unittest discover -s tests/genesis -p test_terrain_cross.py -v
NGXY_TERRAIN_CROSS_SMOKE=1 .venv/bin/python -m unittest discover -s tests/genesis -p test_terrain_cross.py -v
NGXY_TERRAIN_COMMANDS_SMOKE=1 .venv/bin/python -m unittest discover -s tests/genesis -p test_terrain_commands.py -v
NGXY_TERRAIN_DIFFICULTY_SMOKE=1 .venv/bin/python -m unittest discover -s tests/genesis -p test_terrain_difficulty.py -v
NGXY_TERRAIN_WEIGHTS_SMOKE=1 .venv/bin/python -m unittest discover -s tests/genesis -p test_terrain_weights.py -v
.venv/bin/python -m tests.genesis.smoke_training
```

当前 recovery/jump 提供接入框架、完整姿态模板复位及起身/平地跳奖励适配。
旧版动态释放、难度筛选、台阶助跑和地形分配等专用逻辑保留在 `src_old`。
