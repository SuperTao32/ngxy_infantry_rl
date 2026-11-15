# Genesis 环境框架

`env.py` 是唯一环境，保留 locomotion 的场景、控制、物理状态、传感器、
观测、命令采样、终止、奖励和课程。不再按任务分配并行环境，`src` 不依赖 `src_old`。

## 组件职责

| 所有者 | 接口 | 自有状态 |
| --- | --- | --- |
| env | `reset()`、`step(actions)`、`_resample_commands(mask)` | 物理状态、locomotion 奖励与回合统计 |
| recovery | `sample_reset_pose(...)`、`reset(env, mask)`、`update(env)` | 姿态模板/采样编号、`RecoveryBuffers`、`RecoveryRewards` |
| jump | `on_resample(env, mask, previous_vx)`、`update(env)`、`apply_commands(commands)` | `JumpBuffers`、`JumpRewards` |

`configuration.py` 提供环境的 locomotion 命令/重置配置方法，`rewards.py`
提供 locomotion 奖励项。`reward_buffer.py` 只复用结算和累计逻辑；env、recovery、
jump 各自创建实例，不共享奖励 buffer 或回合累计。

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

新配置标记为 `wheel_leg_infantry_v2`，checkpoint 的环境状态保存在
`infos.infantry_env`。旧配置的任务调度不在新入口恢复；可用 `--load-weight`
将结构兼容的 actor/critic 权重载入新训练。

`--resume-config saved` 恢复 checkpoint 的奖励权重，再应用对应课程；
`--resume-config current` 使用当前配置和课程的权重，不被 checkpoint 中的旧权重覆盖。
当前 ToF 已关闭，不创建 Raycaster；需要使用时在观测配置中开启 `tof.enabled`。

每轮 PPO 输出 recovery 的尝试/完成次数和成功率，以及按模式划分的 jump 成功率、
失败率和失效率。成功率分母只包含本轮结束的尝试，仍在执行的尝试不计失败。
`termination/*` 分别记录倾倒、机身/腿部接触、求解器错误、恢复超时、时间及边界截断；
物理失败原因可能同时命中。各组件独立持有计数，不影响奖励 buffer 或回合累计。

训练前验证（在项目根目录运行，CPU 短训练只临时修改内存中的验证配置）：

```bash
.venv/bin/python -m unittest discover -s tests/genesis -p test_training_readiness.py -v
.venv/bin/python -m tests.genesis.smoke_training
```

当前 recovery/jump 提供接入框架、完整姿态模板复位及起身/平地跳奖励适配。
旧版动态释放、难度筛选、台阶助跑和地形分配等专用逻辑保留在 `src_old`。
