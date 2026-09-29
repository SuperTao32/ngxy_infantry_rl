# 域随机化

`DomainRandomizationManager` 负责采样和应用动力学参数，`LocomotionEnv`
负责在物理 reset 时调用；外力推扰由 `core/push.py` 按物理控制步调度，
在 `scene.step()` 前施加。`JumpEnv` 继承相同路径，任务交接不重新采样或重置计时。

## 配置与启动

统一入口是各任务的 `env_cfg["randomization"]`，包含独立的 `dynamics` 和 `sensors`。
以下为 `env_cfg["randomization"]["dynamics"]` 的内容（推扰暂保留在 `dynamics.push`）：

```python
"dynamics": {
    "enabled": True,
    "strength": 1.0,  # 无课程覆盖时的强度，范围 0–1
    "push": {
        "enabled": True,
        "targets": ["base", "left_wheel", "right_wheel"],
        "interval_s_range": [3.0, 8.0],
        "duration_s": 0.02,
        "force_x_range": [-30.0, 30.0],  # N，世界坐标系
        "force_y_range": [-30.0, 30.0],
        "force_z_range": [-30.0, 30.0],  # 正值向上，负值向下
    },
    "friction": {"enabled": True, "ratio_range": [0.8, 1.2]},
    "base_mass": {"enabled": True, "added_mass_range": [-1.0, 1.0]},  # kg
    "com_displacement": {"enabled": True, "displacement_range": [-0.01, 0.01]},  # m
    "motor_strength": {"enabled": True, "ratio_range": [0.9, 1.1]},
    "motor_offset": {"enabled": True, "offset_range": [-0.02, 0.02]},  # rad
    "passive_joints": {
        "enabled": True,
        "frictionloss_range": [0.005, 0.015],  # N·m，绝对值
        "damping_range": [0.005, 0.015],  # N·m·s/rad，绝对值
        "overrides": {
            # 可为每个被动铰链单独设置；缺少的字段沿用上面的公共范围。
            "left_front2_joint": {
                "frictionloss_range": [0.01, 0.03],
                "damping_range": [0.02, 0.02],  # 固定值，不随 reset 变化
            },
        },
    },
    "gas_spring": {
        "enabled": True,
        "preload_force_range": [0.9, 1.1],
        "stiffness_range": [0.9, 1.1],
        "damping_range": [0.9, 1.1],
    },
    "motor_gains": {
        "enabled": True,
        "joint_kp_enabled": True,
        "joint_kd_enabled": True,
        "wheel_kd_enabled": True,
        "joint_kp_range": [0.9, 1.1],
        "joint_kd_range": [0.9, 1.1],
        "wheel_kd_range": [0.9, 1.1],
    },
}
```

所有范围均匀采样；推扰、质量、质心、零点和被动关节参数使用标注的物理单位，其余为倍率。
上面的被动关节 override 仅演示单独配置，默认 `overrides={}`。
新 locomotion 配置默认启用全部项目，但课程从强度 0 开始逐渐增加；
旧配置缺少动力学随机化字段时关闭；旧 `env_cfg["domain_rand"]` 会自动迁移到新入口。
已保存的配置若缺少新增的推扰、质量、质心、强度、零点、气弹簧或被动关节组，该组默认关闭，
避免续训时改变原实验分布；需要启用时在配置中显式添加对应组。
jump 的动力学配置独立设置，暂时默认关闭，
以后将对应配置中的 `default_domain_rand_cfg(enabled=False)` 改为 `True` 即可。
这些范围是温和的起始值，尚未经过训练效果验证。

```bash
# 按当前配置训练（默认开启）
python -m experiments.genesis.wheel_leg_infantry.tasks.locomotion.train
# 选择另一套训练配置（包含动力学随机化、地形、观测、奖励和课程）
python -m experiments.genesis.wheel_leg_infantry.tasks.locomotion.train --config config_rand
# 续训时使用所选配置文件；不加 --resume-config current 则沿用已保存的配置
python -m experiments.genesis.wheel_leg_infantry.tasks.locomotion.train --resume latest --resume-config current --config config_rand
# 只加载模型权重，用 config_rand.py 从第 0 轮重新训练（替换为实际 checkpoint 路径）
python -m experiments.genesis.wheel_leg_infantry.tasks.locomotion.train --config config_rand --load-weights /path/to/model_6000.pt
# 普通评估默认关闭动力学域随机化；启用时采用已保存的范围
python -m experiments.genesis.wheel_leg_infantry.tasks.locomotion.eval --domain-rand
```

locomotion 训练通过 `--config config`（默认，`config.py`）或 `--config config_rand`
（`config_rand.py`）选择配置，域随机化开关和地形在所选文件中设置。
locomotion 和 jump 的训练、评估均支持 `--domain-rand` / `--no-domain-rand`，
以及独立的 `--sensor-noise` / `--no-sensor-noise`。
训练默认沿用配置；评估默认关闭动力学随机化，传感器噪声沿用保存的配置。
命令行覆盖会在保存新运行配置之前应用。固定动作延迟仍由 `simulate_action_latency` 控制。

`--load-weights PATH` 加载 actor/critic 权重（包含模型中保存的观测归一化状态），
不加载源运行配置、优化器、学习率或迭代数。训练和课程从所选配置的第 0 轮开始，
`--max_iterations` 为本次新训练的总轮数，结果保存到新版本目录。
该选项不能与 `--resume`、`--checkpoint` 或 `--resume-config current` 混用；
网络结构、观测和动作定义应与源权重兼容，权重加载使用严格形状检查。

## 传感器噪声

默认值定义在 `core/sensor_noise.py::default_sensor_noise_cfg()`：

```python
sensors = env_cfg["randomization"]["sensors"]
sensors["enabled"] = True
sensors["strength"] = 1.0
sensors["joint_vel"].update(std=0.1, bias_range=[-0.02, 0.02])  # rad/s
sensors["wheel_vel"].update(std=0.2, bias_range=[0.0, 0.0])  # rad/s
sensors["joint_pos"].update(enabled=True, std=0.001, bias_range=[0.0, 0.0])  # rad
sensors["imu"]["acc_noise"] = 0.05  # m/s²
sensors["imu"]["gyro_noise"] = 0.003  # rad/s
```

`config.py` 和 `config_rand.py` 均开启传感器总开关；普通课程前三阶段强度为 0，后两阶段为 0.4、0.6；随机化课程为 0.6、0.8、1.0。
关节位置噪声默认关闭。jump 继承来源 locomotion 的传感器配置，并深拷贝以便独立调参。
默认标准差是待实机标定的起点；应依据固件滤波后实际送入策略的反馈数据调整。

编码器通道各有 `enabled`、高斯白噪声 `std` 和均匀分布 `bias_range`。
每次物理 reset 为选中的环境、各个关节/轮子独立采样偏置，回合内保持；白噪声每控制步更新一次。
同一拍的 actor、critic 观测前缀、速度估计器共享测量缓存，重复组装观测不重新加噪。
局部 reset 只更新对应环境；locomotion → jump 交接保留测量和偏置状态。

`strength` 范围为 `[0, 1]`，只控制传感器误差幅度，与动力学课程独立。
课程通过独立的 `targets.sensor_noise` 覆盖幅度，省略时继承上一阶段：

```python
"targets": {
    "domain_rand": {"strength": 0.4},
    "sensor_noise": {"strength": 0.4},
}
```

课程不会覆盖 `enabled`，`--no-sensor-noise` 始终有效；没有此课程目标的旧配置保持原强度。
幅度为 0 时关闭白噪声、偏置和随机游走增量；`enabled=False` 还关闭固定延迟/抖动。
编码器关闭时不消耗额外随机数。阶段切换后下一次测量使用新幅度；已有编码器偏置按比例缩放，
从 0 开启时初始化偏置，不重置机器人物理状态。IMU 使用 Genesis 公开 setter，仍只加噪一次。
IMU 随机游走更新的是后续增量幅度，历史累积状态沿用 Genesis 的生命周期。

IMU 白噪声、固定偏置和随机游走幅度由此缩放后交给 Genesis；
`delay`/`jitter` 使用秒，启用时保留配置值，不随幅度缩放；`jitter <= delay`。
随机游走沿用 Genesis 每传感器更新的增量定义，改变更新频率时需要重新标定。
IMU 偏置保留 Genesis 的固定参数语义，不是编码器的每回合随机偏置。
IMU 在 reset 首帧仍沿用原来的初始化读数，从后续仿真步开始读取 Genesis 传感器。

安装信息 `link_name`、`pos_offset` 仍在 `obs_cfg["imu"]`，观测缩放仍在 `obs_scales`。
旧存档中的 IMU 误差字段会迁移，保留原值，不为旧实验自动开启编码器噪声。
如果同时存在新旧误差配置，以新 `randomization` 中对应组为准，保存时移除旧入口。
新旧配置不会在 IMU 上重复加噪。

测量误差在物理单位下施加，然后才做观测缩放。
`joint_pos/joint_vel/wheel_vel` 继续保留真值供控制、奖励、终止判断和 PD 诊断使用；
`measured_*` 供观测及轮速估计使用。腿长、腿角从带噪关节位置重新计算。
观测顺序和维度不变，critic 仍继承 actor 观测并追加仿真特权信息。
本次未加入姿态估计误差，`projected_gravity` 仍来自仿真姿态；
编码器滤波、随机游走和延迟也尚未建模。

```bash
# 纯净评估
python -m experiments.genesis.wheel_leg_infantry.tasks.locomotion.eval --no-domain-rand --no-sensor-noise
# 只测试传感器误差（使用 checkpoint 保存的幅度）
python -m experiments.genesis.wheel_leg_infantry.tasks.locomotion.eval --no-domain-rand --sensor-noise
# 两组同时开启
python -m experiments.genesis.wheel_leg_infantry.tasks.locomotion.eval --domain-rand --sensor-noise
# config_rand 训练时只保留动力学随机化
python -m experiments.genesis.wheel_leg_infantry.tasks.locomotion.train --config config_rand --no-sensor-noise
```

注意：旧存档若保存的幅度为零，`--sensor-noise` 只打开开关，不自动替换这些幅度。
动力学课程目标名仍为 `domain_rand`，仅影响 `randomization.dynamics`。

## 随机化强度课程

在课程阶段的 `targets` 中加入：

```python
{
    "name": "gentle_motion",
    "start_iteration": 1000,
    "targets": {
        "domain_rand": {"strength": 0.2},
        # 其他已有课程目标继续放在这里
    },
}
```

`strength` 必须在 `[0, 1]` 内；0 表示域随机化使用标称参数且不触发推扰，
1 使用完整配置范围。设为 0 不会关闭命令、初始姿态或地形等独立课程，
也不改变 IMU 噪声和原有固定动作延迟。

目前 locomotion 的默认阶段为：

| PPO iteration 起点 | 阶段 | strength |
| --- | --- | --- |
| 0 | stand | 0.0 |
| 1000 | gentle_motion | 0.2 |
| 2000 | locomotion | 0.4 |
| 4000 | locomotion2 | 0.6 |
| 6000 | locomotion3 | 0.8 |
| 8000 | full_range | 1.0 |

强度收缩规则是 `标称值 + strength × (完整范围采样值 - 标称值)`：

- 摩擦、电机增益/强度和气弹簧倍率的标称值为 1。
  例如 `[0.8, 1.2]` 在强度 0.5 下变成 `[0.9, 1.1]`。
- 质量增量、质心和电机零点偏移的标称值为 0。
  例如质量 ±1 kg 在强度 0.5 下为 ±0.5 kg。
- 被动关节摩擦/阻尼围绕 scene.build 后读取的模型值插值，
  而非围绕 0；`overrides` 中的固定值也参与此插值。
- 推扰三轴分力乘以该环境的回合强度，强度 0 不触发事件；
  触发间隔和持续时间不随强度改变。

课程只更新**下一次物理 reset 的目标强度**，不会立即重置环境或覆盖正在运行的
动力学参数。局部 reset 只让选中的环境采用新强度；长回合可能在旧强度下继续运行。
推扰同样保持本回合强度，正在施加的脉冲不会因课程切换突然改变。
`begin_jump_rollout()` 不属于物理 reset，因此保留当前强度和推扰计时。

课程的强度不会隐式打开被配置关闭的项目，也不会修改保存的完整随机化范围。
续训时已有 `set_training_iteration()` 会恢复对应课程阶段后 reset，采用恢复点强度。
旧课程不含此目标时沿用 `env_cfg["domain_rand"]["strength"]`，缺省为 1；
jump 也可在自己的阶段 `targets` 中使用相同字段，其默认域随机化仍关闭。
评估不重跑课程，`--domain-rand` 使用环境配置中的强度（默认 1）。

`env.domain_rand.diagnostics(env_idx)` 的 `domain_rand_strength` 是下次 reset 的目标，
`domain_rand_episode_strength` 是该环境当前回合实际使用的强度。

## 采样与交接规则

- 外力推扰：每环境独立计时，每次从 `targets` 中均匀选一个位置，分别从
  `force_x_range`、`force_y_range`、`force_z_range` 独立均匀采样三个有符号分量，
  在整个脉冲内保持不变。施力点为目标 link 的质心，力使用世界坐标，
  Z 正值向上、负值向下。轮子受到的力会通过接触和连杆传递到机身。
  范围是各轴分力的边界，不是合力大小的边界；`[value, value]` 固定该轴，
  `[0, 0]` 关闭该轴。例如只需要水平推扰时设置 `force_z_range=[0, 0]`。
  原 `force_xy_norm_range` 已移除；旧的推扰配置需删掉该字段并显式填写三轴范围，
  新的分量独立采样方式与旧的“水平幅值+角度”分布不同，不进行隐式转换。
  `base` 按 `base_link_name` 定位，左右轮按 `wheel_link_names` 的左右顺序定位，
  solver 接口使用解析后的全局 link 索引。
  只推轮子可设 `targets=["left_wheel", "right_wheel"]`，只推左轮可设 `["left_wheel"]`。
  同一环境的同次事件只选一个位置，不会同时给三个位置各施加一份力。
  `interval_s_range` 是两次脉冲起点的间隔，最小值必须不小于 `duration_s`。
  时间向上量化为控制步数；当前 dt=0.02 s，0.02 s 脉冲持续一拍，0.1 s 持续五拍。
  reset 后先等待一个采样间隔；每个待施力控制步调用一次外力接口，Genesis 在这个
  控制步的全部物理子步完成后自动清除外力，多拍脉冲需逐拍重新提交同一个力。
  冲量约为力乘以实际持续时间。完整强度下默认各轴 ±30 N、3–8 s；这些范围尚未验证训练效果。
  物理 reset 清除该环境的脉冲状态并重采样计时；PPO 时钟、jump 阶段切换不影响计时。
  jump 开启后预热和跳跃阶段均可能触发，不按腾空/接触状态过滤。
- 摩擦：每个环境采样一个倍率，应用于机器人和全部地形实体（包括箱体楼梯和
  jump 评估台阶）。Genesis 取接触双方摩擦系数的最大值，因此两侧同步缩放。
  极低系数仍受 Genesis 接触摩擦最小值约束。
- 基座质量：每环境采样一个质量增量，使用 Genesis 的 `set_mass_shift`。
  根据 `base_link_name` 查找 link，通过 entity 接口换算 solver 索引，不写死编号。
  标称基座质量加最小增量必须大于零。重复 reset 设置新的增量，不累加。
- 质心：每环境、每轴独立采样位移，使用 `set_COM_shift`，仅影响基座。
  质量和质心均使用 Genesis 的 shift 机制，不额外随机化惯量张量。
- 增益：每个环境、每个电机独立采样。腿部 Kp、腿部 Kd、轮子 Kd 可分别开关。
- 电机强度：每环境一个倍率，所有受控电机共享；同时乘到 PD 增益和力矩上限，
  等价于 `strength * clip(PD, -nominal_limit, nominal_limit)`。
  增益随机化和强度随机化同时启用时，倍率相乘；不缩放动作幅度，也不缩放气弹簧力。
  保留 Genesis 每个物理子步的闭环控制。增益或强度随机化开启时，
  场景使用 `batch_dofs_info=True`。
- 被动关节：自动选择未在腿电机、轮电机和气弹簧施力名单中的 revolute 关节。
  当前模型是左右侧 `front2_joint`、`front3_joint`、`front4_joint`、`rear2_joint`、
  `spring1_joint`，共 10 个铰链。XML 的 `class="motor"` 是默认参数类，
  不代表这些关节有电机驱动。默认模型参数均为 frictionloss=0.01、damping=0.01。
  `frictionloss` 是关节库仑摩擦力矩，`damping` 是被动黏性阻尼（力矩为 `-damping * qdot`），
  分别调用 `set_dofs_frictionloss` 和 `set_dofs_damping`；与接触摩擦、电机 Kd 分开。
  每环境、每关节、每项独立采样绝对值，因此标称为零的参数也可改为非零。
  组开启时启用 `batch_dofs_info=True`。设置范围 `[value, value]` 可固定参数，
  `[0, 0]` 可取消该摩擦/阻尼；范围必须非负。组关闭则保留模型参数。
  `overrides` 的名称必须属于上述被动关节，拼错或填入主动关节会报错。
- 电机零点：腿部每电机独立采样 rad 偏移，在延迟后的动作转成位置目标之后加上，
  然后经过原有几何限位。正偏移表示增大位置目标。轮子采用速度控制，角度零点
  不影响其速度目标；不会把 rad 偏移加到 rad/s 上。初始关节姿态和真实观测不修改。
- 气弹簧：每环境、每根弹簧、每项参数独立采样，顺序与 `spring_names` 一致。
  三个范围均为相对于 `gas_spring_preload_force`、`gas_spring_stiffness`、
  `gas_spring_damping` 标称值的倍率。实际施力使用
  `F = F0 + k * clamp(max_compression - position, 0, max_compression) - c * velocity`。
  默认标称值 420 N、1400 N/m、50 N·s/m 对应采样范围 378–462 N、1260–1540 N/m、
  45–55 N·s/m。可将某一倍率范围设为 `[1.0, 1.0]` 固定该参数，或用组内
  `enabled=False` 关闭整组。`gas_spring_max_compression` 是机械行程，保持原值。
  原有施力时机不变：每个控制步读取弹簧状态、更新施力，物理子步内保持该施力。
- reset：仅改变被选中的环境；动力学参数倍率在整个物理回合内保持不变，外力事件独立计时。
- jump 交接：先保存 locomotion 的原始标称参数；切换时计算
  `当前模式标称参数 × 当前环境增益倍率 × 电机强度倍率`，不会覆盖倍率或累乘。
  `begin_jump_rollout()` 保留回合随机化强度、质量、质心、摩擦、控制偏移、气弹簧、被动关节参数、推扰计时及已有动作历史。
- 观测与奖励：保持原有维度和真值路径。新增参数不自动进入 actor/critic。

`env.domain_rand.diagnostics(env_idx)` 返回摩擦倍率、质量增量、质心位移、
电机强度、零点偏移、各电机增益倍率，以及气弹簧的实际参数和倍率；
被动关节随机化开启时还返回 `passive_joint_names`、`passive_joint_damping`、
`passive_joint_frictionloss`，张量顺序与关节名一致。
`push_target` 和 `push_force_world_N` 表示最近一个控制步的目标与外力；
无施力时分别为 `None` 和零向量。`push_steps_remaining` 是当前脉冲还需施力的拍数，
`push_steps_until_next` 是下次脉冲的倒计时。
`env.get_pd_diagnostics(env_idx)` 返回该环境实际使用的增益。
`env.joint_kp` 等运行时电机参数现在统一为 `[num_envs, num_motors]`。
`env.gas_spring_preload_force/stiffness/damping` 为 `[num_envs, num_springs]`，
与 manager 的实际参数共用 buffer；任务配置中的标称值保持不变。

后续动力学参数可扩展 manager 的 `bind/reset`。传感器误差统一由
`randomization.sensors` 配置，IMU 的执行仍交给 Genesis，避免重复加噪。


覆盖关闭兼容、随机种子、范围校验、局部 reset 隔离、接触双方摩擦更新、
电机模式切换及重复 reset 不累乘等行为。可选仿真测试检查 solver 中的
实际增益、质量和质心偏移、限幅前后电机力矩、位置目标偏移、气弹簧实际施力、
被动关节参数写入及主动关节隔离、轮子外力施加和自动清除、步进后观测有限性、
旧配置兼容，以及 jump 交接前后随机参数和动作历史连续性。
课程测试覆盖零强度、各类参数的插值、局部 reset 延迟生效及按迭代数恢复强度。
