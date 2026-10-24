# Locomotion 训练

本目录只维护经典行走训练：速度和高度跟踪、姿态保持、静止站立及随机化课程。
多地形通关任务统一位于 [`../multi_terrain/`](../multi_terrain/README.md)。

Actor 的动作历史按整块并列排列：`actions[26:32]`、`last_actions[32:38]`，随后是 ToF 历史。
`step(a_t)` 返回的两块分别为裁剪后的 `a_t` 和 `a_(t-1)`，不再施加位置/轮速单位缩放。
历史仅在接收新动作时推进；重复读取或组装观测不推进历史，reset 将两块清零。
开启动作延迟时，本步使用 `a_(t-1)`；关闭延迟仍保留同样的两拍观测。
默认 3 帧、4 个 ToF，actor 为 `38 + 4 × 3 = 50` 维；关闭 ToF 观测时为 38 维。
Critic 包含相同动作历史。旧 checkpoint 不做输入适配，需要重新训练。

Critic 的传感器通道使用无噪声真值：IMU 使用 `read_ground_truth()`，关节位置/速度、轮速和
腿部几何量使用仿真状态，ToF 使用标称安装位姿的独立无噪声历史（不含安装误差、偏置和白噪声）。
Actor 继续使用传感器测量值。两者保持相同的通道顺序、缩放和采样节奏，critic 的特权信息继续保留。

## 文件职责

| 文件 | 职责 |
| --- | --- |
| `config_locomotion.py` | Locomotion 奖励、命令和迭代课程 |
| `config_common.py` | 机器人、动作、观测配置与命令合并工具 |
| `env.py` / `rewards.py` | 仿真环境、通用扩展接口与经典奖励 |
| `tracking_gate.py` / `velocity_estimator.py` | 跟踪门控与速度估计 |
| `train.py` / `eval.py` | 训练、续训与交互评估 |
| `interactive_viewer.py` | 评估键盘和显示辅助 |

## 训练和续训

命令配置支持按目标机身高度限制速度（多地形任务可在各自 `commands` 中配置）：

```python
"high_base_height_velocity_limits": {"base_height_threshold": 0.34, "max_abs_lin_vel": 1.5, "max_abs_ang_vel": 2.0},
```

采样的目标 `base_height > 0.34 m` 时，限制 `|vx| ≤ 1.5 m/s`、`|wz| ≤ 2.0 rad/s`；
等于阈值时使用原范围。速度在收紧后的区间内均匀采样，原区间与限速区间不相交时使用最近的限速边界。
与 `high_speed_ang_vel` 同时触发时取更严格的角速度上限，高速判定使用收紧后的 `vx`。
静止样本仍只清零速度、保留高度。省略此配置或设为 `None` 可关闭。

```bash
.venv/bin/python -m experiments.genesis.wheel_leg_infantry.tasks.locomotion.train \
  -e locomotion_v4 -B 64 --max_iterations 10001

.venv/bin/python -m experiments.genesis.wheel_leg_infantry.tasks.locomotion.train \
  -e locomotion_v4 --resume --max_iterations 15001
```

只有 `config_locomotion.py` 一套配置，不再需要 `--config`。
默认环境数为 8192；用 `-B` 根据可用显存调整。`--vis` 显示训练环境。
`--resume` 恢复保存配置、优化器及迭代；`--resume-config current` 显式改用当前 locomotion 配置。
`--load-weights CHECKPOINT` 只加载策略与价值网络，优化器和课程从零开始。
动力学、传感器噪声和双轮独立打滑可分别通过 `--domain-rand/--no-domain-rand`、
`--sensor-noise/--no-sensor-noise`、`--wheel-slip/--no-wheel-slip` 覆盖。

### 双轮独立随机打滑

配置位于 `env_cfg["randomization"]["wheel_slip"]`，默认值定义在
[`core/wheel_slip.py`](../../core/wheel_slip.py)：

```python
{
    "enabled": True,
    "strength": 1.0,
    "interval_s_range": [3.0, 8.0],
    "duration_s_range": [0.2, 0.8],
    "ratio_range": [0.1, 0.3],
}
```

每个环境、每个轮子独立采样事件起点间隔、持续时间及低摩擦倍率，允许左右轮事件重叠。
时间单位为秒，向上取整到控制步；最小间隔须不小于最大持续时间。
`ratio_range` 是相对于该回合域随机化摩擦的倍率，实际倍率为
`episode_friction_ratio * (1 + strength * (sampled_ratio - 1))`。
事件只改变接触摩擦，轮速命令和编码器读数仍走原控制/传感器流程；低摩擦不保证每次都产生滑动。

课程使用独立目标 `"wheel_slip": {"strength": 0.4}`。新 locomotion 和 multi_terrain
配置随现有随机化阶段逐步增加强度；课程值在下一次物理 reset 生效，任务阶段交接保留事件。
`enabled=False` 或回合强度为零时不采样事件。关闭域随机化不自动关闭打滑。
旧配置缺少 `wheel_slip` 时保持关闭，可通过 `--wheel-slip` 开启；评估默认关闭，需显式开启。

Genesis 的接触摩擦取双方较大值。事件期间，所有地形实体在对应环境中使用两轮倍率的最小值，
未打滑轮仍保持自身摩擦，事件结束或局部 reset 恢复回合基准。
为保留未打滑轮的原始抓地力，启用时检查车轮模型摩擦不低于所有地形模型摩擦；
当前模型满足该条件。地形降摩擦也可能影响原始摩擦低于地形的其他机器人部位接触。
Genesis 的有效摩擦下限为 0.01。
`env.wheel_slip.diagnostics()` 返回左右轮事件状态、倍率和倒计时，评估控制器诊断会打印状态。

新配置默认启用 `obs_cfg["critic_wheel_slip"]`，在 critic 落地状态之后、任务特权信息之前
加入 `privileged_wheel_slip` 两维，顺序为左轮、右轮：打滑事件激活为 `1.0`，否则为 `0.0`。
该标志表示低摩擦随机化事件，不是由轮速与地速估计的实际滑移；actor 输入保持不变。
默认 locomotion critic 为 62 维，关闭 ToF 时为 50 维。旧配置缺少此开关时保留原维度，
可继续加载原 checkpoint；启用新增观测后，旧 critic 权重不能直接加载。

`config_locomotion.py` 的课程先练站立，再逐步扩大速度、转向和高度范围，最后增加随机化。
默认地面为 plane；`--terrain` 可指定单一测试地形，仍使用 locomotion 的跟踪目标和奖励。
每个训练阶段使用单一地形，不包含混合分配或按成功率升降级；这些功能统一由 `multi_terrain` 管理。
地形几何参数见 [`terrains/`](../../terrains/README.md)。

## 评估

```bash
.venv/bin/python -m experiments.genesis.wheel_leg_infantry.tasks.locomotion.eval \
  -e locomotion_v4

.venv/bin/python -m experiments.genesis.wheel_leg_infantry.tasks.locomotion.eval \
  -e locomotion_v4 --terrain square_wave --difficulty 2

# 多地形训练的 checkpoint 也可使用同一交互评估入口
.venv/bin/python -m experiments.genesis.wheel_leg_infantry.tasks.locomotion.eval \
  -e multi_terrain_v4 --version 0 --ckpt 700
```

评估使用课程最终命令范围，支持键盘命令、ToF 显示以及周期性观测和控制器诊断。
加 `--wheel-slip` 开启随机低摩擦事件。GUI 右上角的 `wheel slip` 行分别显示左右轮：
`READY` 为等待事件（含下次事件倒计时），红色 `ACTIVE` 为低摩擦正在生效（含相对回合基准的
摩擦倍率和剩余仿真秒数），黄色 `ENDED` 为已结束的事件，提示保留约两秒实际时间。
关闭时显示 `OFF` 和开启参数。`wheel vs true vx` 同时显示轮速换算速度、真实前进速度及差值，
便于在平地接地直行时辅助判断空转；`ACTIVE` 表示低摩擦生效，不是实际滑移检测结果。
`--version` / `--ckpt` 选择版本与 checkpoint；`--terrain-size LENGTH WIDTH` 覆盖地形尺寸。
加载多地形 checkpoint 时，从保存的 locomotion 子任务提取命令范围、终止条件和默认地形，
累计应用全局课程对该子任务的最终命令覆盖；仍可用 `--terrain` / `--difficulty` 切换测试地形。
机器人、控制器、观测和网络结构沿用训练快照，评估只加载网络权重。

### 不同速度的刹停距离

`braking_eval.py` 在 Genesis 平地上测试当前策略收到零速命令后的刹停表现，默认无窗口：

```bash
.venv/bin/python -m experiments.genesis.wheel_leg_infantry.tasks.locomotion.braking_eval \
  -e locomotion_v6 --speeds 0.5 1.0 1.5 2.0 --trials 3

# 高度单位为米，支持一次扫描多个高度（--heights 是 --height 的别名）
.venv/bin/python -m experiments.genesis.wheel_leg_infantry.tasks.locomotion.braking_eval \
  -e locomotion_v6 --height 0.20 0.25 0.30 0.38 --speeds 0.5 1.0 --trials 1

# 支持倒车，以及 locomotion / multi_terrain 的指定 checkpoint
.venv/bin/python -m experiments.genesis.wheel_leg_infantry.tasks.locomotion.braking_eval \
  -e multi_terrain_v7 --version 0 --ckpt 700 \
  --speeds -1.0 -0.5 0.5 1.0 --height 0.22 --vis
```

默认测试 0.20、0.25、0.30、0.38 m 四个高度，每个“高度 × 速度”组合重复 `--trials` 次。
只测试一个高度可传 `--height 0.25`；20 cm 应写为 `0.20`，不要写成 `20`。
每次复位为正立静止，先给零速命令等待 2 s，再给指定速度。
实际机身前向速度连续 0.5 s 落在目标 ±0.1 m/s 内，且横向速度、角速度和姿态满足条件后，
将 `vx/wz` 命令归零，高度保持不变。加速最多等待 10 s，未达标记为 `speed_not_reached`。
制动从零速命令送入策略之前计时，保留模型动作延迟和观测历史，不直接清零轮速或动作。
实际三维线速度 ≤0.05 m/s、三维角速度 ≤0.1 rad/s、roll/pitch 均 ≤10°、机身未接触地面，
连续满足 0.5 s 才记为 `stopped`；制动超过 10 s 则记为 `stop_timeout`。
阈值和等待时间均可通过 `--help` 中的参数调整，时间精度为一个控制周期。

结果默认保存到所选训练版本的 `braking_eval/<时间戳>/`，也可用 `--output-dir` 指定新目录：

| 文件/字段 | 含义 |
| --- | --- |
| `trials.csv` | 每次试验的目标高度、目标速度、状态、实际刹车初速度、距离和时间 |
| `target_height_m` | 当前试验的高度命令（米），同时写入逐次结果、轨迹和汇总 |
| `path_distance_m` | 从制动开始到确认停稳的水平累计路程，包括回退和确认窗口内的运动 |
| `displacement_m` | 制动起点与终点之间的水平直线距离 |
| `forward_distance_m` / `max_forward_distance_m` | 沿制动开始时车头方向（倒车取反）的净位移/最大位移 |
| `max_excursion_m` | 制动期间距起点最远的水平距离 |
| `left_wheel_max_forward_distance_m` / `right_wheel_max_forward_distance_m` | 左/右轮轴原点相对各自制动起点，沿制动开始时行驶方向的最大位移 |
| `wheel_max_forward_distance_m` | 左右轮上述最大位移的较大值，包含车体移动与腿部前伸 |
| `wheel_max_forward_from_base_m` | 两个轮轴原点到达的最前方坐标，以制动开始时的 base 位置为零点，包含轮子初始前后偏置 |
| `stop_time_s` | 确认停稳所需时间，包含连续停稳窗口 |
| `summary.csv` | 按目标高度和目标速度分别统计成功/失败数，以及成功试验的均值、总体标准差、最大值 |
| `trace.csv` | 每个控制周期的阶段、命令、base/左右轮轴位置和真实速度；`brake` 的 `phase_time_s=0` 为制动起点 |
| `metadata.json` | checkpoint、随机种子、参数、实际控制周期和评估环境配置 |

超时的距离仅代表观测截止时的运动，`stop_time_s` 留空；环境终止、自动复位或数值异常时距离留空，
不把复位跳变计入路程。所有失败均排除在成功均值之外；查看汇总时同时检查成功数量和实际初速度。
“停稳”只保证在配置的连续窗口内满足条件，不保证之后永不漂移。

轮子位置直接读取 `left_wheel_link` / `right_wheel_link`（名称沿用保存配置）的链接原点，
逐帧记录为 `left_wheel_x/y/z` 和 `right_wheel_x/y/z`，单位米。
最大位移保留制动开始至确认停稳之间的峰值，腿回收或车辆回退不会减小已记录的最大值；
投影方向固定为制动开始时的车头方向，倒车时取反。轮子初始帧也纳入统计。
`wheel_max_forward_from_base_m` 可与 base 的 `max_forward_distance_m` 在同一起点下比较，
若轮轴始终处于该起点后方，则前者为负。这些指标测量轮轴原点，不含轮胎半径，不能直接作为轮胎外缘的障碍物间隙。
`summary.csv` 同样输出新增轮子指标在成功试验中的均值、总体标准差和最大值。

默认关闭动力学随机化、传感器噪声和随机打滑，可分别用 `--domain-rand`、`--sensor-noise`、
`--wheel-slip` 开启。所有试验共享 `--seed` 对应的随机序列，每次 reset 重新采样已启用的随机项。
目标高度使用 `--height` 列表，初始机身高度取保存 reset 范围中点；地形固定为 plane。
重复试验在无随机化时通常相同，可用单次试验快速比较速度。`--backend cpu` 可用于无 GPU 环境。
多地形模型使用与交互式 locomotion eval 相同的子任务配置适配。

## 转入多任务训练

上/下 20 cm 台阶、上/下 23° 坡、变摩擦起伏路和小高度单级上台阶，全部在
[`multi_terrain/config.py`](../multi_terrain/config.py) 的统一任务表中配置。
各任务共用环境与课程实现，没有单独的台阶专项环境或配置文件。

```bash
.venv/bin/python -m experiments.genesis.wheel_leg_infantry.tasks.multi_terrain.train \
  -e multi_terrain_v1 -B 64 --load-weights /path/to/locomotion/model_N.pt
```

旧多地形/上台阶专项 checkpoint 可以作为相同观测接口的权重来源，
其配置和旧专项课程不会通过 locomotion 训练入口继续运行。
多任务完整续训与单任务评估使用 `multi_terrain.train` / `multi_terrain.eval`。
