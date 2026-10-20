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
