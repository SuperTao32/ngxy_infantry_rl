# Jump：平地 / 20 cm / 40 cm 台阶

一个共享 actor，用三维 one-hot 选择固定任务：

| mode | actor 输入末尾 | 目标台面高度 |
| --- | --- | --- |
| flat | `[1, 0, 0]` | 0 m（仍需执行跳跃） |
| step_20cm | `[0, 1, 0]` | 0.20 m |
| step_40cm | `[0, 0, 1]` | 0.40 m |

actor 为 47D，最后 3D 是任务 mode。locomotion teacher 读取前 32D。
critic 为 66D，额外可见边缘距离和落稳进度；actor 不读取位置、台阶距离或 ToF。
训练 warmup teacher 时，将 locomotion 的 `tof.include_in_observation` 设为 `False`；
已有兼容的 32D teacher 也可继续使用。包含 ToF 观测的 teacher 会在加载前被拒绝。
速度估计已从观测中删除；旧输入维度的 checkpoint 不能直接加载，需重新训练 locomotion teacher 和 jump。
动作空间保持原来的 4 个腿关节 + 2 个轮子。

默认流程为起跳 0.15 s → 腾空 0.45 s → 落地 0.80 s，总时长 1.40 s。
预热后直接进入起跳，不设 crouch 阶段及相关奖励。critic 的物理阶段编码为三维 one-hot；
实际离地和触地事件决定物理阶段，actor 仍使用六维连续时间编码。

## env 指定起跳距离

jump 默认关闭 ToF，起跳位置使用下面的速度—距离表。
locomotion 的观测开关详见 [ToF 配置](../../core/TOF.md)。

每轮先在远离障碍的平地上用冻结的 locomotion teacher 稳速，随后：

1. 根据速度指令和 mode，在 `config.py` 的 `jump_modes.distance_tables` 中线性插值。
2. 台阶模式加上 `distance_jitter_m` 范围内的均匀扰动，平地不使用边缘距离。
3. 只平移机器人的 x/y 到指定起跳点，保留高度、姿态、关节/机身速度和动作历史。
4. 从这里启动完整 jump rollout，结束后重新采样下一批任务。

距离定义为 **jump 开始时机身原点到台阶前沿的 +x 距离**，不是离地时轮子到边缘的距离。
台阶前沿为 x=0，三条通道的 y 分别为 0、4、8 m。两座平台为真实固定 Box，长 6 m、宽 2 m。

这是给定触发条件下的技能训练，不训练从远处自主接近或决定起跳时刻。
真机需要上层在对应起跳位置触发；可以使用 ToF，也可以用定位、里程计或已知场地坐标。

距离表是初始标定值，**尚不代表已验证的最佳起跳距离或 40 cm 成功能力**。
例如当前 1 m/s 对应 20 cm 模式 0.30 m、40 cm 模式 0.40 m。根据评估结果修改即可。
超出表格的速度会报错，不静默外推；扩展速度课程时要同步扩展距离表和预热区长度。

## 配置与奖励

所有跳跃配置集中在唯一的 `config.py`，不再提供 `--config` 选项。按调参内容查找：

| 配置位置 | 内容 |
| --- | --- |
| `get_cfgs()` | 阶段时长、电机参数、初始状态、终止条件与配置组装 |
| `_get_warmup_cfg()` | teacher 预热指令范围与站稳条件 |
| `_get_jump_modes_cfg()` | 各模式 base 到轮底距离轨迹、起跳距离、场地与落台判定 |
| `_get_obs_cfg()` | 观测维度与缩放 |
| `_get_reward_cfg()` | 奖励参数、权重与禁用项 |
| `_get_curriculum_cfg()` | 任务采样比例、速度和奖励课程 |

base 到轮底的距离轨迹集中在 `base_to_wheel_bottom_trajectories`，按 `flat`、`step_20cm`、`step_40cm` 独立配置。
平地与 20 cm 模式目前数值相同，但可以分别调整。对外仍输出原有的 `height_references` 列表。

任务参数位于 `env_cfg["jump_modes"]`：

- `mode_probabilities`：运行时采样权重，从课程的 `targets.terrain.mode_probabilities` 更新；
  调参请修改 `_get_curriculum_cfg()` 各阶段中的该字段，顺序为平地 / 20 cm / 40 cm。
  例如 `[1/3, 1/3, 1/3]` 表示等概率，`[1, 0, 0]` 表示只训练平地。后续阶段省略该字段时继承前一阶段；
  关闭课程时使用首阶段的初始比例。切换课程后在下一次 reset 采样，mode 在整个 rollout 内冻结。
  `assignment="cyclic"` 按环境编号分配模式，不使用随机权重。
- `clearance_targets_m`：base_link 原点相对起跳面的峰值目标，默认 0.30 / 0.30 / 0.50 m。
- `distance_tables`、`distance_jitter_m`：每个模式的速度—起跳距离关系。
- `height_references`：每个模式的目标 base 到左右轮底的平均世界竖直距离轨迹。
  实测距离为 `base_z - mean(wheel_center_z - wheel_radius)`，单独用于伸腿/收腿姿态跟踪。
  参考在整个 jump 周期内生成，并写入 `commands[:, 2]`；warmup 仍使用 teacher 的 base 原点离地高度指令。
  `base_to_wheel_bottom_distance_tracking` 用该相对距离计算奖励，仅在落地双轮有效支撑时生效；权重为 `10.0`。
  跟踪容差为 `0.02 m`，`height_reference_sigma=0.0025 m²`；起跳和腾空期间参考仍作为观测输入。
- 跳高进度、峰值、缺高、跳高跟踪及平地成功判定使用 `jump_base_height`，即 base 原点相对起跳地面的竖直高度。
  所有模式都从 z=0 地面起跳，进入台阶上方也不扣除台阶高度；不使用碰撞体底面或轮底高度。
  机身位置不变时，收腿不会提高跳高得分；腿部距离跟踪仍可独立鼓励所需的伸腿/收腿动作。
  高度目标数值不变，表示绝对离起跳地面高度，不是相对初始站姿的上升量。
  critic 的跳高与峰值通道使用同一机身高度，维度和缩放保持不变。
  日志使用 `peak_jump_base_height_m`。
- 起跳速度基准为 `sqrt(2*g*max(目标绝对高度 - 最后支撑拍 base 高度, 0))`。
  支撑高度与速度同步更新，首次离地后锁存；归一化分母最低为 `1 m/s`，避免剩余高度趋零时奖励骤增。
  持续速度奖励下限为 `-1`，起跳事件速度奖励下限为 `0`；两者正向均不封顶，超过理论所需速度仍增分。
- `flight_peak_height` 按新增峰值除以目标高度给分，`flight_height_progress` 按当前高度除以目标高度给分，
  两者均无上界；失稳时峰值奖励全额撤回，包括超过目标的部分。缺高惩罚仍只在未达标时生效。
  默认关闭 `flight_height_tracking`，避免高斯跟踪在接近目标时变平、超过目标后反向扣减收益。
  目标高度用于奖励归一化，成功判定使用独立阈值；目标不再是高度奖励上限，策略可能学到超过目标的跳高。
- `flight_wheel_clearance` 在 `flight_gate` 内奖励较低轮底相对起跳面的高度（米），
  到 `jump_base_height - 0.18 m` 后饱和；轮底高度及饱和上限均不低于零。
  这是相对机身的收腿上限，随实际机身高度上升，不限制整体跳高收益。
  默认权重为 `10.0`，用于鼓励双腿收起，不再奖励超过该上限的收腿动作。
- `landing_vertical_velocity`：首次起跳后的首次任意轮/机身触地时，按触地前最后腾空拍的
  base_link 世界竖直速度平方惩罚，向上/向下均计入；失稳、错误落台也不免除。
  每回合仅结算一次，后续支撑或二次弹跳不重复结算。默认权重 `-100.0`，在 `reward_scales` 中调整；
  实际奖励为 `权重 × dt × vz²`，不依赖跳高 target，也不受落地有效支撑门控影响。
- `min_forward_speeds_m_s`：台阶默认至少 0.6 m/s；平地包含静止跳和移动跳。
- 速度与任务采样比例均由课程阶段控制；被采样的台阶模式，其最小速度不能高于该阶段速度上限。

共用 jump 的姿态、跳高和动作平滑奖励，并增加目标落地与成功奖励。
台阶成功要求发生过起跳、双轮在目标台面内部接触、姿态和竖直速度稳定持续 0.16 s。
无台阶的平地跳跃使用 `jump_modes.ground_success_wheel_clearance_m` 和
`jump_modes.ground_success_base_height_m`，按 `flat`、`step_20cm`、`step_40cm` 排序，独立于奖励 target。
默认 `flat` 要求轮底 > 0.32 m、base_link > 0.50 m；`step_40cm` 移走台阶时要求轮底 > 0.40 m、base_link > 0.55 m。
轮底使用首次触地前两侧较低轮底的峰值，base_link 使用首次腾空峰值；高度相对 z=0 起跳地面，等于阈值不算达标。
撞立面、落回台阶下方、悬在台面上方都不算成功。
二次腾空和 base_link 碰撞不直接判失败、不提前终止；`landing_airborne` 和 `base_contact` 默认权重均为 `-1000.0`，
发生期间逐步扣分并乘 dt（dt=0.02 s 时每项每步扣 20 分）。课程继承这两个权重。
恢复无机身接触的双轮稳定支撑后仍可成功；首次触地后冻结高度成绩，二次弹跳不能补足高度或重复领取起跳奖励。
到回合结束仍未满足成功条件，照常判任务失败。
正常任务失败允许提前终止 PPO 样本；solver error 仍中止训练。
日志分别记录 `success_flat`、`success_step_20cm`、`success_step_40cm`。

## 终止配置

统一修改 `config.py` 的 `_get_termination_cfg(episode_length_s)`：

| 配置 | 默认值 | 含义 |
| --- | --- | --- |
| `jump_max_tilt_deg` | `10.0` | 合成倾角超过阈值即锁存任务无效 |
| `jump_termination.on_invalid` | `True` | 过倾、撞沿或错误落台后立即提前终止 |
| `termination_if_roll_greater_than` | `30.0` | 父环境 roll 绝对值阈值（度） |
| `termination_if_pitch_greater_than` | `30.0` | 父环境 pitch 绝对值阈值（度） |
| `tilt_termination_duration_s` | `0.5` | 父环境 roll 或 pitch 连续超限多久终止 |
| `base_contact_termination_duration_s` | `None` | 关闭父环境机身触地终止，仅保留碰撞惩罚 |

`on_invalid` 控制无效任务是否提前结束；关闭后仍锁存过倾、撞沿或错误落台的失败。
旧配置中的 `on_rebound` 不再生效。父环境倾角终止条件独立生效，也适用于 teacher 预热。
默认 jump 的 10° 过倾会经 `on_invalid` 立即终止，不会等到父环境的持续时间门槛。
回合超时由 `phase_durations_s` 的总时长决定；solver error 始终终止并中止训练。

## 使用

在仓库根目录执行。配置检查不需要 Genesis 初始化或 checkpoint：

```bash
uv run --locked python -m experiments.genesis.wheel_leg_infantry.tasks.jump.train --dry-run
```

从现有 locomotion checkpoint 初始化新 actor（新增加的输入列初始为零）：

```bash
uv run --locked python -m experiments.genesis.wheel_leg_infantry.tasks.jump.train \
  --locomotion-log-root log_shared --locomotion-exp-name infantry_locomotion_v3 \
  -e infantry_jump -B 1024 --max-iterations 2001
```

`--resume` 用于恢复当前统一任务的新训练结果：

```bash
uv run --locked python -m experiments.genesis.wheel_leg_infantry.tasks.jump.train \
  -e infantry_jump --resume --max-iterations 4001
```

恢复默认使用保存的配置；要使用当前 `config.py`，追加 `--resume-config current`。
模式含义和输入顺序必须保持一致。

自动评估三个模式（默认可视化，追加 `--headless` 可关闭）。`all` 每批轮换模式分配，
画面中的环境 0 按 `flat → step_20cm → step_40cm → flat` 循环，切换时镜头重新对准机器人。
`-B 3` 每批仍覆盖三个模式；`-B 1` 也支持轮播，至少运行 `--episodes 3` 才能看完三个模式。
`--episodes` 表示总批数，每批每个环境执行一次跳跃，控制台打印环境 0 的当前模式及累计成功率。

```bash
uv run --locked python -m experiments.genesis.wheel_leg_infantry.tasks.jump.eval \
  -e infantry_jump --mode all --speed 1.0 -B 3 --episodes 10
```

只评估 40 cm：`--mode step_40cm --speed 1.0 -B 1`。
评估关闭距离扰动和动力学随机化；默认随机速度来自保存的初始 warmup 配置，建议用 `--speed` 分别检查各速度点。
