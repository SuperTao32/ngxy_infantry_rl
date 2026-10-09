# 行走地形

地形代码统一位于 [`terrains/`](../../terrains/README.md)。每种地形的默认参数、难度表和几何实现放在同名文件中，公共尺寸在 `terrains/config.py` 中修改。

当前可选预设为 `plane`、`stairs`、`platform_ridge`、`trapezoidal_wave`、`square_wave`、`random_rough`。`stairs` 使用固定箱体楼梯。

平地统一使用 `plane`（z=0 解析平面），移除重复的 `flat` 预设。`plane` 默认没有边界复位；设置 `terrain["bounded"] = True` 可按 `tile_size` 与 `boundary_margin` 开启边界 timeout。旧配置中的 `preset="flat"` 自动迁移到 `plane`，默认保留其边界复位行为；其余地形继续按原规则检查边界。

单向斜坡 `slope` / `sloped_terrain` 及由它和平地组成的 `mixed` 已移除。旧配置若选用了 `slope` 或 `mixed`，需通过 `--terrain plane` 或 `--terrain trapezoidal_wave` 等选项显式选择替代地形，避免恢复训练时悄悄改变任务。旧配置中多余的斜坡参数会被忽略。

## 直接选择地形

训练、评估和模型检查均使用 `--terrain` 直接选择地形；不指定时沿用配置中的 `terrain.preset`（默认 `plane`，恢复实验时使用保存的配置）。例如：

```bash
.venv/bin/python -m experiments.genesis.wheel_leg_infantry.tasks.locomotion.train \
  --terrain trapezoidal_wave -e locomotion_trapezoidal_wave -B 64
```


## 地形参数与难度等级

几何参数和实现按地形分文件维护：[`terrains/`](../../terrains/README.md)。
`plane.py`、`stairs.py`、`platform_ridge.py`、`trapezoidal_wave.py`、`square_wave.py`、`random_rough.py`
各自包含 `PARAMETERS`（默认参数）、`LEVELS`（难度表）及对应几何类。等级内未写的字段继承 `PARAMETERS`，不继承前一级。
平地只有 0 级，其余默认 0–4 级；可以增加连续的整数等级。距离单位 m，坡角单位度。

例如在 `stairs.py` 修改：

```python
LEVELS = {
    0: {"step_height": 0.04},
    1: {"step_height": 0.08},
    2: {"step_height": 0.12},
    3: {"step_height": 0.16},
    4: {"step_height": 0.20},
}
```

课程只选择类型和等级：

```python
{"name": "easy", "start_iteration": 0,
 "targets": {"terrain": {"preset": "stairs", "difficulty": 0}}},
{"name": "hard", "start_iteration": 2000,
 "targets": {"terrain": {"preset": "stairs", "difficulty": 3}}},
```

`config_downstairs.py` 已改为等级引用：梯形波 4 级 → 楼梯 4 级 → 平台凸台 0–4 级。
平台等级保留此前各阶段的第二平台高度和接近距离，几何参数现已集中到参数文件。
旧的 `subterrain_parameters` 写法仍可使用；指定 `difficulty` 时，该类型的几何以难度表为准。

## 按比例混合与自动升降级

训练按 `config_locomotion → config_mix_terrain → config_downstairs` 交接权重。
[`config_common.py`](config_common.py) 共享机器人、动作与观测定义，每次返回独立字典；
三份配置各自维护奖励、命令和课程，任务需要的接触阈值、终止条件可在本地覆盖。

```bash
# 1. 平地行走训练
.venv/bin/python -m experiments.genesis.wheel_leg_infantry.tasks.locomotion.train \
  --config config_locomotion -e locomotion_base -B 64

# 2. 加载第一阶段权重，开始混合地形课程（替换为实际 checkpoint 路径）
.venv/bin/python -m experiments.genesis.wheel_leg_infantry.tasks.locomotion.train \
  --config config_mix_terrain -e locomotion_mixed -B 64 \
  --load-weights /path/to/locomotion/model_N.pt

# 3. 加载混合地形权重，开始 downstairs 课程
.venv/bin/python -m experiments.genesis.wheel_leg_infantry.tasks.locomotion.train \
  --config config_downstairs -e locomotion_downstairs -B 64 \
  --load-weights /path/to/mixed/model_N.pt
```

跨阶段使用 `--load-weights`：只加载 actor/critic，优化器和课程迭代从 0 重新开始。
同一阶段中断后使用 `--resume`，恢复已保存的配置、优化器、迭代和地形等级。
`--config config` 和 `--config config_mixed` 仍作为旧名称别名保留。

[`config_mix_terrain.py`](config_mix_terrain.py) 从第 0 次迭代直接训练混合地形，无站立预热。
默认启用平地 / 梯形波 / 方波 / 随机起伏，权重为 10 / 30 / 30 / 30；
楼梯和平台凸台条目保持注释，交给后续 downstairs 阶段训练，也可自行启用。
各非平地类型从 0 级开始，在 0–4 级内独立调整。
前进指令保持 0.5–1.5 m/s，yaw 角速度为 ±0.2 rad/s，站立概率为 0。

混合课程在 0 / 2000 / 4000 次迭代分别使用 0 / 0.4 / 1.0 的动力学与传感器随机化强度，
高度命令从 0.22–0.28 m 扩展至 0.22–0.32 m、0.22–0.36 m。
后两阶段不覆盖地形配置，保留自动升降级的进度。
奖励参数在 `_reward_cfg()` 独立维护：保留速度跟踪和动作平滑、启用落地振荡惩罚，
并放宽越障时的姿态门控。默认参数是迁移训练起点，尚需实际训练验证收敛效果。

修改 `_terrain_cfg()` 中的混合比例和自动升降条件；也可在课程的 `targets["terrain"]` 中覆盖，例如：

```python
"terrain": {
    "mixture": [
        {"preset": "plane", "weight": 20, "difficulty": 0},
        {"preset": "stairs", "weight": 50, "difficulty": 1,
         "min_difficulty": 0, "max_difficulty": 4},
        {"preset": "random_rough", "weight": 30, "difficulty": 0,
         "min_difficulty": 0, "max_difficulty": 4},
    ],
    "adaptive": {
        "enabled": True,
        "window_episodes": 100,
        "promote_threshold": 0.80,
        "demote_threshold": 0.30,
        "min_distance": 2.0,
        "min_duration_s": 2.0,
        "max_lin_vel_rmse": 0.5,
        "max_ang_vel_rmse": 1.0,
        "max_tilt_deg": 60.0,
    },
}
```

- `weight` 是每次 reset 时的抽样权重，无需合计为 1；0 表示不抽取。同一类型只写一次。
  比例是长期回合采样比例，不保证每一时刻的并行环境数量比例精确一致。
- `difficulty` 是初始等级。省略 `min_difficulty` / `max_difficulty` 时，两者都等于初始等级，固定该等级。
  只有一种地形也可以使用一个 `mixture` 条目获得自动升降级。
- 成功必须同时满足：无失败终止、全回合无机身接地或超过倾斜阈值；水平净位移达标；实际回合时长达标；
  前向线速度和 yaw 角速度的全回合跟踪 RMSE 达标。边界 timeout 和自然 timeout 均可作为完成回合；
  初始化、手动 reset 和阶段切换截断不计入统计。不要给要求越障的混合阶段配置纯站立指令。
- 每种地形独立累计当前等级的已完成回合。达到至少 `window_episodes` 个样本后计算一次成功率；
  同一仿真步同时结束的回合整体纳入窗口，达到升级阈值升一级，达到降级阈值降一级，中间区间保持。
  每次判断后清空窗口。升级后在途旧等级回合不会污染新窗口。
- 其他环境继续当前回合，新等级仅影响下次 reset。类型、难度的碰撞几何均预构建在独立 patch，
  出生高度、奖励高度和边界判断使用对应 patch 的局部坐标；平地 patch 同样有边界。
- 后续阶段可以覆盖 `mixture` 整个列表或部分 `adaptive` 字段；改变地形配置时会统一截断并复位所有环境，
  新配置从其初始等级开始。回到单地形时设置 `"mixture": None` 并指定 `preset`、`difficulty`。

训练日志记录 `terrain/<类型>/difficulty`、`success_rate`（最近完成窗口）和 `window_episodes`（当前窗口样本数）。
完整 checkpoint 保存每种地形的等级、窗口计数与成功数；恢复时沿用，`--load-weights` 则重新开始课程。
训练配置保存难度表快照，因此修改参数文件不会改变旧实验的默认续训；需要新几何时使用新课程/加载权重。

训练时不要传 `--terrain`，否则会改成指定的单地形默认参数并移除各阶段的混合配置和等级引用。
多类型、多等级预构建会增加场景内存与碰撞开销。训练入口在任一阶段含混合地形时，
省略 `-B` 默认使用 64 个环境（包括恢复保存的混合课程）；其他课程仍默认 8192。
显式 `-B` 始终优先，建议先验证 64，再逐步增加并行环境数。
碰撞容量按单个环境同时接触的地形计算，不随预构建类型、等级或课程阶段数累加。
若看到 `Jacobian shape ... is too large`，说明约束数 × 自由度数 × 环境数超出 Genesis 的索引上限，
应降低 `-B`；这发生在权重加载之前，与 checkpoint 是否兼容无关。

## 评估指定等级

```bash
.venv/bin/python -m experiments.genesis.wheel_leg_infantry.tasks.locomotion.eval \
  -e locomotion_mixed --terrain stairs --difficulty 2
```

评估不运行自动课程。通过 `--terrain` 选择类型，通过 `--difficulty` 选择当前参数文件里的等级；
`--terrain-size LENGTH WIDTH` 覆盖尺寸。不指定等级时使用该类型的 `PARAMETERS` 默认几何。
混合训练日志的基础 preset 可能为 plane，评估时应显式指定要检查的类型。

## 验证

```bash
.venv/bin/python -m unittest tests.genesis.test_mixed_terrain.MixedTerrainTests -v
NGXY_MIXED_TERRAIN_SMOKE=1 .venv/bin/python -m unittest tests.genesis.test_mixed_terrain.MixedTerrainSmokeTests -v
```

## 方波地形

`square_wave` 沿 x 方向周期排列高平台和低平台，台阶为垂直立面，沿 y 方向铺满地形宽度。默认高差 20 cm，高、低平台各长 1.5 m，周期为 3 m。机器人默认出生在高平台中央 `(0, 0)`；沿 +x 前进 0.75 m 后下台阶。平台使用固定箱体，复位和奖励查询使用同一截面。

```python
"terrain": {
    "preset": "square_wave",
    "subterrain_parameters": {
        "square_wave": {
            "height": 0.20,
            "high_length": 1.50,
            "low_length": 1.50,
            "base_thickness": 0.10,
        },
    },
},
```

## 随机起伏地形

`random_rough` 在 x/y 两个方向生成连续随机起伏，默认相对 `z=0` 的高度范围为 ±5 cm。随机控制点间距默认 0.5 m，经双三次插值生成细网格，中心半径 0.5 m 内保持平整，并平滑过渡到周围起伏。出生中心高度为 0，机器人姿态仍由通用 reset 配置控制。

```python
"terrain": {
    "preset": "random_rough",
    "horizontal_scale": 0.10,  # 最终碰撞网格的水平间距
    "subterrain_parameters": {
        "random_rough": {
            "height": 0.05,  # 最大正负起伏；0 表示平地
            "noise_scale": 0.50,  # 控制点间距，越小起伏越密；不能小于 horizontal_scale
            "spawn_flat_radius": 0.50,  # 平整出生区半径；0 表示不设平整区
            "base_thickness": 0.10,
            "seed": 0,
        },
    },
},
```

地形使用独立的 `seed`，相同参数可复现同一形状，不受训练 `--seed` 和其他随机采样影响。所有并行环境共享该形状，每次 episode reset 不重新生成；要更换形状，可修改地形 `seed` 或在课程下一阶段覆盖它。`height` 以米为单位，不受 `vertical_scale` 量化。

碰撞使用闭合的非凸分块网格，保留凹谷；高度查询按相同三角面插值。多个随机地形阶段拥有独立碰撞实体，切换时由课程统一移入/移出场景。随机地形的首次构建及碰撞开销高于平地。

两种地形均可直接用于训练、评估和模型查看：

```bash
uv run --locked python -m experiments.genesis.wheel_leg_infantry.tasks.locomotion.train \
  --config config_downstairs --terrain square_wave -e locomotion_square -B 64

uv run --locked python -m experiments.genesis.wheel_leg_infantry.tasks.locomotion.train \
  --config config_downstairs --terrain random_rough -e locomotion_random -B 64

uv run --locked python -m experiments.genesis.wheel_leg_infantry.tasks.locomotion.eval \
  --log-root log_shared -e locomotion_v2 --terrain random_rough
```

课程示例（加入 `curriculum_cfg["stages"]`，保持迭代数递增、阶段名唯一）：

```python
{
    "name": "square_steps",
    "start_iteration": 2000,
    "targets": {"terrain": {
        "preset": "square_wave",
        "subterrain_parameters": {"square_wave": {"height": 0.10}},
    }},
},
{
    "name": "random_surface",
    "start_iteration": 4000,
    "targets": {"terrain": {
        "preset": "random_rough",
        "subterrain_parameters": {"random_rough": {"height": 0.05, "seed": 17}},
    }},
},
```

按课程切换类型时不要传训练 `--terrain`；该选项会覆盖所有阶段的地形类型。eval 继续加载当前 `default_terrain_cfg` 中对应类型的默认参数。

```bash
.venv/bin/python -m unittest tests.genesis.test_square_random_terrain -v
NGXY_SQUARE_RANDOM_SMOKE=1 .venv/bin/python -m unittest tests.genesis.test_square_random_terrain.SquareRandomSmokeTests -v
```

## 平台—窄凸台—平台地形

`platform_ridge` 是沿 **+x** 连续排列的三个固定刚体箱体。各高度均相对于同一地面 `z=0`，不是逐级累加的高度；长度与窄凸台宽度均沿 x 方向测量。

下表为 `approach_length=1.0, second_length=0.8` 的截面示例；实际默认值以 `default_terrain_cfg` 为准。

| 区段 | 高度 | 沿 x 方向长度 | 示例 x 范围 |
| --- | --- | --- | --- |
| 一级平台 | 20 cm | 80 cm | 1.00–1.80 m |
| 窄凸台 | 35 cm | 15 cm | 1.80–1.95 m |
| 二级平台 | 30 cm | 80 cm | 1.95–2.75 m |

三段紧邻，一级平台到凸台再上升 15 cm，凸台到二级平台下降 5 cm，二级平台末端下降 30 cm 回到地面。机器人每次复位在二级平台中央，上述示例中 `(x, y) = (2.35, 0)`，朝向 **-x** 的窄凸台和一级平台方向（yaw 为 180°）。机身 z 为平台高度加配置中的初始离地高度（默认 `0.30 + 0.22 = 0.52 m`）。此地形固定复位 x/y 和 yaw，覆盖通用的 x/y、yaw 随机化；其余复位参数仍按配置采样。`approach_length` 表示 tile 中心到一级平台前沿的距离。横向铺满 `tile_size` 的宽度。使用真实箱体保留垂直立面和 15 cm 窄顶面，不受高度场 10 cm 网格分辨率限制。重置和奖励的地面高度查询使用相同箱体边界。

启动训练或用已有策略评估：

```bash
.venv/bin/python -m experiments.genesis.wheel_leg_infantry.tasks.locomotion.train \
  --terrain platform_ridge -e locomotion_platform_ridge -B 64

.venv/bin/python -m experiments.genesis.wheel_leg_infantry.tasks.locomotion.eval \
  --log-root log_shared -e locomotion_v2 --terrain platform_ridge
```

配置项均以米为单位：

```python
"terrain": {
    "preset": "platform_ridge",
    "subterrain_parameters": {
        "platform_ridge": {
            "first_height": 0.20,
            "first_length": 0.80,
            "ridge_height": 0.35,
            "ridge_width": 0.15,
            "second_height": 0.30,
            "second_length": 0.80,  # 可选 None：延伸到地形边界
            "approach_length": 1.0,
            "base_thickness": 0.10,
        },
    },
},
```

```bash
.venv/bin/python -m unittest tests.genesis.test_platform_ridge -v
NGXY_PLATFORM_RIDGE_SMOKE=1 .venv/bin/python -m unittest tests.genesis.test_platform_ridge.PlatformRidgeSmokeTests -v
```


## 梯形波状起伏地形

`trapezoidal_wave` 沿 **x** 方向周期排列：高平台 → 下坡 → 低平台 → 上坡。低平台高度为 0，高平台高度 **20 cm**；高、低平台的水平长度各为 **1.5 m**，上下坡与世界水平面的夹角均为 **23°**。坡段水平长度为 `0.20 / tan(23°) ≈ 0.4712 m`，一个周期约 **3.9423 m**；1.5 m 平台长度不包含坡段。

默认复位于高平台中央 `(x, y) = (0, 0)`，机身 z 为 `0.20 + 0.22 = 0.42 m`，朝向 +x，前方 0.75 m 开始下坡。复位扰动仍由通用配置控制。波形铺满整块地形，边缘处截断，沿 y 方向无起伏。各平台和坡段使用独立固定凸棱柱，避免高度场网格改变平台长度和坡角；复位高度与奖励查询使用相同解析截面。保留地形边缘 timeout 自动复位机制。

```bash
.venv/bin/python -m experiments.genesis.wheel_leg_infantry.tasks.locomotion.train \
  --terrain trapezoidal_wave -e locomotion_trapezoidal_wave -B 64

.venv/bin/python -m experiments.genesis.wheel_leg_infantry.tasks.locomotion.eval \
  --log-root log_shared -e locomotion_v2 --terrain trapezoidal_wave
```

```python
"terrain": {
    "preset": "trapezoidal_wave",
    "subterrain_parameters": {
        "trapezoidal_wave": {
            "height": 0.20,
            "platform_length": 1.50,
            "slope_angle_deg": 23.0,
            "base_thickness": 0.10,
        },
    },
},
```

### 斜坡上的 IMU 与姿态参考系

`base_euler` 是相对世界水平面的 roll/pitch/yaw（度）。初始参考四元数固定为单位四元数，程序没有扣除地形坡角，也没有把每次复位姿态设为新的角度零点。

策略的姿态输入是 `projected_gravity = R_world_to_body × [0, 0, -1]`，即世界重力方向在机身坐标系中的表示。它保留相对世界水平面的绝对 roll/pitch 信息，但不包含绝对 yaw。机身在坡上保持水平时，该向量仍为 `[0, 0, -1]`；机身 pitch 为 +23° 时约为 `[0.3907, 0, -0.9205]`，不会因与坡面平行而归零。该姿态输入当前来自仿真真实姿态，不是带误差的 IMU 姿态融合结果。

原始 IMU 观测 `imu_ang_vel` 是传感器坐标系下的角速度（rad/s），`imu_lin_acc` 是传感器坐标系下的比力 `R_world_to_sensor × (a_world - g_world)`（m/s²），均不是欧拉角。默认传感器轴与机身轴对齐。现有策略观测维度保持不变，已满足世界水平面基准的倾角需求；若需要策略感知绝对航向 yaw，需要另外扩展观测并重新训练。

```bash
.venv/bin/python -m unittest tests.genesis.test_trapezoidal_wave -v
NGXY_WAVE_SMOKE=1 .venv/bin/python -m unittest tests.genesis.test_trapezoidal_wave.TrapezoidalWaveSmokeTests -v
```

## 单点 ToF

默认在底盘前侧面左右上角安装两个向前下方 45° 的 ToF，测距沿光轴计算。
下底面左右边缘距前侧 20 cm 处另有两个沿机身 -Z 垂直向下的 ToF，共四个。
`obs_cfg["tof"]` 配置安装位置、俯角及量程（默认 1.2 米）。策略在原有输入末尾追加
前左、前右、下左、下右四维归一化距离，locomotion 输入从 32 维变为 36 维，需要重新训练。
`history_frames=1` 控制每个 ToF 的历史帧数（正整数，包含当前帧）；例如设为 `5` 时，
actor 输入为 `32 + 4 * 5 = 52` 维。排列为 `[前左旧→新, 前右旧→新, 下左旧→新, 下右旧→新]`。
`update_hz=50.0` 配置 ToF 更新频率（缺省也是 50 Hz），必须为正有限数且不高于环境控制频率。
只有新测距帧到来时追加历史，期间保持上一帧；重复读取观测不会推进历史。
reset 立即测距并用首帧填满该环境的历史，同时重置该环境的采样相位，不影响其他环境。
旧存档缺少此字段时按单帧处理；改变帧数后需要使用匹配输入维度的权重重新训练。
历史名义时间跨度为 `(history_frames - 1) / update_hz`；50 Hz、5 帧对应 80 ms。
当前 Genesis 控制频率为 100 Hz（每两步更新 ToF），MuJoCo 为 50 Hz（每步更新 ToF）。
采样时刻对齐控制步；非整数步周期在下一控制步发布，保留采样相位余量以避免累计漂移。
Genesis Raycaster 内部缓存仍随物理仿真刷新，`update_hz` 控制对外测距及策略历史的更新频率。
`config_common.py` 的 `get_obs_cfg()` 中可分别设置 `forward_reference_distance_m=1.0`、
`downward_reference_distance_m=0.5`；同组左右共用参考距离，观测为 `clamp(距离 / 参考距离, 0, 1)`，测距量程由 `max_range_m=1.2` 独立配置。
训练 jump 的 warmup 权重时，在 `config_common.py` 中设置
`"tof": default_tof_cfg(include_in_observation=False)`，actor/critic 都不加入 ToF，actor 恢复 32 维。
此时传感器仍测距；若同时不需要测距开销，可将 `tof.enabled` 设为 `False`。
Genesis 与 MuJoCo 使用相同配置；详见 [ToF 配置与观测契约](../../core/TOF.md)。
eval 窗口右上角实时显示 `Front L/R`、`Down L/R` 的米制距离、量程和观测开关；
`[no hit/range]` 表示未命中或超出有效量程。旧权重未启用测距时，GUI 会自动开启仅供显示的测距，保持策略输入维度。
