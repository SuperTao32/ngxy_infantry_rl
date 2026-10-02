# 行走地形

当前可选预设为 `plane`、`stairs`、`platform_ridge`、`trapezoidal_wave`。`stairs` 使用固定箱体楼梯。

平地统一使用 `plane`（z=0 解析平面），移除重复的 `flat` 预设。`plane` 默认没有边界复位；设置 `terrain["bounded"] = True` 可按 `tile_size` 与 `boundary_margin` 开启边界 timeout。旧配置中的 `preset="flat"` 自动迁移到 `plane`，默认保留其边界复位行为；其余地形继续按原规则检查边界。

单向斜坡 `slope` / `sloped_terrain` 及由它和平地组成的 `mixed` 已移除。旧配置若选用了 `slope` 或 `mixed`，需通过 `--terrain plane` 或 `--terrain trapezoidal_wave` 等选项显式选择替代地形，避免恢复训练时悄悄改变任务。旧配置中多余的斜坡参数会被忽略。

## 直接选择地形

训练、评估和模型检查均使用 `--terrain` 直接选择地形；不指定时沿用配置中的 `terrain.preset`（默认 `plane`，恢复实验时使用保存的配置）。例如：

```bash
.venv/bin/python -m experiments.genesis.wheel_leg_infantry.tasks.locomotion.train \
  --terrain trapezoidal_wave -e locomotion_trapezoidal_wave -B 64
```


## 按课程调整地形

在阶段的 `targets["terrain"]` 中设置 `preset` 和 `subterrain_parameters`，字段与 `env_cfg["terrain"]` 一致。后续阶段按字段累计覆盖，没有写出的参数继承前一阶段；第一阶段以基础环境配置为起点。

```python
"stages": [
    {"name": "flat", "start_iteration": 0,
     "targets": {"terrain": {"preset": "plane"}}},
    {"name": "low_steps", "start_iteration": 1000,
     "targets": {"terrain": {
         "preset": "platform_ridge",
         "subterrain_parameters": {"platform_ridge": {
             "first_height": 0.10, "ridge_height": 0.18,
             "second_height": 0.15, "second_length": 1.20,
         }},
     }}},
    {"name": "high_steps", "start_iteration": 3000,
     "targets": {"terrain": {"subterrain_parameters": {"platform_ridge": {
         "first_height": 0.20, "ridge_height": 0.35, "second_height": 0.30,
     }}}}},
    {"name": "stairs", "start_iteration": 5000,
     "targets": {"terrain": {"preset": "stairs"}}},
]
```

`platform_ridge` 的高度、长度、凸台宽度、接近距离、底板厚度均可调整，`second_length=None` 仍表示延伸到 tile 边界。`ridge_height` 必须高于两侧平台，所有几何必须能放进 `tile_size`。同样可以切换 `trapezoidal_wave`，并配置相应参数。

`config_downstairs.py` 提供平地 → 低平台 → 标准平台 → 楼梯 → 梯形波的完整示例，可直接修改阶段迭代数和参数：

```bash
.venv/bin/python -m experiments.genesis.wheel_leg_infantry.tasks.locomotion.train \
  --config config_downstairs -e downstairs_course -B 64
```

指定训练 `--terrain` 时，基础配置和各课程阶段的 preset 都被覆盖为该类型，阶段中的几何参数仍然生效。若需要课程切换不同类型，不传 `--terrain`。

所有不同的课程地形在启动时预构建，阶段切换会在当前控制步完成动作和奖励后结束所有回合，再按新地形复位；runner 收到截断标志和新观测。恢复 checkpoint 时按恢复迭代数选中对应地形。多地形课程中的平地使用有限地面，各阶段都启用 tile 边界复位。预构建的实体增加场景内存与碰撞开销；未指定 `-B` 时默认使用 8192 个并行环境。

## eval 地形默认值

评估保留日志中的地形类型（或 `--terrain` 指定的类型），但几何参数和 `tile_size` 每次重新读取当前 `default_terrain_cfg(preset)`，不沿用日志内的旧几何，也不再自动缩成 12 × 6 m。`--terrain-size LENGTH WIDTH` 可显式覆盖尺寸。eval 不运行训练课程；要查看某阶段地形，用 `--terrain` 选择类型，并在 `default_terrain_cfg` 中设置所需参数。


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
