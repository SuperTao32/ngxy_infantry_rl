# 行走地形

当前可选预设为 `plane`、`stairs`、`loose_spheres`、`platform_ridge`、`trapezoidal_wave`。`stairs` 使用固定箱体楼梯。

平地统一使用 `plane`（z=0 解析平面），移除重复的 `flat` 预设。`plane` 默认没有边界复位；设置 `terrain["bounded"] = True` 可按 `tile_size` 与 `boundary_margin` 开启边界 timeout。旧配置中的 `preset="flat"` 自动迁移到 `plane`，默认保留其边界复位行为；其余地形继续按原规则检查边界。

单向斜坡 `slope` / `sloped_terrain` 及由它和平地组成的 `mixed` 已移除。旧配置若选用了 `slope` 或 `mixed`，需通过 `--terrain plane` 或 `--terrain trapezoidal_wave` 等选项显式选择替代地形，避免恢复训练时悄悄改变任务。旧配置中多余的斜坡参数会被忽略。

## 散落刚性球地形

`loose_spheres` 是平地上的独立动态刚体球，直径 **17 mm**（半径 8.5 mm），单球质量 **3.2 g**，材料邵氏硬度 **90A**。球与地面、机器人及其他球参与碰撞，可滚动、可被车轮推动。每个并行环境独立采样；每次物理 reset 重新散布并清零球体线速度和角速度，仅影响被重置的环境。

默认在以出生点为中心的 **4 × 2 m** 区域内散布 **256** 个球，中央 **1 × 1 m** 留空供机器人出生。随机选择互不重复的网格并在格内抖动，避免球体初始重叠。散布范围外仍是平地，球可以被推出散布区；地形 patch 边界继续使用原有 timeout 机制。所有高度奖励和机器人出生高度均以承托球体的平地为基准。

启用训练（先用少量并行环境验证速度和显存，再逐步增加）：

```bash
.venv/bin/python -m experiments.genesis.wheel_leg_infantry.tasks.locomotion.train \
  --terrain loose_spheres -e locomotion_spheres -B 4
```

也可结合原有 `--load-weights CHECKPOINT_PATH` 从已有策略开始新训练。`--terrain` 在读取配置后生效，包括恢复训练时；不指定则保留原配置。评估已有策略：

```bash
.venv/bin/python -m experiments.genesis.wheel_leg_infantry.tasks.locomotion.eval \
  --log-root log_shared -e locomotion_v2 --terrain loose_spheres
```

在所选 `config.py` 或 `config_2real.py` 的 `env_cfg` 中可调整：

```python
"terrain": {
    "preset": "loose_spheres",
    "subterrain_parameters": {
        "loose_spheres": {
            "diameter": 0.017,       # m
            "count": 256,
            "scatter_size": [4.0, 2.0],  # m，须落在 tile_size 内
            "spawn_clearance": 0.5, # 中央无球正方形的半边长，m
            "mass": 0.0032,         # kg，单球 3.2 g
            "shore_a": 90.0,        # 材料硬度记录，当前刚体模型不模拟球体变形
            "friction": 0.3,        # 球体材质摩擦系数
            "ground_friction": 0.8,
        },
    },
},
```

密度由质量和球体积计算：`ρ = mass / (4πr³/3)`，默认约 **1243.95 kg/m³**；Genesis 用球体解析体积和该密度计算质量及转动惯量。调整单球重量请设置 `mass`，旧的 `density` 配置项不再接受。`shore_a` 只记录材料信息，当前刚体模型未模拟 90A 材料的压缩变形，也未将硬度用于设置接触刚度、恢复系数或摩擦系数；摩擦参数仍需标定。启用已有摩擦随机化时，球和地面同样参与。球数在场景构造时确定，reset 只改变位置。`--seed` 控制采样复现；当前课程不改变球数或质量。

动态球显著增加自由度和接触求解开销。不指定 `-B` 时，散落球地形默认使用 4 个并行环境，其他地形保持 8192；显式 `-B` 优先。首次构建可能需要数分钟编译。环境保留 20 个物理子步，并按球数提高碰撞对预算。本改动提供后续抗打滑训练场景，未增加专用抗打滑奖励，也未执行完整策略训练。

本地验证：

```bash
.venv/bin/python -m unittest tests.genesis.test_loose_spheres -v
NGXY_SPHERES_SMOKE=1 .venv/bin/python -m unittest tests.genesis.test_loose_spheres.LooseSpheresSmokeTests -v
```

## 平台—窄凸台—平台地形

`platform_ridge` 是沿前进方向 **+x** 连续排列的三个固定刚体箱体。各高度均相对于同一地面 `z=0`，不是逐级累加的高度；长度与窄凸台宽度均沿 x 方向测量。

| 区段 | 高度 | 沿前进方向长度 | 默认 x 范围 |
| --- | --- | --- | --- |
| 一级平台 | 20 cm | 80 cm | 1.00–1.80 m |
| 窄凸台 | 35 cm | 15 cm | 1.80–1.95 m |
| 二级平台 | 30 cm | 80 cm | 1.95–2.75 m |

三段紧邻，一级平台到凸台再上升 15 cm，凸台到二级平台下降 5 cm，二级平台末端下降 30 cm 回到地面。机器人每次复位在二级平台中央，默认 `(x, y) = (2.35, 0)`，朝向 **+x** 的下台阶方向，距离末端 40 cm。机身 z 为平台高度加配置中的初始离地高度（默认 `0.30 + 0.22 = 0.52 m`）。此地形固定复位 x/y 和 yaw，覆盖通用的 x/y、yaw 随机化；其余复位参数仍按配置采样。`approach_length` 表示 tile 中心到一级平台前沿的距离。横向铺满 `tile_size` 的宽度。使用真实箱体保留垂直立面和 15 cm 窄顶面，不受高度场 10 cm 网格分辨率限制。重置和奖励的地面高度查询使用相同箱体边界。

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
