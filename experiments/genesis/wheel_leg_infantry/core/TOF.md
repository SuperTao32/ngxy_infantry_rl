# ToF 配置与随机化

`obs_cfg["tof"]` 定义标称安装位姿、量程和策略观测。
`env_cfg["randomization"]["sensors"]["tof"]` 定义测距误差和安装误差。
Genesis 训练、评估与 MuJoCo locomotion 使用同一套 ToF 配置。

## 标称传感器与观测

- `enabled`：是否创建传感器；关闭时也关闭策略 ToF 输入。
- `include_in_observation`：是否把 ToF 历史追加到 actor/critic；可独立关闭以训练 jump teacher。
- `link_name`：安装 link，默认 `base_link`。
- `sensors`：列表顺序就是策略输入顺序。`pos_offset` 单位 m，位于安装 link 坐标系；
  `downward_angle_deg` 是从 +X 朝 -Z 的俯角，默认前向组 45°、下向组 90°。
- `min_range_m` / `max_range_m`：有效量程，默认 `[0, 1.2)` m。
- 每个 sensor 的 `reference_distance_m` 决定归一化：`clamp(distance / reference, 0, 1)`。
- `history_frames`：每个传感器保存的帧数，包含当前帧，按传感器分组、组内从旧到新排列。
  输入维度为 `32 + 传感器数 × history_frames`，不受随机化影响。
- `update_hz`：测距更新频率，默认 50 Hz，不能高于控制频率。
  非测距步保持当前读数与历史；reset 的首帧填满该环境的历史。

## 安装外侧余量

默认四路安装高度为 `base_link` 坐标系的 `z=-0.16 m`，向下比原位置外移 10 mm；
X/Y 坐标和标称俯角保持原值。对当前 `wheelbipeV14_2` 的底盘碰撞箱，
三轴位置 ±5 mm 的所有组合仍有至少 10 mm 外侧余量，三轴角度 ±2° 时射线始终向外，
因此不会被该底盘碰撞箱遮挡。初始关节姿态下的位置边界与角度边界/中点组合射线
也未命中其他机器人碰撞体；这一检查不代表任意腿部姿态的遮挡保证。
更换碰撞模型或扩大误差范围时需重新检查。已有存档继续使用保存的 `pos_offset`，不会自动迁移。

## 测距与安装随机化

新训练配置默认包含：

```python
sensors = env_cfg["randomization"]["sensors"]
sensors["enabled"] = True
sensors["tof"] = {
    "enabled": True,
    "std_m": 0.01,                             # 测距高斯白噪声标准差，m
    "bias_range_m": [-0.01, 0.01],             # 每回合固定测距偏置，m
    "position_range_m": [[-0.005, 0.005]] * 3,  # 安装位置 x/y/z，m
    "rotation_range_deg": [[-2.0, 2.0]] * 3,    # 安装角 roll/pitch/yaw，度
}
```

这些幅度是实机标定前的起点。各范围采用均匀分布，环境、传感器和坐标轴之间独立采样。
位置误差叠加到标称 `pos_offset`；角度误差绕安装 link 的 X/Y/Z 轴旋转光轴，
顺序为 `Rz(yaw) @ Ry(pitch) @ Rx(roll)`，不绕机身原点旋转安装位置。
位置和角度误差实际改变射线查询，因此会改变地形交点，而不是仅修正输出距离。

偏置、位置和角度只在该环境 reset 时重采样，回合内保持固定。
白噪声仅在发布新 ToF 帧时采样，包括 reset 首帧；重复组装观测不重复加噪。
有效回波加偏置和噪声后截断到量程边界；无回波或原始无效距离保留 `max_range_m`，
避免无回波经负噪声变成虚假的近距离障碍。

`sensors.strength` 和课程目标 `sensor_noise.strength` 对所有幅度进行线性缩放。
白噪声在下一测距帧使用新强度；测距偏置与安装误差在下一次 reset 使用新强度。
当前 locomotion 课程在第 4000 轮升至 0.4，第 7000 轮升至 1.0，前期为 0。

- `--no-sensor-noise` 关闭本次运行的传感器随机化，包括 ToF；`--sensor-noise` 开启总开关。
- 仅关闭 ToF 随机化：`sensors["tof"]["enabled"] = False`。
- 仅关闭测距误差：`std_m=0` 且 `bias_range_m=[0, 0]`。
- 仅关闭安装误差：位置和角度的三轴范围全部设为 `[0, 0]`。
- 固定偏移可用退化范围，如 `position_range_m=[[0.003, 0.003], [0, 0], [0, 0]]`；
  固定值仍乘课程强度。永久标称安装位置应直接修改 `obs_cfg["tof"]["sensors"]`。

旧存档缺少 `sensors.tof` 时保持理想 ToF；仅开启总开关不会给旧存档自动新增误差组。
需要在旧配置显式加入上述组后才会启用。
MuJoCo 的 `--sensor-noise/--no-sensor-noise` 仅覆盖 ToF，其他传感器仍沿用其理想测量实现；
默认使用存档中传感器的开关及 `strength`，不运行训练课程。

## 后端与验证

随机化配置入口位于 `randomization_config.py`；测距、偏置和安装误差统一位于
`sensor_randomization.py`，由 `ToFRandomization` 执行。
`tof.py` 负责标称安装配置、采样时钟、历史和 Genesis 射线查询。

Genesis 当前版本在 build 时将标称安装位姿烘焙进共享射线。
有安装误差时，适配器使用独立查询缓存和每环境传感器坐标系，复用现有 BVH；
不修改共享 Raycaster cache，也不推进 IMU 或仿真时间。
此路径依赖 Genesis 的私有 Raycaster metadata/query 接口，升级 Genesis 时应重跑 smoke 测试。
没有安装误差时继续使用原有测距缓存路径。
MuJoCo 在 reset 时更新 site 位置与方向，然后刷新运动学。

```bash
.venv/bin/python -m unittest tests.genesis.test_tof_randomization -v
NGXY_GENESIS_SMOKE=1 .venv/bin/python -m unittest tests.genesis.test_tof_randomization -v
```

测试覆盖噪声统计、局部 reset、随机种子、旧配置兼容、采样与历史，
以及两后端真实平面射线交点。测试遵循仓库约定，位于 Git 忽略的 `tests/`。
