# 地形模块

本目录管理共享地形几何、参数和场景布局。任务专用的通道及分配逻辑位于各任务目录的 `terrain.py`。

| 文件 | 职责 |
| --- | --- |
| `plane.py` | 平地：默认参数、难度表、碰撞实体和高度查询 |
| `stairs.py` | 下楼梯：默认参数、难度表、箱体几何和高度查询 |
| `stairs_up.py` | 上楼梯：沿 +x 升高的箱体台阶，默认单级，供搭台专项训练 |
| `platform_ridge_down.py` | 下行平台凸台：沿 +x 从第二平台越过凸台，经第一平台降到地面 |
| `platform_ridge_up.py` | 上行平台凸台：沿 +x 从地面登上第一平台、越过凸台，落到第二平台 |
| `trapezoidal_wave.py` | 梯形波：默认参数、难度表、网格和高度查询 |
| `slope_up.py` / `slope_down.py` | 独立上下坡；下坡复用上坡的参数、难度与几何实现 |
| `scene.py` | `TerrainPatches` 统一区域摆放、绑定、高度查询和出生调整；`OffsetScene` 统一坐标偏移和平地有限化 |
| `square_wave.py` | 小高度单级上台阶（保留原入口名）：默认参数、难度表、箱体几何和高度查询 |
| `base.py` | 各单地形共用的 `TerrainGeometry` 接口 |
| `registry.py` | 注册各地形模块，读取参数和难度表快照 |
| `config.py` | 公共尺寸、地形预设与配置校验 |
| `manager.py` | 调用统一几何接口，解析配置、创建几何并提供统一的出生位置和边界接口 |

通用入口：

```python
from src.genesis.wheel_leg_infantry.terrains import (
    TerrainManager, TERRAIN_PRESETS, default_terrain_cfg,
)

terrain = TerrainManager({"preset": "stairs", "difficulty": 1})
step_height = terrain.geometry.step_height
```

地形专用数据统一从 `terrain.geometry` 读取，例如 `step_height`、`boxes`、`heights`；管理器只提供跨地形的公共接口。

每种地形只维护一个同名模块，文件内统一包含 `PRESET`、`PARAMETER_KEY`、`PARAMETERS`、`LEVELS` 和几何类，并通过 `TERRAIN_CLASS` 导出该类。修改难度参数和几何实现都在同一个文件完成。

各几何类使用相同的构造参数，并实现 `add_to_scene()`、`height_at()`；通过公共接口 `bind()` 和 `adjust_spawn()` 处理查询缓存与特殊出生姿态。新增地形时实现该模块并加入 `registry.py`，管理器无需增加地形分支。

每个文件的 `LEVELS` 覆盖其 `PARAMETERS`；等级之间不继承。等级数量以各模块的 `LEVELS` 为准。单地形通过 `difficulty` 选择等级；`TerrainManager(config)` 创建单一地形。多地形配额与自动升降范围统一在 `tasks/multi_terrain` 配置。
Jump 的参数、奖励和平台通道由 `tasks/jump/config.py`、`rewards.py` 和 `terrain.py` 管理。

参数说明和运行示例见 [locomotion 地形说明](../tasks/locomotion/README.md)。

`multi_terrain` 的 task 引用地形类型和难度范围，复用相同几何；成功率升降级由该任务的 `TaskCurriculum` 管理。
有限障碍通过 `traversal_end` 提供最后一个障碍边缘的局部 x 坐标；任务负责落地余量和成功判定。
两种平台凸台都以出生点的名义位置 x=0、朝向 +x 构建，不覆盖 reset 采样的 x/y 或姿态。
`approach_length` 表示出生点到第一个障碍的前向距离：下行是凸台前沿，上行是第一平台前沿。
`second_length` 是第二平台长度：下行从凸台向后延伸且必须覆盖 x=0；上行从凸台向前延伸。
`second_length=None` 表示第二平台延伸到该方向的地块边界。高度和长度单位均为 m。
难度表快照随训练配置保存，续训和评估不受后续修改 LEVELS 数值影响。

`tasks/multi_terrain` 使用 `TerrainPatches` 组织独立区域；区域内高度由地形模块计算，区域外及区域间隙统一返回 0（表示没有该区域的地面，不会创建额外平面）。固定任务配额、独立奖励、通关终点和失败判定均由任务层管理。

上下楼梯通过继承复用参数校验和实体创建，上下坡复用同一几何实现，保留独立模块作为配置入口。`tasks/jump/terrain.py` 管理 Jump 的专用通道和平台。成功率窗口、升降级和检查点状态由 `tasks/multi_terrain/curriculum.py` 管理。

训练和评估统一通过包根目录的 `train.py`、`eval.py` 入口。
