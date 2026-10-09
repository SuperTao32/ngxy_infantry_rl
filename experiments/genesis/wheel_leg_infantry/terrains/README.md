# 地形模块

所有地形代码集中在本目录，locomotion 和 Jump 通过这里的接口使用地形。

| 文件 | 职责 |
| --- | --- |
| `plane.py` | 平地：默认参数、难度表、碰撞实体和高度查询 |
| `stairs.py` | 楼梯：默认参数、难度表、箱体几何和高度查询 |
| `platform_ridge.py` | 平台凸台：默认参数、难度表、几何、出生姿态和高度查询 |
| `trapezoidal_wave.py` | 梯形波：默认参数、难度表、网格和高度查询 |
| `square_wave.py` | 方波：默认参数、难度表、箱体几何和高度查询 |
| `random_rough.py` | 二维随机起伏：默认参数、难度表、网格和高度查询 |
| `base.py` | 六种单地形共用的 `TerrainGeometry` 接口 |
| `registry.py` | 注册各地形模块，读取参数和难度表快照 |
| `config.py` | 公共尺寸、地形预设与配置校验 |
| `mixture.py` | 按比例分配地形、成功率窗口与自动升降级 |
| `manager.py` | 调用统一几何接口，管理出生 patch、边界和旧接口兼容 |
| `curriculum.py` | 课程参数合并、地形预构建和切换 |
| `jump_terrain.py` | Jump 三种任务通道的平台几何 |

通用入口：

```python
from experiments.genesis.wheel_leg_infantry.terrains import (
    make_terrain, TerrainManager, TERRAIN_PRESETS, default_terrain_cfg,
)
from experiments.genesis.wheel_leg_infantry.terrains.curriculum import TerrainCourse
```

每种地形只维护一个同名模块，文件内统一包含 `PRESET`、`PARAMETER_KEY`、`PARAMETERS`、`LEVELS` 和几何类，并通过 `TERRAIN_CLASS` 导出该类。修改难度参数和几何实现都在同一个文件完成。

各几何类使用相同的构造参数，并实现 `add_to_scene()`、`height_at()`；通过公共接口 `bind()` 和 `adjust_spawn()` 处理查询缓存与特殊出生姿态。新增地形时实现该模块并加入 `registry.py`，管理器无需增加地形分支。

每个文件的 `LEVELS` 覆盖其 `PARAMETERS`；等级之间不继承。平地只有 0 级，其他地形默认 0–4 级。课程通过 `difficulty` 选择等级，通过 `mixture` 配置比例与自动升降范围；统一入口 `make_terrain()` 同时支持单地形与混合地形。
Jump 的任务参数和奖励仍由 `tasks/jump` 管理，地形模块读取其几何相关设置。

参数说明和运行示例见 [locomotion 地形说明](../tasks/locomotion/README.md)。
