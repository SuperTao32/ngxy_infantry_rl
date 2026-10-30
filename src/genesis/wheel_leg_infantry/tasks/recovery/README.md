# Recovery

从原 stand_up 功能独立出的起身/恢复任务，为统一策略提供恢复回合。

`config.py` 配置初始姿态、连续站稳条件、奖励和终止；`reset_pos.json` 为当前初始姿态，`reset_pose.py` 将其解析为可保存的数值配置。`buffers.py` 管理站稳计时与成功事件，`rewards.py` 定义恢复奖励，`task.py` 接入共享环境。

交互 eval 的 R 键从 `config.py::get_recovery_reset_cfgs()` 提供的数值姿态池随机采样并复位起身。默认池只有 `reset_pos.json`；扩展时在 `RECOVERY_RESET_POSES` 中追加其他 JSON 快照路径或 `"default"`，无需修改 eval 或按键代码。训练默认复位仍由 `RESET_POSE` 决定。

恢复成功需要连续满足高度、姿态、腿角、接触和速度条件；成功回合作为正常终局结束。恢复任务和另外三个任务共用动作缩放、观测与 GRU policy。

统一评估入口支持 `--tasks recovery`。整体用法见 [框架说明](../../README.md)。

## 采集自然倒地姿态

从仓库根目录运行：

```bash
# 默认轮流使用 default 和 reset_pos.json 的关节初态，从随机高度、RPY 释放。
uv run --locked python -m src.genesis.wheel_leg_infantry.tools.collect_reset_poses --num-poses 32

# 查看落地过程；也可以用 --cpu 无 GPU 运行。
uv run --locked python -m src.genesis.wheel_leg_infantry.tools.collect_reset_poses \
  --num-poses 8 --vis --height 0.6 1.0 --roll -180 180 --pitch -90 90

# 自定义关节初态来源；输入快照的基座坐标会被重新采样的高度、朝向覆盖。
uv run --locked python -m src.genesis.wheel_leg_infantry.tools.collect_reset_poses \
  --reset-poses src/genesis/wheel_leg_infantry/tasks/recovery/reset_pos.json \
  --num-poses 16 --max-attempts 160 --seed 2
```

采集器使用训练的 Genesis MJCF、0.01 s 控制步、20 个物理子步和气弹簧参数。电机为零力矩，保留原模型的摩擦、阻尼与闭链约束；每次完整复位场景后，从零速度释放，不在落地过程中清零速度或增加阻尼。若释放姿态穿地，会自动抬高到碰撞几何离地至少 2 cm，实际初态写入报告。

默认至少模拟 2 s，之后基座和全部关节（包括被动关节、气弹簧和轮子）的速度必须连续 0.5 s 低于阈值，且持续接触地面。线速度各分量阈值为 0.01 m/s，角速度各分量阈值为 0.05 rad/s；可用 `--linear-tol`、`--angular-tol`、`--hold-time` 调整。每次最多 20 s，超时或非有限/求解器错误状态不保存为姿态。`--max-time` 调整单次时间上限。

静止后的左右侧躺姿态会被过滤：机身左右轴与世界竖直线的锐角不超过 30° 时，记录为 `side_lying` 并继续下一次释放，不写入姿态 JSON，也不计入目标数量。可用 `--side-lying-angle-deg` 调整范围，数值越大排除范围越宽；判定不受全局 yaw 影响，前后倾倒和翻面不会仅因倾角大被过滤。`--keep-duplicates` 不会关闭侧躺过滤。

输出位于 `tasks/recovery/reset_pos/run_<UTC时间>/`（可用 `--output-dir` 修改）。`pose_*.json` 与原 `reset_pos.json` 格式一致，包含浮动基座 XYZ、wxyz 四元数和全部活动关节位置，能直接传给 `get_reset_pose_cfg()`。`report.json` 单独记录参数、初态、原始终态速度、超时和去重结果。每次采集创建新目录，已保存的样本在 Ctrl-C 后仍然可用；未达到目标数量时以非零退出码返回。

默认在本次采集内去重，忽略水平平移、全局 yaw 和轮子转角；基座高度差不超过 1 cm、重力方向夹角不超过 5°，且所有其他转动关节差不超过 0.05 rad、滑动关节差不超过 5 mm 时视为重复。`--keep-duplicates` 可关闭。不同释放条件可能落入同一终态，因此尝试次数可能多于保存数。

要扩展交互评估的 R 键姿态池，在 `config.py` 中设置：

```python
RECOVERY_RESET_POSES = (
    RESET_POSE,
    *sorted(Path(__file__).with_name("reset_pos").glob("run_*/pose_*.json")),
)
```

训练仍通过 `RESET_POSE` 指定单个初态；可将其设为某个生成的 JSON。采集脚本不会自动修改训练或评估配置。
