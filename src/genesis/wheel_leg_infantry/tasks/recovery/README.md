# Recovery

从原 stand_up 功能独立出的起身/恢复任务，为统一策略提供恢复回合。

`config.py` 配置姿态池、动静态比例、连续站稳条件、奖励和终止；`reset_pose.py` 解析完整快照，`reset_sampling.py` 为每次 reset 独立采样。`buffers.py` 管理站稳计时与成功事件，`rewards.py` 定义恢复奖励，`task.py` 接入共享环境。

新训练自动读取 `reset_pos.json` 和 `reset_pos/**/pose_*.json`，过滤左右侧躺快照，每个 recovery 环境每次 reset 独立抽取一个完整关节构型。默认 80% 为静态零速度起身，20% 为低难度动态恢复；尚未采集其他快照时，使用原始 `reset_pos.json` 作为种子。

交互 eval 的 R 键仍从 `config.py::get_recovery_reset_cfgs()` 提供的独立数值姿态池采样，默认只有 `reset_pos.json`。可在 `RECOVERY_RESET_POSES` 中追加快照路径或 `"default"`；训练姿态池的自动发现不改变 R 键行为。

恢复成功需要连续满足高度、姿态、腿角、接触和速度条件；成功回合作为正常终局结束。恢复任务和另外三个任务共用动作缩放、观测与 GRU policy。

整体用法见 [框架说明](../../README.md)。

## 训练使用采集结果

采集到默认目录后即可启动训练，无需逐个填写 JSON：

```bash
uv run --locked python -m src.genesis.wheel_leg_infantry.tools.collect_reset_poses --num-poses 32

# 当前根配置 recovery 权重为 0，这里显式启用，聚焦恢复训练。
uv run --locked python -m src.genesis.wheel_leg_infantry.train \
  -e infantry_recovery --tasks recovery --task-weights recovery=1 -B 1024

# 也可以直接指定采集根目录、某次 run 目录或多个快照 JSON。
uv run --locked python -m src.genesis.wheel_leg_infantry.train \
  -e infantry_recovery --tasks recovery --task-weights recovery=1 -B 1024 \
  --recovery-poses /path/to/collection/run_20261030T000000_000000Z \
  --recovery-dynamic-probability 0.2
```

目录递归读取 `pose_*.json`，忽略 `report.json`；显式指定来源时替换默认训练池。重复文件路径只加入一次，侧躺过滤与采集脚本共用 `reset_pose.py::SIDE_LYING_ANGLE_DEG`（当前为左右轴距竖直线不超过 20°）。完整 JSON 的活动关节必须与机器人完全匹配，避免丢失被动关节或拼接出不满足闭链关系的初态；显式 `default` 允许使用默认关节位置。

动态样本保留全部关节位置，仅对机身施加局部前后俯仰扰动（±10°），并抬高 5–15 cm。旋转后会做碰撞几何检查，必要时继续抬高，使最低碰撞点相对地面至少保持本次采样的离地间隙。基座世界系初始线速度范围为 x/y ±0.15 m/s、z −0.2–0 m/s，角速度范围为 x/z ±0.1 rad/s、y ±0.5 rad/s；全部关节初始速度为零。策略从释放时开始控制，不等待落稳。动态样本首次接地前暂停轮子离地惩罚，首次轮子或机身接地后恢复原有惩罚。原 5 s 回合时限、连续站稳成功条件和大倾角持续 3 s 的终止规则保持不变。

默认参数在 `config.py::get_recovery_training_reset_cfg()` 中设置。`--recovery-dynamic-probability 0` 可单独测静态起身，`1` 可单独测动态恢复；每个环境独立按概率抽样，单个小 batch 不保证恰好 80/20。训练配置中的 `recovery_reset.pose_pool` 保存全部数值和源路径映射，`--resume` 默认复用已保存的姿态池，不需要保留采集文件；要在续训时换数据，可显式使用 `--recovery-poses`。不含此字段的旧存档仍使用原单姿态复位，显式指定新来源后才启用新采样器。

TensorBoard 的 `recovery/metrics/` 中保留整体指标，并按以下路径细分：

- `static/success`、`dynamic/success`：按实际结束回合分组的成功率；对应 `recovery_duration_s` 为各组结束回合用时（含失败）。
- `static/front_down/success`、`dynamic/rear_down/success` 等：区分机身近水平 `level`、前倾 `front_down`、后倾 `rear_down`、翻面 `inverted` 和其他倾斜 `tilted` 的种子姿态。
- `pose/0000/success` 等：每份完整快照的成功率，可比较不同收腿构型；编号对应保存配置中的 `pose_pool` 顺序。
- `dynamic_fraction`：本轮结束回合中的动态样本占比。

没有真实结束样本的分组不生成指标。先比较静态与动态、不同姿态的成功率，再调整动态比例、离地高度和俯仰范围。

## 采集自然倒地姿态

从仓库根目录运行：

```bash
# 默认轮流使用 default 和 reset_pos.json 的关节初态，从随机高度、RPY 释放。
uv run --locked python -m src.genesis.wheel_leg_infantry.tools.collect_reset_poses --num-poses 32

# 查看落地过程；也可以用 --cpu 无 GPU 运行。
uv run --locked python -m src.genesis.wheel_leg_infantry.tools.collect_reset_poses \
  --num-poses 8 --vis --height 0.6 1.0 --roll -180 180 --pitch -90 90

# 指定左右腿初始摆角（依次为左腿、右腿，单位为度）。
uv run --locked python -m src.genesis.wheel_leg_infantry.tools.collect_reset_poses \
  --num-poses 8 --vis --leg-angle-deg 20 -20

# 自定义关节初态来源；输入快照的基座坐标会被重新采样的高度、朝向覆盖。
uv run --locked python -m src.genesis.wheel_leg_infantry.tools.collect_reset_poses \
  --reset-poses src/genesis/wheel_leg_infantry/tasks/recovery/reset_pos.json \
  --num-poses 16 --max-attempts 160 --seed 2
```

采集器使用训练的 Genesis MJCF、0.01 s 控制步、20 个物理子步和气弹簧参数。电机为零力矩，保留原模型的摩擦、阻尼与闭链约束；每次完整复位场景后，从零速度释放，不在落地过程中清零速度或增加阻尼。若释放姿态穿地，会自动抬高到碰撞几何离地至少 2 cm，实际初态写入报告。

`--leg-angle-deg LEFT RIGHT` 覆盖所有来源姿态的左右虚拟腿初始摆角，定义与环境中的 `leg_angle` 一致：`(front1 + rear1) / 2`，相对于机身。通过对同侧两个电机角施加相同偏移来保留原姿态的腿长和闭链形状，其他关节相对角保持原值；不传则使用来源姿态的角度。此参数仅设置释放前的初态，不会在下落过程中保持目标角度，实际关节初态保存在 `report.json` 的 `initial_pose` 中。

默认至少模拟 2 s，之后基座和全部关节（包括被动关节、气弹簧和轮子）的速度必须连续 0.5 s 低于阈值，且持续接触地面。线速度各分量阈值为 0.01 m/s，角速度各分量阈值为 0.05 rad/s；可用 `--linear-tol`、`--angular-tol`、`--hold-time` 调整。每次最多 20 s，超时或非有限/求解器错误状态不保存为姿态。`--max-time` 调整单次时间上限。

静止后的左右侧躺姿态会被过滤：机身左右轴与世界竖直线的锐角不超过 20° 时，记录为 `side_lying` 并继续下一次释放，不写入姿态 JSON，也不计入目标数量。可用 `--side-lying-angle-deg` 调整范围，数值越大排除范围越宽；判定不受全局 yaw 影响，前后倾倒和翻面不会仅因倾角大被过滤。`--keep-duplicates` 不会关闭侧躺过滤。

输出位于 `tasks/recovery/reset_pos/run_<UTC时间>/`（可用 `--output-dir` 修改）。`pose_*.json` 与原 `reset_pos.json` 格式一致，包含浮动基座 XYZ、wxyz 四元数和全部活动关节位置，能直接传给 `get_reset_pose_cfg()`。`report.json` 单独记录参数、初态、原始终态速度、超时和去重结果。每次采集创建新目录，已保存的样本在 Ctrl-C 后仍然可用；未达到目标数量时以非零退出码返回。

默认在本次采集内去重，忽略水平平移、全局 yaw 和轮子转角；基座高度差不超过 1 cm、重力方向夹角不超过 5°，且所有其他转动关节差不超过 0.05 rad、滑动关节差不超过 5 mm 时视为重复。`--keep-duplicates` 可关闭。不同释放条件可能落入同一终态，因此尝试次数可能多于保存数。

要扩展交互评估的 R 键姿态池，在 `config.py` 中设置：

```python
RECOVERY_RESET_POSES = (
    RESET_POSE,
    *sorted(Path(__file__).with_name("reset_pos").glob("run_*/pose_*.json")),
)
```

此列表仅控制 R 键。训练会在启动时自动发现默认采集目录，或使用 `--recovery-poses` 指定的来源。
