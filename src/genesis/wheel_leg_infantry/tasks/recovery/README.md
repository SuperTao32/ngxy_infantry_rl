# Recovery

从原 stand_up 功能独立出的起身/恢复任务，为统一策略提供恢复回合。

`config.py` 配置姿态池、动静态比例、连续站稳条件、奖励和终止；`reset_pose.py` 解析完整快照，`reset_sampling.py` 为每次 reset 独立采样。`buffers.py` 管理站稳计时与成功事件，`rewards.py` 定义恢复奖励，`task.py` 接入共享环境。

新训练只读取 `config.py::RESET_POSE_DIFFICULTIES` 中明确列出的快照，过滤左右侧躺快照，每个 recovery 环境每次 reset 独立抽取一个完整关节构型。默认 80% 为静态零速度起身，20% 为低难度动态恢复。不会自动扫描采集目录，也不会额外加入未配置的 `reset_pos.json`。

复位时按当前模型的碰撞几何检查离地高度。静态 pose 若穿地，只向上抬到至少 0.1 mm 间隙；动态 pose 保留采样的释放间隙。关节、姿态和初始速度按原配置使用，原 JSON 无需改写；交互 R 键加载的旧快照也使用同样的高度校正。

交互 eval 的 R 键从 `config.py::get_recovery_reset_cfgs()` 提供的独立数值姿态池等概率采样，当前清单为 `reset_pos.json` 和 `pose_leg_bb.json`。直接在 `RECOVERY_RESET_POSES` 中写“文件名 -> 难度”，无需逐个定义 Path 变量：

```python
RECOVERY_RESET_POSES = {
    "reset_pos.json": 0,
    "pose_leg_bb.json": 0,
    "pose_pitch90.json": 1,
}
RECOVERY_RESET_DIFFICULTY_RANGE = [0, 0]  # 只测 0 级；None 使用清单全部姿态。
```

`reset_pos.json` 使用 `RESET_POSE`，其余裸文件名自动位于 `RESET_POSE_DIR`（`reset_pos/` 子目录）；也支持 `"default"`、绝对路径及显式相对路径（相对工作目录）。要复用训练清单，可写 `RECOVERY_RESET_POSES = RESET_POSE_DIFFICULTIES.copy()`，再按 eval 难度范围筛选。难度只用于筛选，不是采样权重；每次 R 可重复抽到同一个姿态。未配置文件不会自动加载，筛选无候选时会明确报错。旧式 Path/字符串 tuple/list 继续使用原路径规则，不进行难度筛选。eval 启动时从当前源码读取清单；它不使用 checkpoint 保存的训练池或动态释放概率。

恢复成功需要连续满足高度、姿态、腿角、接触和速度条件。站稳当拍只发放一次起身成功和速度奖励，不立即结束回合；此后继续保持零线速度、零偏航角速度和起身高度命令，直到 `episode_length_s`（默认从 reset 起共 5 s）再作为成功终局结束。保持期间物理失败仍会提前终止。恢复任务和另外三个任务共用动作缩放、观测与 GRU policy。

尚未完成起身便超时或越界截断的 recovery 回合，以及任意阶段发生物理失败的回合，统一触发一次 `death` 惩罚，数值由 `reward_scales.death` 控制，不乘控制步长。已完成起身后的超时或越界截断不扣分；物理失败优先，同拍存在多个结束原因也只扣一次。未成功回合的超时继续保留原有的价值 bootstrap 处理，其他任务的 `death` 规则不变。起身用时在首次站稳后冻结，不包含后续静止阶段。

`standing_drift` 从首次站稳当拍开始持续生效，起身阶段为 0。它复用公共的水平速度 L1 漂移项：`sum(clamp(abs(base_lin_vel_xy) - standing_drift_deadband, min=0))`，默认死区 0.01 m/s，不乘姿态/高度 gate。`reward_scales.standing_drift` 默认 -20.0，按控制步长缩放，漂移越小惩罚越少；日志为 `recovery/rewards/standing_drift`。普通续训沿用存档权重；旧存档要启用此项，使用 `--resume-config current`。

`standing_still` 同样从首次站稳当拍开始逐拍发放，起身阶段为 0，直到回合结束；它与 `standing_drift` 一起约束后续静止，不依赖单拍的 `just_completed`。此项复用 locomotion 的公共指数奖励：`height_gate * attitude_gate * exp(-(vx² + vy²) / lin_scale² - wz² / ang_scale²)`，仅静止命令下生效，无速度死区，直接追踪真实水平零线速度和零偏航角速度。默认 `standing_still_lin_vel_scale=0.01 m/s`、`standing_still_ang_vel_scale=0.02 rad/s`，尺度越小越严格；高度/姿态门不带保底值，门关闭时奖励为 0。`reward_scales.standing_still` 默认 2.0，按控制步长缩放，站稳且速度为零时每步贡献 `2.0 * dt`，日志为 `recovery/rewards/standing_still`。普通续训沿用存档权重；旧存档启用此项需使用 `--resume-config current`。

Recovery 的 `base_height` 使用姿态混合惩罚：`active * (height-target)**2 + (1-active) * clamp(H_REF-height, min=0) / 15`。`active` 由 `-projected_gravity[:, 2]` 计算，倾角 ≤30° 为 1，≥80°（包括翻面）为 0，中间按余弦线性过渡。height 使用相对当地地面的机身高度，target 使用当前高度命令；`stand_up_height_reference_m` 是 H_REF，默认 0.22 m，旧存档未配置时取 `stand_up.height`。权重仍为 `reward_scales.base_height=-10.0`，按控制步长缩放。

`wheel_ground_distance` 仅在翻倒时惩罚两个轮子轮底的离地距离之和：`(1-active) * sum(clamp(wheel_center_z - wheel_radius - ground_z, min=0))`，单位为米。`active` 与 `base_height` 共用姿态门：倾角 ≤30° 不惩罚，≥80°（包括翻面）全额惩罚，中间按姿态余弦渐变。地面高度分别在两轮的世界 XY 位置查询，接地或轻微穿地时该轮贡献为 0。默认权重 `reward_scales.wheel_ground_distance=-50.0`，按控制步长缩放，日志为 `recovery/rewards/wheel_ground_distance`。`wheel_airborne` 仍按离地轮数计算，不受此姿态门影响；动态释放到首次轮子或机身接地前，两项惩罚均暂停。旧存档若没有此权重，普通续训不会自动启用；使用 `--resume-config current` 可采用当前配置。

`stand_up_wheel_push` 在翻倒姿态下奖励轮子蹬地：`(1-active) * clamp(sum(max(F_up, 0)) / F_REF, 0, 1)`。`active` 与 `base_height` 共用姿态门，倾角 ≤30° 时此奖励为 0，≥80°（包括翻面）时全开，中间平滑过渡。`F_up = -dot(wheel_normal_force, projected_gravity)` 从机身坐标系的法向力取出世界竖直向上的支撑分量，再对两轮求和；离地、水平法向力及向下分量不贡献奖励。`stand_up_wheel_push_force_reference_n` 默认 200 N，达到后饱和；`reward_scales.stand_up_wheel_push` 默认 5.0，按控制步长缩放。日志为 `recovery/rewards/stand_up_wheel_push`。动态释放空中无接触力时为 0，触地后立即按实际支撑力计算。普通续训沿用旧存档权重，使用 `--resume-config current` 可启用此项。

`stand_up_leg_angle` 是双腿朝下的连续奖励。`stand_up_leg_angle_tolerance_deg` 默认 10°，双腿绝对摆角均不超过该值时原始奖励饱和为 1；超出部分 `excess = clamp(abs(leg_angle) - radians(tolerance_deg), min=0)` 按 `exp(-mean(excess**2) / stand_up_leg_angle_sigma)` 衰减。腿角单位为弧度，0 表示沿机身 -Z，左右腿误差不会抵消；机身倾斜由独立的 `base_balance` 项约束。sigma 越小，容差外衰减越快。`reward_scales.stand_up_leg_angle` 默认 5.0，按控制步长缩放，并记录到 `recovery/rewards/stand_up_leg_angle`。

`stand_up_speed` 在成功当拍额外奖励起身速度：`clamp(1 - elapsed_s / episode_length_s, 0, 1)`，只发放一次，不乘控制步长。计时从 reset 开始，包含动态释放和连续站稳确认；未成功或成功当拍发生物理失败时为 0。权重 `reward_scales.stand_up_speed` 默认 50，日志为 `recovery/rewards/stand_up_speed`。默认 5 s 回合内，1/3/5 s 成功分别获得额外 40/20/0 分，另有基础 `stand_up_success` 奖励 50 分。

整体用法见 [框架说明](../../README.md)。

## Pose 难度与课程选择

在 `config.py::RESET_POSE_DIFFICULTIES` 中按文件名为每个 pose 指定非负整数难度，数值越大越难。当前初始分级为：`reset_pos.json` 和 `pose_leg_bb.json` 为 0 级，`pose_leg_bf.json` / `pose_leg_fb.json` 为 1 级，`pose_leg_ff.json` 为 2 级，`pose_pitch90.json` / `pose_pitch-90.json` 为 3 级。这些是可调整的人工分级，不根据机身倾角自动推断；可结合训练成功率重新指定。

快照 JSON 继续只存坐标，无需修改格式。`RESET_POSE_DIFFICULTIES` 同时控制默认训练池的文件清单和难度：`reset_pos.json` 对应 `RESET_POSE`，其他文件名相对 `RESET_POSE_DIR`。新增文件必须加入清单，删除清单项即可移出默认训练池；配置文件缺失时直接报错，不自动寻找替代文件。显式 `--recovery-poses` 会替换默认清单，并按文件名匹配难度；仅在此显式覆盖下，未配置难度的文件才按 0 级处理。侧躺过滤仍然生效。

在 recovery 自己的 `config.py::_curriculum_cfg()` 的阶段 `targets` 中设置：

```python
"recovery_reset": {
    "difficulty_range": [0, 1],  # 包含上下界：只从 0、1 级 pose 中等概率抽样
    "dynamic_probability": 0.0,  # 可选：本阶段只练静态起身
},
```

`[2, 2]` 只选 2 级，`None` 使用全部难度。当前默认课程保留单阶段、全部难度和原来的 20% 动态比例。若要逐步增加难度，可把 `stages` 改为以下示例；轮数应按训练表现调整：

```python
"stages": [
    {"name": "easy", "start_iteration": 0,
     "targets": {"recovery_reset": {"difficulty_range": [0, 0], "dynamic_probability": 0.0}}},
    {"name": "medium", "start_iteration": 1000,
     "targets": {"recovery_reset": {"difficulty_range": [0, 2]}}},
    {"name": "hard", "start_iteration": 2000,
     "targets": {"recovery_reset": {"difficulty_range": [0, 3], "dynamic_probability": 0.2}}},
],
```

阶段采用累计覆盖，未写出的设置继承前一阶段；这里 medium 仍为静态起身。`start_iteration` 使用 recovery 独立课程的样本折算轮数，不是全局 PPO 轮数。切换从下一次 reset 生效，进行中的回合和 pose 编号保持原样；若筛选范围内没有可用快照，会明确报错。

难度和全部 pose 数值一起保存在 `recovery_reset.pose_pool`。普通 `--resume` 使用存档的分级和课程；要采用修改后的配置，用 `--resume-config current`，或用 `--recovery-poses` 重新加载姿态池和分级（课程仍由所选续训配置决定）。旧池中缺少 `difficulty` 的条目按 0 级处理。交互 eval 的 R 键仍由独立的 `RECOVERY_RESET_POSES` 控制。

## 训练使用采集结果

采集后，将要使用的快照加入配置清单，或通过 `--recovery-poses` 显式指定来源：

```bash
uv run --locked python -m src.genesis.wheel_leg_infantry.tools.collect_reset_poses --num-poses 32

# 使用配置清单中的 pose；当前根配置 recovery 权重为 0，这里显式启用。
uv run --locked python -m src.genesis.wheel_leg_infantry.train \
  -e infantry_recovery --tasks recovery --task-weights recovery=1 -B 1024

# 也可以直接指定采集根目录、某次 run 目录或多个快照 JSON。
uv run --locked python -m src.genesis.wheel_leg_infantry.train \
  -e infantry_recovery --tasks recovery --task-weights recovery=1 -B 1024 \
  --recovery-poses /path/to/collection/run_20261030T000000_000000Z \
  --recovery-dynamic-probability 0.2
```

只有显式传入目录来源时才递归读取其中的 `pose_*.json`，忽略 `report.json`；显式指定来源时替换默认训练池。重复文件路径只加入一次，侧躺过滤与采集脚本共用 `reset_pose.py::SIDE_LYING_ANGLE_DEG`（当前为左右轴距竖直线不超过 20°）。完整 JSON 的活动关节必须与机器人完全匹配，避免丢失被动关节或拼接出不满足闭链关系的初态；显式 `default` 允许使用默认关节位置。

动态样本保留全部关节位置，仅对机身施加局部前后俯仰扰动（±10°），并抬高 5–15 cm。旋转后会做碰撞几何检查，必要时继续抬高，使最低碰撞点相对地面至少保持本次采样的离地间隙。基座世界系初始线速度范围为 x/y ±0.15 m/s、z −0.2–0 m/s，角速度范围为 x/z ±0.1 rad/s、y ±0.5 rad/s；全部关节初始速度为零。策略从释放时开始控制，不等待落稳。动态样本首次接地前暂停轮子离地惩罚，首次轮子或机身接地后恢复原有惩罚。原 5 s 回合时限、连续站稳成功条件和大倾角持续 3 s 的终止规则保持不变。

默认参数在 `config.py::get_recovery_training_reset_cfg()` 中设置。`--recovery-dynamic-probability 0` 可单独测静态起身，`1` 可单独测动态恢复；每个环境独立按概率抽样，单个小 batch 不保证恰好 80/20。训练配置中的 `recovery_reset.pose_pool` 保存全部数值和源路径映射，`--resume` 默认复用已保存的姿态池，不需要保留采集文件；要在续训时换数据，可显式使用 `--recovery-poses`。不含此字段的旧存档仍使用原单姿态复位，显式指定新来源后才启用新采样器。

TensorBoard 的 `recovery/metrics/` 中保留整体指标，并按以下路径细分：

- `static/success`、`dynamic/success`：按实际结束回合分组的成功率；对应 `recovery_duration_s` 为各组结束回合用时（含失败）。
- `static/front_down/success`、`dynamic/rear_down/success` 等：区分机身近水平 `level`、前倾 `front_down`、后倾 `rear_down`、翻面 `inverted` 和其他倾斜 `tilted` 的种子姿态。
- `pose/pose_leg_bb/success`、`pose/pose_leg_bb/recovery_duration_s` 等：按快照文件名（去掉 `.json`）记录每份姿态的成功率和恢复用时；原始 `reset_pos.json` 对应 `pose/reset_pos/`。显式来源中若有同名快照，会追加 `__0000` 等池内编号以区分。
- `difficulty/0/success` 等：各难度的成功率及 `recovery_duration_s`；`difficulty` 为本轮结束回合的平均难度。
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

此列表仅控制 R 键。训练只使用 `RESET_POSE_DIFFICULTIES` 的配置清单，或使用 `--recovery-poses` 显式指定的来源。
