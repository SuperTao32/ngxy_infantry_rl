# Jump

统一策略中的条件跳跃任务，只使用 `JUMP_20` / `JUMP_40` 两种跳高模式，独热编码分别为 `[1, 0]` / `[0, 1]`。高度模式与地形分别采样，每种跳高都训练平地跳跃和跳台阶；平地是地形类型，不占独热编码维度。各高度模式的概率、目标、距离表和奖励列表均按此顺序配置。原 flat 独热类别及其专用静止跳采样参数已移除。

`jump_modes.mode_probabilities` 控制两种跳高的采样权重；`jump_modes.platform_probabilities=[0.5, 0.5]` 分别控制 JUMP_20 / JUMP_40 采样台阶的概率，其余回合采样平地。`jump_modes.platform_height_levels_m=[[0.10, 0.15, 0.20], [0.25, 0.30, 0.35, 0.40]]` 配置各模式从易到难的台阶高度，单位 m。场景预建两条平地通道和七条台阶通道；课程只改变新回合的通道分配，不移动正在使用的台面。`platform_enabled` 可由课程关闭某种高度的全部台阶，此时该高度只采样平地。独热编码只表示跳高，不包含地形，地形由实际场景和 ToF 感知。

模式 one-hot 和四维相位输入由 `config.py::get_cfgs()` 中独立的 `jump_input_duration_s=[0.80, 1.10]` 配置，按 `JUMP_20` / `JUMP_40` 顺序分别指定 20 cm / 40 cm 的时长；同一高度的平地和台阶共用。从 jump 触发开始计时，两项输入按本回合模式的窗口同时归零。`jump_phase_durations_s=durations` 仍配置起跳等阶段，修改它不再改变输入时长。成功奖励在达标时结算，成功终止必须等本模式的输入结束；过倾、碰撞、越界等仍可提前终止。每个模式的回合超时取阶段总时长与该模式输入时长的较大值：输入较短时可继续等待落稳，输入较长时自动延长回合。旧标量配置仍对两种模式共用，缺少该字段时沿用阶段总时长。

```python
jump_phase_durations_s=durations,
jump_input_duration_s=[0.80, 1.10],  # [20 cm, 40 cm]，独热编码和相位共用，单位 s。
```

- `config.py`：模式比例、阶段时长、距离参考上下限、起跳距离表、奖励和课程。
- `buffers.py` / `reward_state.py`：回合状态、连续腾空确认、峰值与落地结算。
- `task.py`：阶段更新、命令、终止、奖励权重解析/课程合并/旧配置转换和公共环境 hook。
- `approach.py`：站稳确认、台阶助跑达速及保留物理状态的起跳位置切换。
- `rewards.py`：任务奖励。
- `terrain.py`：任务专用平台通道和模式分配，复用 `terrains` 的公共几何接口。
- `curriculum.py`：两个模式独立的台阶成功率窗口、升降级及检查点恢复。
- `geometry.py` / `height_reference.py` / `phase.py`：几何、轨迹与阶段编码。

jump 与其他任务共享观测、动作和 PPO。每回合先从静止出生在空旷跑道（默认 x=-6、y=-4，地面 z=0），只给行走速度和站立高度命令，jump 模式及阶段编码为零。台阶回合要求实际速度与命令误差不超过 0.15 m/s、双轮接地且姿态/高度稳定连续 0.15 s，再保留当前高度、姿态、关节状态和全部实际速度，平移到速度表及抖动给出的起跳位置，并输入模式 one-hot 与 t=0 阶段编码。

平地回合不使用 `min_forward_speeds_m_s`，准备阶段不检查前向达速或侧向/竖向速度容差；只连续确认双轮接地、机身无接触、姿态和高度稳定，再移到相应平地通道的 x=0 处起跳。平地起跳不查询速度—距离表、不加距离抖动，零速度、反向或表外速度均不会因此拒绝起跳。平台被课程关闭时，相应通道也按平地规则处理。命令仍由 `command_cfg.lin_vel_range` 采样，它只定义训练采样分布；可设为 `[0.0, 0.0]` 训练原地跳，平地不会再被最低速度强制抬高。跳跃期间平地和台阶回合均计算 `tracking_lin_vel` 奖励，跟踪当前指令前向速度；速度不参与平地跳高成功判定，零 yaw 跟踪以及起跳资格、姿态、高度和落稳条件继续按各自定义结算。

高度进度、合格起跳事件和起跳速度事件奖励要求首次双轮连续腾空达到 `takeoff_min_airborne_time_s`（默认 0.06 s）。腾空必须同时满足双轮无接触力、机身无接触，以及两轮底部各自高于局部地面 `takeoff_min_wheel_clearance_m`（默认 0.01 m）；台面高度也会扣除，卸载但仍贴地不算腾空。资格确认前只暂存该连续片段的峰值，合格时一次认可；不足最短时间便触地则不给这些奖励。高度进度统一由 `flight_height_progress` 结算，不再单独发放 `flight_peak_height`。峰值保持相对起跳平地的高度，不扣台面或出生时的机身高度；原始峰值仍完整保留用于成功判定和缺高惩罚。落稳成功和软着陆奖励也要求合格起跳。失败惩罚按首次合格腾空峰值计算，触地后站高不能减少惩罚。

`flight_height_excess` 惩罚平地和台阶回合首次合格腾空的新增超高量。两类地形共用本模式的上限 `ground_success_base_height_m + ground_base_height_excess_buffer_m`，缓冲默认 0.03 m；当前 JUMP_20 / JUMP_40 上限分别为 0.46 / 0.66 m，沿用起跳平地 z=0 的固定高度基准，不叠加台阶高度。成功高度与缓冲均在 `jump_modes` 中配置。累计超高为 `max(peak - 上限, 0) / base_height_target`，逐拍只扣新增量，默认权重 -50000，仍乘 dt；下降、重复旧峰值、触地后站高和二次起跳不重复扣罚，失稳不返还已扣部分。缓冲和惩罚不改变成功条件。日志为 `jump/rewards/flight_height_excess`。

首次任意轮或机身触地便冻结跳高与轮底成绩，轮底触及局部地面但接触力低于阈值时也会关闭本次跳跃。从首次触地当拍开始，任一轮仍未着地就按 `landing_airborne` 持续扣分（默认每秒 -10），包括单轮先落地、左右交替支撑及双轮再次离地；双轮均触地后停止。触地按接触力或轮底到局部地面的间隙 ≤0 判断，不套用起跳的 1 cm 腾空阈值，机身接触或失稳也不能免除惩罚。双轮有效支撑和连续站稳仍是落稳成功条件；台阶落稳允许短暂 baselink 碰撞，连续碰撞达到 `base_contact_termination_duration_s`（默认 0.20 s）仍由通用计时器判死亡，死亡优先于同拍成功，最终成功率不计入死亡回合。平地落稳仍要求机身无接触。单轮支撑不记为反弹，`rebound_seen` 仍只记录再次双轮离地。二次起跳不能补发起跳/峰值奖励或补足成功高度，也不会单独因此提前终止，旧存档的 `jump_termination.on_rebound` 字段仍不生效。下一回合 reset 后才可重新取得完整奖励。

落地正奖励（`soft_landing`、`landing_stability`、`landing_base_height_tracking`、`base_to_wheel_bottom_distance_tracking`）和任务成功均要求首次合格腾空的 `peak_clearance >= base_height_target`，目标来自 `clearance_targets_m`，当前 JUMP_20 / JUMP_40 分别为 0.45 / 0.63 m；落地后站高和二次起跳不能补足目标。原有落台支撑、平地轮底成功高度、姿态和速度条件继续生效。未达目标时仍正常计算落地冲击、加速度和悬轮等惩罚。

`flight_height_progress` 仅奖励首次合格腾空的新高度进度：`p = clamp((peak - min_distance_m) / (ground_success_base_height_m[mode] - min_distance_m), 0, 1)`，逐拍发放 `max(p - previous_p, 0)`。最低距离处为 0，本模式成功高度处为 1；平地和台阶使用相同上限，不叠加超高惩罚的缓冲。不足最短腾空时间、下降、重复经过旧高度、成功高度以上继续升高及二次起跳均不能额外领取；失稳时按同一公式一次追回已发进度。该项仍乘 dt，默认权重按 JUMP_20 / JUMP_40 为 `[1024.6376811594203, 1401.5873015873017]`，dt=0.01 s 时每次跳跃累计最多约 +10.25 / +14.02。权重由原峰值项 2000 与进度项 100 在各模式成功高度处的总收益折算，课程继续使用这一项。日志只输出 `jump/rewards/flight_height_progress`；旧存档的两项权重会在任务构造时自动合并，课程中只覆盖其中一项的旧配置也按累计覆盖关系转换，不重复计分。

`flight_airtime` 按首次有效腾空的实际累计秒数给分，最多奖励 `jump_phase_durations_s.flight` 秒。间隙暂时低于离地阈值不会重置预算，首次触地后也不能重开；时长不是 dt 整数倍时，最后一拍只发剩余比例。

`landing_base_height_tracking` 在首次触地后等待 `reward_cfg.landing_base_height_tracking_delay_s`（默认 0.20 s），再持续奖励 base_link 高度追踪。等待期内该项始终为 0，为落地缓冲留出时间。延迟从首次任意轮或机身触地开始，不因后续另一轮触地或反弹重置；按控制步向上取整，dt=0.01 s 时首次触地当拍为 t=0，到 t=0.20 s 才生效；设为 0 可取消延迟。实际高度为 `jump_base_height - landing_surface_height`，目标为当前 `commands[:, 2]`；平地和台阶都按相对落地面的真实高度跟踪，单轮抬高不会改变测量基准。奖励质量为 `exp(-max(abs(高度误差) - height_reference_tolerance_m, 0)**2 / height_reference_sigma)`，默认容差 0.02 m、sigma 0.0025 m²，再乘配置的权重和 dt。延迟结束后单轮有效支撑给半额，双轮有效支撑给满额；首次触地前、双轮反弹、机身撑地、错误落台和失稳不给奖励。保留原来的 `base_to_wheel_bottom_distance_tracking` 配置，默认权重仍为 0。日志为 `jump/rewards/landing_base_height_tracking`。

`landing_base_acceleration` 从首次任意轮或机身触地当拍起，持续惩罚 base_link 原点世界系三轴运动加速度平方和 `sum(((v_now - v_previous) / dt)**2)`。使用真实世界系线速度差分，不使用 IMU 比力或机体系速度；包含首次冲击，错误落台、失稳和触地后反弹也不能免除。默认权重 -0.01，100 m/s² 的加速度持续一个 0.01 s 控制拍扣 1；静止或匀速时为零。物理 reset 和保速切入 jump 时以当前速度重新起算，避免跨回合差分。

`flight_leg_angle` 仅在首次有效腾空时惩罚虚拟腿摆角超限：`mean(clamp(abs(leg_angle) - radians(flight_leg_angle_limit_deg), 0)**2)`，默认角度限值 10°、权重 -50。摆角相对机身定义，0 表示沿机身 -Z；分别计算左右腿后取均值，正负方向不会抵消。容差内无惩罚，起跳支撑期和触地后不生效。

`airborne_wheel_vel` 逐轮惩罚空转：`sum(airborne_i * wheel_vel_i**2)`，使用轮关节的真实角速度，单位 rad/s，默认权重 -0.01，乘控制周期 dt。每个轮子须同时满足无接触力和轮底高于局部地面 `takeoff_min_wheel_clearance_m`（默认 0.01 m）；台面高度会扣除，接地轮及卸载但仍贴地的轮子不扣分。jump 触发后单轮离地、首次腾空、落地后反弹和失稳当拍均生效，不依赖合格起跳或有效腾空奖励门；助跑不结算。双轮各以 10 rad/s 空转 0.4 s，累计扣 0.8。日志为 `jump/rewards/airborne_wheel_vel`。

`takeoff_wheel_push` 奖励计划 takeoff 期间接触轮向上的法向支撑力。力从机身坐标投影到世界竖直方向，仅累加接触轮的正向分量，再除以 `takeoff_wheel_push_force_reference_n`（默认 200 N）；权重默认 10，按控制周期 dt 累计，超过参考力仍线性增长。奖励沿用 `takeoff_gate`，仅在首次离地前、机身无接触且跳跃有效时生效；助跑、腾空、落地和二次起跳不计分。

两种模式共用 `get_cfgs()` 中的 `jump_height_reference`：`min_distance_m`（当前 0.22 m）、`max_distance_m`（当前 0.38 m）和独立的收腿时长 `retract_duration_s`（当前 0.20 s）。距离指 base 原点到左右轮底的平均竖直距离，独立于机身跳高和台阶高度。参考值在 `jump_phase_durations_s.takeoff` 期间从最低平滑升至最高，随后在 `retract_duration_s` 内从最高平滑降至最低，完成后保持最低；两段均使用五次平滑插值，边界速度和加速度为零。当前在 0.00 / 0.20 / 0.40 s 分别为 0.22 / 0.38 / 0.22 m，之后保持 0.22 m。改变 flight 时长不改变收腿速度；若收腿时长超过 flight，则收腿继续到 landing，相位和独热编码按独立的输入窗口归零，完整回合结束时参考命令终止。训练与交互评估使用同一函数，不再配置逐模式轨迹节点。旧存档缺少收腿时长时，沿用该存档的 flight 时长；缺少距离上下限时，从旧节点提取最低/最高值。

跳跃课程起始按 `JUMP_20:JUMP_40 = 1:2` 采样平地跳高；任务课程第 1000 轮进入 `with_platform`，开启台阶采样。开放台阶后，每种模式默认各有 50% 平地和 50% 台阶回合。两个模式独立从最低台阶开始：JUMP_20 为 10 → 15 → 20 cm，JUMP_40 为 25 → 30 → 35 → 40 cm。两种模式都从静止开始，台阶回合实际加速到目标速度，平地回合无助跑速度门槛。

`config.py::_get_curriculum_cfg()` 中的 `height_curriculum` 控制高度课程：

```python
"height_curriculum": {
    "enabled": True,
    "window_episodes": 100,
    "promote_threshold": 0.80,
    "demote_threshold": 0.40,
},
```

每种模式累计至少 100 个当前等级的台阶结束回合后，按本次完整结束批次计算成功率：≥80% 升一级，≤40% 降一级，中间保持等级。每次评估清空窗口，最低和最高等级均限幅；平地、未结束回合和课程阶段切换造成的强制截断不计入。助跑超时算台阶失败；成功须在任务结算时仍有效，结算前曾经落稳后失稳或异常终止也算失败。成功后剩余时间切到 locomotion，其后失败不改写本次台阶成绩。升降级立即更新后续 reset 的分配，已分配的旧回合继续使用原台阶；等级及分配代次共同隔离旧成绩，降回同高度时也不混用旧窗口。检查点保存等级、代次、成功数、回合数与上次评估成功率，旧检查点缺少该状态时从最低等级开始。关闭高度课程、关闭任务课程或评估模式时使用最终 20/40 cm 台阶，不自动升降级。跳高模式、奖励高度目标和起跳距离表沿用原配置。

日志 `jump/metrics/mode/{JUMP_20,JUMP_40}/height_curriculum/{level,height_m,success_rate,window_episodes}` 分别记录当前等级（从 1 开始）、下个回合的台阶高度、上次评估成功率及当前窗口已累计回合数。`platform_success`、`platform_height_m` 仅统计该模式实际结束的台阶回合，用于区别混合平地的模式汇总指标。

跳跃计时从移动完成后开始，加速时间不占用跳跃时长。相位输入为 `[sin(πp), cos(πp), sin(πp/2), cos(πp/2)]` 四维，`p = t / jump_input_duration_s[mode]`；默认 20 cm / 40 cm 分别按 0.80 s / 1.10 s 归一化。成功结算等到对应输入结束；若跳跃预算仍有剩余，则平移到 locomotion 平地区域继续采样，预算用完才 reset，失败仍可提前终止。切换不产生 done、不重置 GRU，保持动作延迟历史；下一拍观测前立即刷新 ToF 测距历史。助跑仅运行同一策略推理，不结算奖励，不参与 PPO 更新、GAE、观测归一化统计或跳跃课程样本计数。达速动作仍属于助跑；只有接收到模式 one-hot 与 t=0 阶段编码后产生的动作及后续跳跃轨迹进入训练。GRU 保留助跑结束时的隐状态，训练从该隐状态开始，不回放或反向传播助跑观测；物理 reset 的 done 仍会重置 GRU。没有 teacher 或独立 runner。

平地与台阶前向速度跟踪、零 yaw 跟踪、机身平衡、机身接触、腿部对称、关节速度、动作变化及腿部限位奖励在跳跃期间生效；跳跃专用奖励按各自阶段门控结算。权重由 `config.py::_get_reward_cfg()` 及课程配置提供。动作变化使用公共 `wheel_action_rate` / `leg_action_rate`，基础权重分别为 -1 / -0.1；旧存档的 `action_rate` 仍可加载。助跑专用的 `base_height` 和 `wheel_airborne` 已移除。准备 5 s 未达标则 reset，该准备回合不产生训练奖励或 PPO 样本；超时、台阶速度容差、站稳时间及出生点由 `config.py::get_approach_cfg()` 配置。日志的 `jump_started` 和 `approach_duration_s` 可区分准备未完成与跳跃失败。

`reward_cfg.reward_scales` 中的每一项都可按实际平地/台阶分别配置。以下为配置示例，具体数值可独立调整：

```python
"flight_wheel_clearance": {
    "without_platform": 100.0,  # 无台阶回合
    "with_platform": 300.0,     # 有台阶回合
},
"tracking_lin_vel": {
    "without_platform": [2.0, 3.0],  # JUMP_20、JUMP_40
    "with_platform": [5.0, 8.0],
},
```

统一标量和 `[JUMP_20, JUMP_40]` 列表继续对两类地形共用；初始地形字典须同时包含 `without_platform` 与 `with_platform`。选择依据是本回合锁存的 `landing_surface_height > 0`：混合采样的平地回合、课程移除台阶后的回合均使用 `without_platform`，低高度台阶和最终台阶均使用 `with_platform`，与当前计划阶段或输入编码是否归零无关。物理 reset 分配新通道后自动更新，助跑仍不结算奖励，`death` 不乘 dt，其余 jump 项继续乘 dt。默认 `flight_wheel_clearance` 已展开为两组 100，其余默认数值保持原状。

课程的 `targets.reward_scales` 支持相同格式，并可只填写一种地形，例如 `{"tracking_lin_vel": {"with_platform": 8.0}}`；另一组沿用已有权重。填写统一标量或逐模式列表则重新对两类地形共用。默认课程只列实际变化项，避免开启台阶时用重复的统一值覆盖独立配置；开启台阶阶段将平地和台阶共用的 `flight_height_excess` 权重调为 -100000。

回合统计保留 `jump/metrics/{指标}` 汇总，并同时输出 `jump/metrics/mode/JUMP_20/{指标}` 和 `jump/metrics/mode/JUMP_40/{指标}`。分模式指标包括 `success`、`peak_base_height_m`、`peak_wheel_clearance_m`、`jump_started` 和 `approach_duration_s`，沿用各汇总指标的统计口径。按 reset 前锁存的 `jump_mode` 筛选实际结束的回合，输入 one-hot 归零不影响分组；本拍没有结束回合的模式不输出该组，不用零值填充。logger 按每组实际回合样本求均值，两个模式的平地和台阶回合仍各自合并。

权重由 `config.py::_get_reward_cfg()` 及课程累计覆盖决定。除 `death` 不乘 dt 外，jump 事件也乘控制周期：默认 dt=0.01 s 时合格起跳 +1，合并后的 `flight_height_progress` 按模式累计最多约 +10.25 / +14.02，异常终止另扣 death=-100。任务失败仍按高度缺额扣分；高度进度在各模式成功高度处封顶，起跳速度项仍允许超目标增长，不能把高度进度的上限当成全部跳高收益的上限。

根配置默认按 locomotion:jump=2:1 分配环境，8192 个环境中约 2731 个用于 jump。助跑不产生有效训练样本，因此 jump 的 PPO 样本比例低于环境比例。任务课程仍以 8192 个环境的有效控制步折算轮数；1500 / 3000 的阶段阈值在 1/3 配额下至少需要约 4500 / 9000 个全局轮次，实际还会因助跑而延后。

所有任务和跳跃模式使用根配置的同一组电机参数：默认腿部 kp=50、kd=2，轮部 kd=0.3，腿部/轮部力矩限幅为 40/5。任务、课程和交互控制不能覆盖这些标称值。共享域随机化仍按统一规则调整各环境的实际增益与电机强度。

通过统一 `eval.py --terrain plane` 连续操纵同一个策略：1/2 选择 `JUMP_20`/`JUMP_40` 编码，J 在当前位置启动跳跃命令，完成后继续操纵，不触发任务成功/失败复位。入口和网络约定见 [框架说明](../../README.md)。
