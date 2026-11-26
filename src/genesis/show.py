"""悬挂轮腿机器人，交互调整电机目标并观察闭链关节。"""

import argparse
import math
from pathlib import Path
import xml.etree.ElementTree as ET

from .utils.gas_spring import gas_spring_force


def suspended_mjcf(path, base_pos):
    """在内存中移除自由基座，保留 MJCF 的闭链和网格。"""
    tree = ET.parse(path)
    base = tree.find("./worldbody/body[@name='base_link']")
    base.remove(base.find("freejoint[@name='floating_base']"))
    base.set("pos", " ".join(map(str, base_pos)))
    compiler = tree.find("compiler")
    for attr in ("assetdir", "meshdir", "texturedir"):
        directory = compiler.get(attr, compiler.get("assetdir", ""))
        compiler.set(attr, str(path.parent / directory))
    return ET.tostring(tree.getroot(), encoding="unicode")


class SuspendedRobot:
    def __init__(self, scene, robot, cfg, initial_state):
        import genesis as gs
        import numpy as np

        self.scene = scene
        self.robot = robot
        self.active_names = (*cfg["joint_names"], *cfg["wheel_names"])
        joints = [j for j in robot.joints if j.type in (gs.JOINT_TYPE.REVOLUTE, gs.JOINT_TYPE.PRISMATIC)]
        self.dofs = {j.name: j.dof_start - robot.dof_start for j in joints}
        self.passive = [(j.name, j.type == gs.JOINT_TYPE.PRISMATIC) for j in joints if j.name not in self.active_names]
        self.springs = [self.dofs[name] for name in cfg["spring_names"]]
        self.spring_cfg = {
            name: cfg[f"gas_spring_{name}"]
            for name in ("preload_force", "stiffness", "quadratic_stiffness", "damping", "max_compression")
        }

        with np.load(initial_state, allow_pickle=False) as saved:
            names, positions = saved["joint_names"], saved["qpos"]
            if names.ndim != 1 or positions.shape != names.shape or not np.isfinite(positions).all():
                raise ValueError(f"Invalid suspended joint positions in {initial_state}")
            q_by_name = dict(zip(names, positions, strict=True))
        q = robot.get_dofs_position()
        for name, index in self.dofs.items():
            q[index] = float(q_by_name[name])
        robot.set_dofs_position(q, zero_velocity=True)
        scene.reset(scene.get_state())

        self.active_dofs = [self.dofs[name] for name in self.active_names]
        lo, hi = robot.get_dofs_limit(self.active_dofs)
        self.position_limits = list(zip(lo.tolist(), hi.tolist(), strict=True))
        self._set_gains(cfg, cfg["joint_names"], "joint")
        self._set_gains(cfg, cfg["wheel_names"], "wheel")
        positions, _, _ = self.read()
        self.initial_targets = tuple(
            (0, positions[index]) if name in cfg["joint_names"] else (1, 0.0)
            for name, index in zip(self.active_names, self.active_dofs, strict=True)
        )
        self.targets = self.initial_targets

    def _set_gains(self, cfg, names, group):
        from .utils.tensor_utils import as_gain_tensor

        indices = [self.dofs[name] for name in names]

        def gain(key):
            return as_gain_tensor(cfg[key], len(indices), device="cpu")

        kp = gain("joint_kp")
        if group == "wheel":
            # 训练中的车轮只调速度。调试位置模式时使用腿电机的平均位置增益。
            kp = kp.mean().expand(len(indices))
        self.robot.set_dofs_kp(kp.tolist(), indices)
        self.robot.set_dofs_kv(gain(f"{group}_kd").tolist(), indices)
        limit = gain(f"{group}_force_limit")
        self.robot.set_dofs_force_range((-limit).tolist(), limit.tolist(), indices)

    def read(self):
        # 物理线程发布整帧读数，界面只读取这个快照。
        spring_forces = self.robot.get_dofs_control_force(self.springs).tolist()
        self.measured = (
            tuple(self.robot.get_dofs_position().tolist()),
            tuple(self.robot.get_dofs_velocity().tolist()),
            dict(zip(self.springs, spring_forces, strict=True)),
        )
        return self.measured

    def step(self):
        targets = self.targets
        for mode, control in enumerate((self.robot.control_dofs_position, self.robot.control_dofs_velocity)):
            selected = [(index, value) for index, (choice, value) in zip(self.active_dofs, targets, strict=True) if choice == mode]
            if selected:
                indices, values = zip(*selected)
                control(values, indices)

        q = self.robot.get_dofs_position(self.springs)
        dq = self.robot.get_dofs_velocity(self.springs)
        force = gas_spring_force(q, dq, **self.spring_cfg)
        self.robot.control_dofs_force(force, self.springs)
        self.scene.step()
        self.read()

    def draw_panel(self, imgui, reset_scene):
        view = imgui.get_main_viewport()
        width = 590
        imgui.set_next_window_pos((max(0, view.work_size.x - width - 12), 12), imgui.Cond_.first_use_ever)
        imgui.set_next_window_size((width, min(670, view.work_size.y - 24)), imgui.Cond_.first_use_ever)
        visible, _ = imgui.begin("Suspended Robot")
        if visible:
            if imgui.button("Reset pose and targets"):
                self.targets = self.initial_targets
                reset_scene()
            imgui.same_line()
            imgui.text("Ctrl+click to enter a target.")
            q, dq, spring_forces = self.measured
            imgui.separator_text("Motors (targets in rad or rad/s)")
            self._draw_motors(imgui, q, dq)
            imgui.separator_text("Passive joints")
            if _begin_table(imgui, "passive", ("Joint", "Position", "Converted", "Spring force (N)")):
                for name, linear in self.passive:
                    index = self.dofs[name]
                    value = q[index]
                    _cell(imgui, name)
                    _cell(imgui, f"{value:+.4f} {'m' if linear else 'rad'}")
                    _cell(imgui, f"{value * 1000:+.2f} mm" if linear else f"{math.degrees(value):+.2f} deg")
                    force = spring_forces.get(index)
                    _cell(imgui, f"{abs(force):.2f}" if force is not None else "")
                imgui.end_table()
        imgui.end()

    def _draw_motors(self, imgui, q, dq):
        if not _begin_table(imgui, "motors", ("Joint", "Mode", "Target", "q (rad)", "dq (rad/s)")):
            return
        targets = list(self.targets)
        for row, (name, index, (lo, hi)) in enumerate(zip(self.active_names, self.active_dofs, self.position_limits, strict=True)):
            mode, target = targets[row]
            _cell(imgui, name.removesuffix("_joint"))
            imgui.table_next_column()
            imgui.set_next_item_width(-1)
            changed, mode = imgui.combo(f"##mode_{name}", mode, ["Angle", "Speed"])
            if changed:
                target = 0.0 if mode else min(max(q[index], lo), hi)
            bounds = (lo, hi) if mode == 0 and math.isfinite(lo) and math.isfinite(hi) else (0.0, 0.0)
            imgui.table_next_column()
            imgui.set_next_item_width(-1)
            edited, value = imgui.drag_float(
                f"##target_{name}", target, 0.05 if mode else 0.005, *bounds,
                "%.3f" if mode else "%.4f", imgui.SliderFlags_.clamp_on_input,
            )
            if edited and math.isfinite(value):
                target = value
            targets[row] = (mode, target)
            _cell(imgui, f"{q[index]:+.4f}")
            _cell(imgui, f"{dq[index]:+.3f}")
        imgui.end_table()
        self.targets = tuple(targets)


def _begin_table(imgui, name, columns):
    if not imgui.begin_table(name, len(columns), imgui.TableFlags_.row_bg | imgui.TableFlags_.borders_inner_v):
        return False
    weights = {"Joint": 1.65, "dq (rad/s)": 1.2}
    for col in columns:
        imgui.table_setup_column(col, imgui.TableColumnFlags_.width_stretch, weights.get(col, 1.0))
    imgui.table_headers_row()
    return True


def _cell(imgui, text):
    imgui.table_next_column()
    imgui.text_unformatted(text)


def build_scene(gs, cfg, *, show_viewer):
    from tools.sysid.settle_suspended_robot import CFG

    model_path = Path(__file__).resolve().parents[2] / cfg["robot_mjcf"]
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=cfg["control_dt"], substeps=20),
        rigid_options=gs.options.RigidOptions(enable_self_collision=False, tolerance=1e-5),
        viewer_options=gs.options.ViewerOptions(
            enable_gui=show_viewer, res=(1440, 900),
            camera_pos=(1.8, -2.8, 2.0), camera_lookat=(0.0, 0.0, 0.8), camera_fov=37,
        ),
        profiling_options=gs.options.ProfilingOptions(show_FPS=False),
        show_viewer=show_viewer,
    )
    scene.add_entity(gs.morphs.Plane())
    robot = scene.add_entity(gs.morphs.MJCF(file=suspended_mjcf(model_path, CFG.base_pos)))
    scene.build()
    return SuspendedRobot(scene, robot, cfg, CFG.out)


def main(argv=None):
    parser = argparse.ArgumentParser(description="在 Genesis 窗口控制悬挂机器人并观察被动关节。")
    parser.add_argument("--cpu", action="store_true", help="使用 CPU 后端")
    parser.add_argument("--steps", type=int, help="执行指定控制步数后退出")
    parser.add_argument("--no-vis", action="store_true", help="关闭可视化窗口，需同时指定 --steps")
    args = parser.parse_args(argv)
    if args.steps is not None and args.steps < 1:
        parser.error("--steps 必须大于 0")
    if args.no_vis and args.steps is None:
        parser.error("--no-vis 需要 --steps")

    import genesis as gs

    from .config import get_env_cfg

    gs.init(backend=gs.cpu if args.cpu else gs.gpu, logging_level="warning", performance_mode=True)
    try:
        sim = build_scene(gs, get_env_cfg(), show_viewer=not args.no_vis)
        viewer = sim.scene.viewer
        if viewer is not None:
            from genesis.ext.pyrender.overlay.plugin import ImGuiOverlayPlugin

            overlay = next(plugin for plugin in viewer.plugins if isinstance(plugin, ImGuiOverlayPlugin))
            overlay.register_panel(lambda imgui: sim.draw_panel(imgui, overlay.interactive_scene.reset), section="overlay")
        steps = 0
        while (args.steps is None or steps < args.steps) and (viewer is None or viewer.is_alive()):
            sim.step()
            steps += 1
            if sim.scene.rigid_solver.get_error_envs_mask().any():
                raise RuntimeError("Genesis reported a physics solver error.")
    except KeyboardInterrupt:
        pass
    finally:
        gs.destroy()


if __name__ == "__main__":
    main()
