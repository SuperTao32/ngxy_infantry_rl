"""导出系统辨识使用的实测观测量及拟合前后的仿真曲线。"""

from pathlib import Path

from mujoco import sysid


def save_fit_curves(out, sequences, residual_fn, initial_params, fitted_params):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    _, initial, _ = residual_fn(initial_params.as_nominal_vector(), initial_params, return_pred_all=True)
    _, fitted, measured = residual_fn(fitted_params.as_vector(), fitted_params, return_pred_all=True)

    units = {sysid.SignalType.MjStateQPos: "rad", sysid.SignalType.MjStateQVel: "rad/s"}
    trajectories = zip(sequences.sequence_name, measured, initial, fitted, strict=True)
    for index, (name, (observed,), (before,), (after,)) in enumerate(trajectories, start=1):
        signals = observed.signal_mapping
        figure = Figure(figsize=(10, 2.5 * len(signals)), dpi=140, layout="constrained")
        axes = figure.subplots(len(signals), sharex=True, squeeze=False)[:, 0]
        figure.suptitle(f"{sequences.name} / {name}", parse_math=False)
        for axis, (signal, (kind, _)) in zip(axes, signals.items(), strict=True):
            for series, label, color, linestyle in (
                (observed, "Measured (fit input)", "#20252b", "-"),
                (before, "Before fitting", "#d55e00", ":"),
                (after, "After fitting", "#0072b2", "--"),
            ):
                _, columns = series.get_indices(signal)
                axis.plot(series.times, series.data[:, columns[0]], label=label, color=color, linestyle=linestyle)
            axis.set_title(signal, loc="left", parse_math=False)
            axis.set_ylabel(units[kind])
            axis.grid(alpha=0.25)
        axes[0].legend(ncols=3)
        axes[-1].set_xlabel("Time (s)")

        path = out / f"fit_curves_{index:02d}_{Path(name).name}.png"
        FigureCanvasAgg(figure).print_png(path)
        print(f"[plot] {path}", flush=True)
