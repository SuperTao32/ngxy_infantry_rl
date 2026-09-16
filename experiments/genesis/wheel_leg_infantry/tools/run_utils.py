"""训练版本的创建、参数保存，以及 eval 自动发现工具。"""

from __future__ import annotations

import json
import pickle
import platform
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping


RUN_DIR_PATTERN = re.compile(r"^version_(\d+)$")
CHECKPOINT_PATTERN = re.compile(r"^model_(\d+)\.pt$")
CONFIG_FILENAME = "cfgs.pkl"


def load_runner_class():
    """检查 rsl-rl 主版本，并延迟导入 ``OnPolicyRunner``。"""
    try:
        major_version = int(metadata.version("rsl-rl-lib").split(".")[0])
        if major_version < 5:
            raise ImportError
    except (metadata.PackageNotFoundError, ImportError, ValueError) as exc:
        raise ImportError("Please install 'rsl-rl-lib>=5.0.0'.") from exc

    from rsl_rl.runners import OnPolicyRunner

    return OnPolicyRunner


@dataclass(frozen=True)
class ResumePlan:
    """一次续训所需的源运行目录、checkpoint 和迭代位置。"""

    source_run_dir: Path
    checkpoint_path: Path
    checkpoint_iteration: int

    @property
    def next_iteration(self) -> int:
        """checkpoint 已完成，续训应从下一轮开始。"""
        return self.checkpoint_iteration + 1


def add_resume_arguments(parser: Any) -> None:
    """给任意 task 的训练参数解析器添加统一的续训选项。"""
    parser.add_argument(
        "--resume",
        nargs="?",
        const="latest",
        default=None,
        metavar="VERSION",
        help="resume from a version (for example 0 or version_0000); omit VERSION to use latest",
    )
    parser.add_argument("--checkpoint", type=int, default=None, help="checkpoint number; default: latest")
    parser.add_argument(
        "--resume-config",
        choices=("saved", "current"),
        default="saved",
        help="use the source run's saved config (default) or the task's current config",
    )


def create_versioned_run_dir(log_root: str | Path, exp_name: str) -> Path:
    """为一次训练原子地创建下一个 ``version_NNNN`` 目录。"""
    experiment_dir = Path(log_root) / exp_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    versions = [_version_number(path) for path in experiment_dir.iterdir() if path.is_dir()]
    next_version = max((number for number in versions if number is not None), default=-1) + 1

    # exist_ok=False 保证并发启动训练时不会共用目录：抢到同一版本号失败的
    # 进程会继续尝试下一个编号，也不会删除另一进程已经创建的日志。
    while True:
        run_dir = experiment_dir / f"version_{next_version:04d}"
        try:
            run_dir.mkdir(exist_ok=False)
            return run_dir
        except FileExistsError:
            next_version += 1


def resolve_run_dir(
    log_root: str | Path,
    exp_name: str,
    version: str | int | None = None,
    *,
    require_checkpoint: bool = False,
) -> Path:
    """解析指定版本；未指定时返回编号最大的可用训练版本。

    保留对旧式 ``logs/<experiment>`` 非版本化目录的兼容。自动选择时，
    ``require_checkpoint=True`` 会跳过只有配置、没有模型的中断训练。
    """
    experiment_dir = Path(log_root) / exp_name
    if not experiment_dir.is_dir():
        raise FileNotFoundError(f"Experiment log directory does not exist: {experiment_dir}")

    if version is not None:
        version_name = str(version)
        if version_name.isdigit():
            version_name = f"version_{int(version_name):04d}"
        run_dir = experiment_dir / version_name
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Training version does not exist: {run_dir}")
        _validate_run_dir(run_dir, require_checkpoint=require_checkpoint)
        return run_dir

    # 只识别 version_数字，其他人工创建的目录不会影响“最新版本”的判断。
    candidates = sorted(
        (
            (number, path)
            for path in experiment_dir.iterdir()
            if path.is_dir() and (number := _version_number(path)) is not None
        ),
        reverse=True,
    )
    for _, run_dir in candidates:
        if _is_usable_run(run_dir, require_checkpoint=require_checkpoint):
            return run_dir

    # 新版本目录都不可用时，再尝试原来的 logs/<experiment>/cfgs.pkl 布局。
    if _is_usable_run(experiment_dir, require_checkpoint=require_checkpoint):
        return experiment_dir

    requirement = "configuration and checkpoint" if require_checkpoint else "configuration"
    raise FileNotFoundError(f"No run with a valid {requirement} found in: {experiment_dir}")


def resolve_checkpoint(run_dir: str | Path, checkpoint: int | None = None) -> Path:
    """解析指定 checkpoint；未指定时返回编号最大的 ``model_N.pt``。"""
    run_dir = Path(run_dir)
    if checkpoint is not None:
        checkpoint_path = run_dir / f"model_{checkpoint}.pt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
        return checkpoint_path

    checkpoints = sorted(
        (
            (number, path)
            for path in run_dir.iterdir()
            if path.is_file() and (number := _checkpoint_number(path)) is not None
        ),
        reverse=True,
    )
    if not checkpoints:
        raise FileNotFoundError(f"No model_*.pt checkpoint found in: {run_dir}")
    return checkpoints[0][1]


def resolve_recorded_run(
    source: Mapping[str, Any], log_root: str | Path
) -> tuple[Path, Path]:
    """在当前日志根目录或保存的原路径中定位来源，保留原版本和 checkpoint。

    训练记录包含绝对路径。日志复制到另一台机器后，优先按相同的实验名、
    版本目录和模型文件名重定位；不自动替换为最新模型。
    """
    saved_run_dir = Path(source["run_dir"])
    saved_checkpoint = Path(source["checkpoint"])
    if _version_number(saved_run_dir) is not None:
        relative_run_dir = Path(saved_run_dir.parent.name) / saved_run_dir.name
    else:
        relative_run_dir = Path(saved_run_dir.name)
    relocated_run_dir = Path(log_root) / relative_run_dir
    candidates = (
        (relocated_run_dir, relocated_run_dir / saved_checkpoint.name),
        (saved_run_dir, saved_checkpoint),
    )
    for run_dir, checkpoint_path in candidates:
        if (run_dir / CONFIG_FILENAME).is_file() and checkpoint_path.is_file():
            return run_dir, checkpoint_path

    checked = "\n".join(
        f"  config: {run_dir / CONFIG_FILENAME}; checkpoint: {checkpoint_path}"
        for run_dir, checkpoint_path in candidates
    )
    raise FileNotFoundError(
        f"Recorded source run is unavailable. Copy {CONFIG_FILENAME} and "
        f"{saved_checkpoint.name} from the original source run to {relocated_run_dir}, "
        f"or restore the recorded source paths. Checked:\n{checked}"
    )


def resolve_resume_plan(
    log_root: str | Path,
    exp_name: str,
    version: str | int | None = None,
    checkpoint: int | None = None,
) -> ResumePlan:
    """解析续训来源；省略版本或 checkpoint 时分别选择最新可用项。"""
    if version == "latest":
        version = None
    source_run_dir = resolve_run_dir(log_root, exp_name, version, require_checkpoint=True)
    checkpoint_path = resolve_checkpoint(source_run_dir, checkpoint)
    checkpoint_iteration = _checkpoint_number(checkpoint_path)
    if checkpoint_iteration is None:
        raise ValueError(f"Cannot determine checkpoint iteration from: {checkpoint_path}")
    return ResumePlan(source_run_dir, checkpoint_path, checkpoint_iteration)


def restore_training_state(runner: Any, env: Any, plan: ResumePlan, target_iterations: int) -> int:
    """恢复 RSL-RL 状态和环境课程进度，返回还需要训练的轮数。

    ``target_iterations`` 使用总目标轮数语义。例如 checkpoint 5800 已完成
    0..5800 共 5801 轮，目标为 9001 时会继续 5801..9000 共 3200 轮。

    带课程的环境应提供 ``set_training_iteration(iteration)``，在该方法中
    应用对应课程阶段并重置环境；没有课程的环境可以不提供该方法。
    """
    target_iterations = int(target_iterations)
    if target_iterations <= plan.next_iteration:
        raise ValueError(
            f"target_iterations must be greater than {plan.next_iteration} "
            f"when resuming from checkpoint {plan.checkpoint_iteration}"
        )

    load_kwargs = {}
    if (device := getattr(runner, "device", None)) is not None:
        load_kwargs["map_location"] = device
    runner.load(str(plan.checkpoint_path), **load_kwargs)

    restored_iteration = int(runner.current_learning_iteration)
    if restored_iteration != plan.checkpoint_iteration:
        raise ValueError(
            f"Checkpoint filename says iteration {plan.checkpoint_iteration}, "
            f"but its saved state says {restored_iteration}: {plan.checkpoint_path}"
        )

    # RSL-RL 保存的是刚完成的迭代号；learn() 从 current_learning_iteration
    # 开始，因此这里前移一轮，避免重复训练并覆盖源编号。
    runner.current_learning_iteration = plan.next_iteration

    set_training_iteration = getattr(env, "set_training_iteration", None)
    if callable(set_training_iteration):
        set_training_iteration(plan.next_iteration)
    elif getattr(getattr(env, "curriculum", None), "enabled", False):
        raise TypeError(
            "A curriculum-enabled environment must implement "
            "set_training_iteration(iteration) before it can resume training"
        )

    return target_iterations - plan.next_iteration


def save_run_artifacts(
    run_dir: str | Path,
    configs: Mapping[str, Any],
    arguments: Mapping[str, Any],
) -> None:
    """同时保存可精确恢复的 pickle、便于查看的 JSON 和运行元数据。"""
    run_dir = Path(run_dir)
    # schema_version 用于以后升级配置结构时继续兼容历史训练结果。
    payload = {"schema_version": 2, **dict(configs)}

    with (run_dir / CONFIG_FILENAME).open("wb") as file:
        pickle.dump(payload, file)
    with (run_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, default=_json_default)

    metadata = _collect_metadata(run_dir, arguments)
    with (run_dir / "run_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2, default=_json_default)


def load_run_configs(run_dir: str | Path) -> dict[str, Any]:
    """加载当前字典格式，同时兼容历史的五项或六项列表格式。"""
    config_path = Path(run_dir) / CONFIG_FILENAME
    with config_path.open("rb") as file:
        payload = pickle.load(file)

    if isinstance(payload, dict):
        required = {"env_cfg", "obs_cfg", "reward_cfg", "command_cfg", "train_cfg"}
        missing = required.difference(payload)
        if missing:
            raise ValueError(f"Config is missing required sections {sorted(missing)}: {config_path}")
        payload.setdefault("curriculum_cfg", {"enabled": False, "stages": []})
        return payload

    # 最早的日志没有 curriculum_cfg，读取时补成默认关闭状态。
    if isinstance(payload, (list, tuple)) and len(payload) == 5:
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = payload
        return {
            "schema_version": 1,
            "env_cfg": env_cfg,
            "obs_cfg": obs_cfg,
            "reward_cfg": reward_cfg,
            "command_cfg": command_cfg,
            "curriculum_cfg": {"enabled": False, "stages": []},
            "train_cfg": train_cfg,
        }

    if isinstance(payload, (list, tuple)) and len(payload) == 6:
        env_cfg, obs_cfg, reward_cfg, command_cfg, curriculum_cfg, train_cfg = payload
        return {
            "schema_version": 2,
            "env_cfg": env_cfg,
            "obs_cfg": obs_cfg,
            "reward_cfg": reward_cfg,
            "command_cfg": command_cfg,
            "curriculum_cfg": curriculum_cfg,
            "train_cfg": train_cfg,
        }

    raise ValueError(f"Unsupported config format in: {config_path}")


def _collect_metadata(run_dir: Path, arguments: Mapping[str, Any]) -> dict[str, Any]:
    # 配置决定实验参数；提交号、dirty 状态和依赖版本用于定位代码运行环境。
    return {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "run_dir": str(run_dir.resolve()),
        "arguments": dict(arguments),
        "command": sys.argv,
        "python": sys.version,
        "platform": platform.platform(),
        "packages": _package_versions("genesis-world", "rsl-rl-lib", "torch"),
        "git_commit": _git_output("rev-parse", "HEAD"),
        "git_dirty": bool(_git_output("status", "--porcelain")),
    }


def _git_output(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2,
        ).strip()
    except (FileNotFoundError, subprocess.SubprocessError):
        return None


def _package_versions(*names: str) -> dict[str, str | None]:
    versions = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _validate_run_dir(run_dir: Path, *, require_checkpoint: bool) -> None:
    if not (run_dir / CONFIG_FILENAME).is_file():
        raise FileNotFoundError(f"Run configuration does not exist: {run_dir / CONFIG_FILENAME}")
    if require_checkpoint and not any(
        path.is_file() and _checkpoint_number(path) is not None for path in run_dir.iterdir()
    ):
        raise FileNotFoundError(f"No model_*.pt checkpoint found in: {run_dir}")


def _is_usable_run(run_dir: Path, *, require_checkpoint: bool) -> bool:
    try:
        _validate_run_dir(run_dir, require_checkpoint=require_checkpoint)
    except FileNotFoundError:
        return False
    return True


def _version_number(path: Path) -> int | None:
    match = RUN_DIR_PATTERN.fullmatch(path.name)
    return int(match.group(1)) if match else None


def _checkpoint_number(path: Path) -> int | None:
    match = CHECKPOINT_PATTERN.fullmatch(path.name)
    return int(match.group(1)) if match else None


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    return repr(value)
