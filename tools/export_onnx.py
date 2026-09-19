"""将 rsl_rl actor checkpoint 导出为适合实机部署的 ONNX。

- 从 checkpoint 读取 actor_state_dict，并用 cfgs.pkl 重建 MLPModel。
- 使用 actor.as_onnx() 导出确定性策略。
- 导出后执行 ONNX 结构检查，并用 ONNX Runtime 与 PyTorch 做数值对比。

示例：

    # 直接指定 checkpoint
    python tools/export_onnx.py \
        --checkpoint logs/infantry_jump_v6/version_0006/model_2000.pt

    # 按实验名和版本定位 checkpoint
    python tools/export_onnx.py \
        --exp-name infantry_jump_v6 --log-root logs --version 6

    # 指定输出文件
    python tools/export_onnx.py \
        --checkpoint logs/.../model_2000.pt --output build/policy.onnx
"""

from __future__ import annotations

import argparse
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from tensordict import TensorDict


# 让脚本可以从任意工作目录运行。
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rsl_rl.models import MLPModel  # noqa: E402
from experiments.genesis.wheel_leg_infantry.tools.run_utils import (  # noqa: E402
    load_run_configs,
    resolve_checkpoint,
    resolve_run_dir,
)


@dataclass(frozen=True)
class ActorSpec:
    """重建 actor 所需的结构信息。"""

    obs_dim: int
    num_actions: int
    hidden_dims: list[int]
    activation: str
    distribution_cfg: dict[str, Any]


def load_actor_state_dict(checkpoint_path: Path) -> dict[str, torch.Tensor]:
    """从 checkpoint 中读取 actor 权重。"""
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if "actor_state_dict" not in checkpoint:
        raise KeyError(f"Checkpoint 缺少 actor_state_dict: {checkpoint_path}")

    return checkpoint["actor_state_dict"]


def infer_obs_dim(state_dict: dict[str, torch.Tensor]) -> int:
    """从第一层 Linear 权重推断 policy observation 维度。"""
    linear_layers: list[tuple[int, torch.Tensor]] = []

    for key, value in state_dict.items():
        parts = key.split(".")
        if (
            len(parts) == 3
            and parts[0] == "mlp"
            and parts[1].isdigit()
            and parts[2] == "weight"
        ):
            linear_layers.append((int(parts[1]), value))

    if not linear_layers:
        raise ValueError(
            "actor_state_dict 中没有 mlp.<idx>.weight，无法推断输入维度"
        )

    linear_layers.sort(key=lambda item: item[0])
    first_weight = linear_layers[0][1]

    if first_weight.ndim != 2:
        raise ValueError(
            f"第一层 Linear 权重应为二维，实际 shape={tuple(first_weight.shape)}"
        )

    # Linear.weight shape = [out_features, in_features]
    return int(first_weight.shape[1])


def build_actor_spec(
    state_dict: dict[str, torch.Tensor],
    configs: dict[str, Any],
) -> ActorSpec:
    """根据 checkpoint 和训练配置生成 actor 结构描述。"""
    train_cfg = configs["train_cfg"]
    env_cfg = configs["env_cfg"]
    actor_cfg = train_cfg["actor"]

    model_class = actor_cfg.get("class_name", "MLPModel")
    if model_class != "MLPModel":
        raise NotImplementedError(
            f"当前导出脚本只支持 MLPModel，实际为 {model_class!r}"
        )

    actor_obs_group = train_cfg.get("obs_groups", {}).get("actor", ["policy"])
    if actor_obs_group != ["policy"]:
        raise NotImplementedError(
            "当前导出脚本要求 actor obs_groups == ['policy']，"
            f"实际为 {actor_obs_group!r}"
        )

    obs_dim = infer_obs_dim(state_dict)

    # 防止拿错 cfgs.pkl：配置中的观测维度应与 checkpoint 第一层完全一致。
    cfg_obs_dim = configs.get("obs_cfg", {}).get("num_policy_obs")
    if cfg_obs_dim is not None and int(cfg_obs_dim) != obs_dim:
        raise ValueError(
            "Checkpoint 与 cfgs.pkl 不匹配："
            f"checkpoint obs_dim={obs_dim}，"
            f"cfgs.pkl num_policy_obs={cfg_obs_dim}"
        )

    return ActorSpec(
        obs_dim=obs_dim,
        num_actions=int(env_cfg["num_actions"]),
        hidden_dims=list(actor_cfg["hidden_dims"]),
        activation=str(actor_cfg["activation"]),
        distribution_cfg=dict(actor_cfg["distribution_cfg"]),
    )


def build_actor(spec: ActorSpec) -> MLPModel:
    """按训练配置重建 actor。"""
    dummy_obs = TensorDict(
        {"policy": torch.zeros(1, spec.obs_dim, dtype=torch.float32)},
        batch_size=[1],
    )

    return MLPModel(
        dummy_obs,
        {"actor": ["policy"]},
        "actor",
        spec.num_actions,
        hidden_dims=spec.hidden_dims,
        activation=spec.activation,
        distribution_cfg=spec.distribution_cfg,
    )


def load_checkpoint_and_config(
    args: argparse.Namespace,
) -> tuple[Path, dict[str, Any]]:
    """定位 checkpoint，并读取对应 cfgs.pkl。"""
    if args.checkpoint is not None:
        checkpoint_path = args.checkpoint.expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint 不存在: {checkpoint_path}")

        config_path = checkpoint_path.parent / "cfgs.pkl"
        if not config_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint 同级目录缺少 cfgs.pkl: {config_path}"
            )

        return checkpoint_path, load_run_configs(checkpoint_path.parent)

    run_dir = resolve_run_dir(
        args.log_root,
        args.exp_name,
        args.version,
        require_checkpoint=True,
    )
    checkpoint_path = resolve_checkpoint(run_dir, args.checkpoint_iter)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint 不存在: {checkpoint_path}")

    return checkpoint_path, load_run_configs(run_dir)


def make_reference_inputs(obs_dim: int) -> dict[str, torch.Tensor]:
    """生成固定的测试输入，用于 PyTorch / ONNX 数值一致性检查。"""
    generator = torch.Generator().manual_seed(0)

    return {
        "zero": torch.zeros(1, obs_dim, dtype=torch.float32),
        "small": torch.rand(1, obs_dim, generator=generator) * 0.2 - 0.1,
        "unit": torch.rand(1, obs_dim, generator=generator) * 2.0 - 1.0,
    }


def get_reference_outputs(
    onnx_model: torch.nn.Module,
    inputs: dict[str, torch.Tensor],
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """先用 PyTorch 运行一次，作为 ONNX Runtime 的参考结果。"""
    references: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    with torch.inference_mode():
        for name, x in inputs.items():
            y = onnx_model(x).detach().cpu()
            references[name] = (x, y)

            print(
                f"[verify] torch {name:5s}: "
                f"shape={tuple(y.shape)}, "
                f"min={y.min().item():+.6f}, "
                f"max={y.max().item():+.6f}"
            )

    return references


def export_onnx(
    onnx_model: torch.nn.Module,
    output_path: Path,
    opset: int,
    verbose: bool,
) -> None:
    """导出固定 batch=1 的 ONNX。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        torch.onnx.export(
            onnx_model,
            onnx_model.get_dummy_inputs(),
            str(output_path),
            export_params=True,
            opset_version=opset,
            verbose=verbose,
            input_names=onnx_model.input_names,
            output_names=onnx_model.output_names,
            # 实机每次只推理一个机器人，不使用 dynamic batch。
            # 同时内联权重，方便 STM32 / ST Edge AI 使用单个 .onnx 文件。
            external_data=False,
        )


def verify_onnx(
    output_path: Path,
    references: dict[str, tuple[torch.Tensor, torch.Tensor]],
) -> None:
    """检查 ONNX 合法性，并验证其输出与 PyTorch 一致。"""
    import onnx

    model = onnx.load(str(output_path))
    onnx.checker.check_model(model)
    print("[verify] ONNX 结构校验通过。")

    print("[verify] ONNX 输入输出：")
    for kind, values in (("input", model.graph.input), ("output", model.graph.output)):
        for value in values:
            dims = value.type.tensor_type.shape.dim
            shape = [
                dim.dim_value if dim.dim_value else (dim.dim_param or "?")
                for dim in dims
            ]
            print(f"         {kind:6s} {value.name}: {shape}")

    try:
        import onnxruntime as ort
    except ImportError:
        print("[verify] 未安装 onnxruntime，跳过数值一致性检查。")
        return

    session = ort.InferenceSession(
        str(output_path),
        providers=["CPUExecutionProvider"],
    )

    if len(session.get_inputs()) != 1 or len(session.get_outputs()) != 1:
        raise RuntimeError(
            "当前部署模型应当只有一个输入和一个输出，"
            f"实际 inputs={len(session.get_inputs())}, "
            f"outputs={len(session.get_outputs())}"
        )

    input_name = session.get_inputs()[0].name

    for name, (x, expected) in references.items():
        result = session.run(None, {input_name: x.numpy()})[0]
        actual = torch.from_numpy(result)

        # 不只是打印误差；超出容差直接报错，避免错误模型进入实机部署。
        torch.testing.assert_close(
            actual,
            expected,
            rtol=1e-4,
            atol=1e-5,
        )

        max_diff = (actual - expected).abs().max().item()
        print(
            f"[verify] onnx  {name:5s}: "
            f"shape={tuple(actual.shape)}, max|diff|={max_diff:.3e}"
        )

    print("[verify] PyTorch 与 ONNX 数值一致。")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将 rsl_rl MLP actor checkpoint 导出为 ONNX。"
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path, help="model_*.pt 路径")
    source.add_argument("--exp-name", type=str, help="实验名")

    parser.add_argument("--log-root", type=str, default="logs")
    parser.add_argument(
        "--version",
        type=str,
        default=None,
        help="如 6 或 version_0006；默认取最新版本",
    )
    parser.add_argument(
        "--checkpoint-iter",
        type=int,
        default=None,
        help="checkpoint 迭代号；默认取最新 checkpoint",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="输出 .onnx；默认与 checkpoint 同名",
    )
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--verbose", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    checkpoint_path, configs = load_checkpoint_and_config(args)
    print(f"[export] checkpoint: {checkpoint_path}")

    state_dict = load_actor_state_dict(checkpoint_path)
    spec = build_actor_spec(state_dict, configs)

    print(
        "[export] actor: "
        f"obs_dim={spec.obs_dim}, "
        f"num_actions={spec.num_actions}, "
        f"hidden_dims={spec.hidden_dims}, "
        f"activation={spec.activation}, "
        f"distribution={spec.distribution_cfg.get('class_name')}"
    )

    actor = build_actor(spec)
    actor.load_state_dict(state_dict, strict=True)
    actor.to("cpu").eval()
    print("[export] actor 权重加载完成（strict=True）。")

    # as_onnx() 负责把训练时的分布策略转换成确定性部署输出。
    onnx_model = actor.as_onnx(verbose=args.verbose).to("cpu").eval()

    test_inputs = make_reference_inputs(spec.obs_dim)
    references = get_reference_outputs(onnx_model, test_inputs)

    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else checkpoint_path.with_suffix(".onnx")
    )

    export_onnx(
        onnx_model=onnx_model,
        output_path=output_path,
        opset=args.opset,
        verbose=args.verbose,
    )
    print(f"[export] 已导出: {output_path}")

    verify_onnx(output_path, references)
    print("[done] ONNX 导出与验证全部完成。")


if __name__ == "__main__":
    main()
