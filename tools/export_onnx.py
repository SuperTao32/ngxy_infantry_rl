"""将 rsl_rl actor checkpoint 导出为适合实机部署的 ONNX。

- 从 checkpoint 读取 actor_state_dict，并用 cfgs.pkl 重建 RNNModel / MLPModel。
- 使用 actor.as_onnx() 导出确定性策略。
- 保留观测归一化；循环策略显式输入、输出隐状态。
- 导出后执行 ONNX 结构检查，并用 ONNX Runtime 与 PyTorch 做单步及连续多步数值对比。

默认 GRU 部署接口：obs [1, obs_dim]、h_in [num_layers, 1, hidden_dim]
→ actions [1, num_actions]、h_out [num_layers, 1, hidden_dim]。
首次推理及回合重置时将 h_in 清零，之后每步将 h_out 作为下一步的 h_in。
LSTM 还需同样传递 c_in / c_out；MLP 仅有 obs / actions。

示例：

    # 直接指定 checkpoint
    python tools/export_onnx.py \
        --checkpoint logs/locomotion_only/version_0002/model_200.pt

    # 按实验名和版本定位 checkpoint
    python tools/export_onnx.py \
        --exp-name locomotion_only --log-root logs --version 2

    # 指定输出文件
    python tools/export_onnx.py \
        --checkpoint logs/.../model_2000.pt --output build/policy.onnx
"""

from __future__ import annotations

import argparse
import sys
import warnings
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from tensordict import TensorDict


# 让脚本可以从任意工作目录运行。
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rsl_rl.models import MLPModel, RNNModel  # noqa: E402
from src.genesis.wheel_leg_infantry.tools.run_utils import (  # noqa: E402
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
    distribution_cfg: dict[str, Any] | None
    model_class: str
    obs_normalization: bool
    rnn_type: str = "lstm"
    rnn_hidden_dim: int = 256
    rnn_num_layers: int = 1


@dataclass(frozen=True)
class ReferenceCase:
    inputs: tuple[torch.Tensor, ...]
    outputs: tuple[torch.Tensor, ...]
    # 连续步骤须使用 ONNX 自己的上一时刻输出，验证状态反馈后的累积误差。
    carry_state: bool = False


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
    """从 RNN 的输入权重或 MLP 第一层权重推断观测维度。"""
    rnn_weight = state_dict.get("rnn.rnn.weight_ih_l0")
    if rnn_weight is not None:
        if rnn_weight.ndim != 2:
            raise ValueError(f"RNN 输入权重应为二维，实际 shape={tuple(rnn_weight.shape)}")
        return int(rnn_weight.shape[1])

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
    # TrajectoryRNNModel 只改变带 masks 的训练路径，部署推理与 RNNModel 相同。
    if model_class == "src.genesis.wheel_leg_infantry.trajectory_ppo:TrajectoryRNNModel":
        model_class = "RNNModel"
    if model_class not in {"MLPModel", "RNNModel"}:
        raise NotImplementedError(
            f"当前导出脚本支持 MLPModel / RNNModel，实际为 {model_class!r}"
        )

    actor_obs_group = train_cfg.get("obs_groups", {}).get("actor", ["policy"])
    if actor_obs_group != ["policy"]:
        raise NotImplementedError(
            "当前导出脚本要求 actor obs_groups == ['policy']，"
            f"实际为 {actor_obs_group!r}"
        )

    obs_dim = infer_obs_dim(state_dict)

    # 防止拿错 cfgs.pkl：配置中的观测维度应与 checkpoint 输入层完全一致。
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
        distribution_cfg=deepcopy(actor_cfg.get("distribution_cfg")),
        model_class=model_class,
        obs_normalization=bool(actor_cfg.get("obs_normalization", False)),
        rnn_type=str(actor_cfg.get("rnn_type", "lstm")),
        rnn_hidden_dim=int(actor_cfg.get("rnn_hidden_dim", 256)),
        rnn_num_layers=int(actor_cfg.get("rnn_num_layers", 1)),
    )


def build_actor(spec: ActorSpec) -> MLPModel:
    """按训练配置重建 actor。"""
    dummy_obs = TensorDict(
        {"policy": torch.zeros(1, spec.obs_dim, dtype=torch.float32)},
        batch_size=[1],
    )

    model_class = RNNModel if spec.model_class == "RNNModel" else MLPModel
    recurrent_kwargs = {}
    if spec.model_class == "RNNModel":
        recurrent_kwargs = dict(
            rnn_type=spec.rnn_type,
            rnn_hidden_dim=spec.rnn_hidden_dim,
            rnn_num_layers=spec.rnn_num_layers,
        )

    return model_class(
        dummy_obs,
        {"actor": ["policy"]},
        "actor",
        spec.num_actions,
        hidden_dims=spec.hidden_dims,
        activation=spec.activation,
        obs_normalization=spec.obs_normalization,
        # rsl_rl 会 pop class_name，避免修改可重复使用的 spec。
        distribution_cfg=deepcopy(spec.distribution_cfg),
        **recurrent_kwargs,
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


def make_reference_inputs(onnx_model: torch.nn.Module) -> dict[str, tuple[torch.Tensor, ...]]:
    """生成固定观测和隐状态，覆盖零状态及非零状态。"""
    generator = torch.Generator().manual_seed(0)
    dummy_inputs = onnx_model.get_dummy_inputs()

    def random_inputs(scale: float) -> tuple[torch.Tensor, ...]:
        return tuple(
            (torch.rand(x.shape, generator=generator) * 2.0 - 1.0) * scale
            for x in dummy_inputs
        )

    return {
        "zero": dummy_inputs,
        "small": random_inputs(0.1),
        "unit": random_inputs(1.0),
    }


def tensor_outputs(result: Any) -> tuple[torch.Tensor, ...]:
    """统一 MLP 的 Tensor 和 RNN 的 tuple；GRU 尾部的 None 不属于 ONNX 输出。"""
    if isinstance(result, torch.Tensor):
        return (result,)
    return tuple(value for value in result if value is not None)


def get_reference_outputs(
    onnx_model: torch.nn.Module,
    inputs: dict[str, tuple[torch.Tensor, ...]],
) -> dict[str, ReferenceCase]:
    """先用 PyTorch 运行一次，作为 ONNX Runtime 的参考结果。"""
    references: dict[str, ReferenceCase] = {}

    def record(name: str, args: tuple[torch.Tensor, ...], *, carry_state: bool = False) -> tuple[torch.Tensor, ...]:
        outputs = tuple(y.detach().cpu() for y in tensor_outputs(onnx_model(*args)))
        if len(outputs) != len(onnx_model.output_names):
            raise RuntimeError("PyTorch 输出数量与 ONNX output_names 不一致")
        references[name] = ReferenceCase(args, outputs, carry_state)
        actions = outputs[0]
        print(
            f"[verify] torch {name:7s}: "
            f"shape={tuple(actions.shape)}, "
            f"min={actions.min().item():+.6f}, "
            f"max={actions.max().item():+.6f}"
        )
        return outputs

    with torch.inference_mode():
        for name, args in inputs.items():
            record(name, args)

        if onnx_model.is_recurrent:
            generator = torch.Generator().manual_seed(1)
            obs, *states = onnx_model.get_dummy_inputs()
            for step in range(8):
                obs = torch.rand(obs.shape, generator=generator) * 2.0 - 1.0
                outputs = record(f"step_{step}", (obs, *states), carry_state=step > 0)
                states = outputs[1:]

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
    references: dict[str, ReferenceCase],
    input_names: list[str],
    output_names: list[str],
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

    actual_inputs = [value.name for value in session.get_inputs()]
    actual_outputs = [value.name for value in session.get_outputs()]
    if actual_inputs != input_names or actual_outputs != output_names:
        raise RuntimeError(
            "ONNX 接口与 actor 不一致，"
            f"预期 inputs={input_names}, outputs={output_names}；"
            f"实际 inputs={actual_inputs}, outputs={actual_outputs}"
        )

    previous_outputs = None
    for name, case in references.items():
        values = [value.numpy() for value in case.inputs]
        if case.carry_state:
            if previous_outputs is None:
                raise RuntimeError("连续校验缺少上一时刻的 ONNX 隐状态")
            values[1:] = previous_outputs[1:]
        results = session.run(output_names, dict(zip(input_names, values, strict=True)))
        for output_name, result, expected in zip(output_names, results, case.outputs, strict=True):
            actual = torch.from_numpy(result)
            # 动作和所有隐状态都须满足容差。
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
            max_diff = (actual - expected).abs().max().item()
            print(
                f"[verify] onnx  {name:7s} {output_name:7s}: "
                f"shape={tuple(actual.shape)}, max|diff|={max_diff:.3e}"
            )
        previous_outputs = results

    print("[verify] PyTorch 与 ONNX 数值一致。")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将 rsl_rl RNN / MLP actor checkpoint 导出为 ONNX（含归一化及循环隐状态）。"
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
        f"model={spec.model_class}, "
        f"obs_dim={spec.obs_dim}, "
        f"num_actions={spec.num_actions}, "
        f"hidden_dims={spec.hidden_dims}, "
        f"activation={spec.activation}, "
        f"obs_normalization={spec.obs_normalization}, "
        f"distribution={(spec.distribution_cfg or {}).get('class_name')}"
    )
    if spec.model_class == "RNNModel":
        print(f"[export] {spec.rnn_type}: hidden_dim={spec.rnn_hidden_dim}, num_layers={spec.rnn_num_layers}")

    actor = build_actor(spec)
    actor.load_state_dict(state_dict, strict=True)
    actor.to("cpu").eval()
    print("[export] actor 权重加载完成（strict=True）。")

    # as_onnx() 负责把训练时的分布策略转换成确定性部署输出。
    onnx_model = actor.as_onnx(verbose=args.verbose).to("cpu").eval()

    test_inputs = make_reference_inputs(onnx_model)
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

    verify_onnx(output_path, references, onnx_model.input_names, onnx_model.output_names)
    if spec.model_class == "RNNModel":
        print("[deploy] 每步将 h_out 传回 h_in；首次推理及回合重置时清零隐状态。")
        if spec.rnn_type == "lstm":
            print("[deploy] LSTM 同时需要将 c_out 传回 c_in，并在重置时清零。")
    print("[done] ONNX 导出与验证全部完成。")


if __name__ == "__main__":
    main()
