"""Install the nominal gas spring force law in a MuJoCo model."""

import mujoco

from assets.robot.wheelbipeV14_2.gas_spring import (
    GAS_SPRING_DAMPING,
    GAS_SPRING_MAX_COMPRESSION,
    GAS_SPRING_PRELOAD_FORCE,
    GAS_SPRING_QUADRATIC_STIFFNESS,
    GAS_SPRING_STIFFNESS,
    GAS_SPRING_STROKE_AT_ZERO_POSITION,
    gas_spring_force_at_compression,
)

_SPRING_ACTUATOR_GROUP = 30
_SPRING_ACTUATOR_MASK = 1 << _SPRING_ACTUATOR_GROUP


def configure_gas_spring(spec: mujoco.MjSpec, joint_name: str) -> None:
    """Add the gas spring actuator to a spec for later activation."""
    actuator = spec.actuator(f"{joint_name}_ctrl")
    actuator.gainprm[0] = 0.0
    actuator.biastype = mujoco.mjtBias.mjBIAS_USER
    actuator.biasprm[:6] = (
        GAS_SPRING_STROKE_AT_ZERO_POSITION,
        GAS_SPRING_MAX_COMPRESSION,
        GAS_SPRING_PRELOAD_FORCE,
        GAS_SPRING_STIFFNESS,
        GAS_SPRING_QUADRATIC_STIFFNESS,
        GAS_SPRING_DAMPING,
    )
    # MuJoCo calls actuator biases during compilation before Python model/data exist.
    actuator.group = _SPRING_ACTUATOR_GROUP
    spec.option.disableactuator |= _SPRING_ACTUATOR_MASK


def _gas_spring_bias(model, data, actuator_id):
    if model.actuator_group[actuator_id] != _SPRING_ACTUATOR_GROUP:
        return 0.0

    offset, cap, preload, stiffness, quadratic, damping = model.actuator_biasprm[actuator_id, :6]
    # The gas spring actuator is a unit-gear slide motor.
    stroke = offset - data.actuator_length[actuator_id]
    if stroke < 0.0:
        return 0.0
    return gas_spring_force_at_compression(
        min(stroke, cap), data.actuator_velocity[actuator_id],
        preload_force=preload, stiffness=stiffness, quadratic_stiffness=quadratic, damping=damping,
    )


def enable_gas_springs(model: mujoco.MjModel) -> mujoco.MjModel:
    """Enable the configured gas springs on a compiled model."""
    if mujoco.get_mjcb_act_bias() is not _gas_spring_bias:
        mujoco.set_mjcb_act_bias(_gas_spring_bias)
    model.opt.disableactuator &= ~_SPRING_ACTUATOR_MASK
    return model


def build_model_with_gas_springs(params, spec: mujoco.MjSpec) -> mujoco.MjModel:
    from mujoco import sysid

    return enable_gas_springs(sysid.apply_param_modifiers(params, spec))
