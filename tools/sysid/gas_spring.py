"""Install the nominal gas spring force law in a MuJoCo model."""

import mujoco

from assets.robot.wheelbipeV14_2.gas_spring import (
    GAS_SPRING_DAMPING,
    GAS_SPRING_MAX_COMPRESSION,
    GAS_SPRING_PRELOAD_FORCE,
    GAS_SPRING_QUADRATIC_STIFFNESS,
    GAS_SPRING_STIFFNESS,
)


def configure_gas_spring(spec: mujoco.MjSpec, joint_name: str) -> None:
    """Apply a clipped quadratic force along the positive slide axis."""
    actuator = spec.actuator(f"{joint_name}_ctrl")
    actuator.gainprm[0] = 0.0
    actuator.biastype = mujoco.mjtBias.mjBIAS_AFFINE
    actuator.biasprm[:3] = (GAS_SPRING_PRELOAD_FORCE, 0.0, -GAS_SPRING_DAMPING)

    # The fixed tendons have length q and each acts only above its rest interval.
    # The first adds the quadratic from q=0. The second cancels its growth above
    # the compression cap by subtracting the polynomial translated to that cap.
    cap = GAS_SPRING_MAX_COMPRESSION
    k1 = GAS_SPRING_STIFFNESS
    k2 = GAS_SPRING_QUADRATIC_STIFFNESS
    for suffix, threshold, linear, quadratic in (
        ("rise", 0.0, -k1, -k2),
        ("cap", cap, k1 + 2.0 * k2 * cap, k2),
    ):
        tendon = spec.add_tendon(
            name=f"{joint_name}_{suffix}",
            springlength=(-float("inf"), threshold),
            stiffness=(linear, quadratic, 0.0),
        )
        tendon.wrap_joint(joint_name, 1.0)
