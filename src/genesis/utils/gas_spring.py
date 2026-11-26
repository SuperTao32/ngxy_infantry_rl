"""训练与悬挂调试共用的气弹簧力曲线。"""

from assets.robot.wheelbipeV14_2.gas_spring import GAS_SPRING_STROKE_AT_ZERO_POSITION, gas_spring_force_at_compression


def gas_spring_force(position, velocity, *, preload_force, stiffness, quadratic_stiffness, damping, max_compression):
    stroke = GAS_SPRING_STROKE_AT_ZERO_POSITION - position
    compression = stroke.clamp(0.0, max_compression)
    force = gas_spring_force_at_compression(
        compression, velocity, preload_force=preload_force, stiffness=stiffness, quadratic_stiffness=quadratic_stiffness, damping=damping
    )
    return force.masked_fill(stroke < 0.0, 0.0)
