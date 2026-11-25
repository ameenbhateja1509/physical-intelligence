
from physical_intelligence.physical_state import PhysicalState


def vehicle_example():
    return PhysicalState(
        machine_type="vehicle",
        position=[10.0, 5.0],
        velocity=[2.0, 0.0],
        timestamp_s=1.0,
        metadata={"description": "2D simulated vehicle"},
    )


def mobile_robot_example():
    return PhysicalState(
        machine_type="mobile_robot",
        position=[3.0, 4.0],
        velocity=[0.5, 0.2],
        timestamp_s=1.0,
        metadata={"description": "2D mobile robot"},
    )


def industrial_machine_example():
    return PhysicalState(
        machine_type="industrial_machine",
        position=[2.0],
        velocity=[0.1],
        timestamp_s=1.0,
        metadata={"description": "Single-axis machine"},
    )


def validate_examples():
    examples = [
        vehicle_example(),
        mobile_robot_example(),
        industrial_machine_example(),
    ]

    return [
        {
            "machine": state.machine_type,
            "valid": state.validate(),
            "dimensions": len(state.position),
        }
        for state in examples
    ]
