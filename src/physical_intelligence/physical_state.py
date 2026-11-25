
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PhysicalState:
    """Common representation of a simulated physical system."""

    machine_type: str

    position: list[float]
    velocity: list[float]

    timestamp_s: float

    units: dict[str, str] = field(default_factory=lambda: {
        "position": "metres",
        "velocity": "metres_per_second",
        "time": "seconds",
    })

    coordinate_frame: str = "world"

    uncertainty: dict[str, float] = field(
        default_factory=dict
    )

    observed: bool = True

    metadata: dict = field(default_factory=dict)

    def validate(self):
        if not self.machine_type:
            raise ValueError("Machine type is required")

        if len(self.position) != len(self.velocity):
            raise ValueError(
                "Position and velocity dimensions must match"
            )

        if self.timestamp_s < 0:
            raise ValueError("Timestamp cannot be negative")

        if not all(
            isinstance(value, (int, float))
            for value in self.position + self.velocity
        ):
            raise ValueError("State values must be numeric")

        return True
