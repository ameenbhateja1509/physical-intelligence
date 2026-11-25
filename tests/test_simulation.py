
import pytest

from physical_intelligence.simulation import (
    VehicleState,
    VehicleAction,
    step,
)


def test_vehicle_moves_at_constant_velocity():
    state = VehicleState(
        x=0, y=0, vx=2, vy=3
    )

    action = VehicleAction(ax=0, ay=0)

    result = step(state, action, dt=1)

    assert result.x == pytest.approx(2)
    assert result.y == pytest.approx(3)
    assert result.vx == pytest.approx(2)
    assert result.vy == pytest.approx(3)


def test_acceleration_changes_velocity():
    state = VehicleState(
        x=0, y=0, vx=0, vy=0
    )

    action = VehicleAction(ax=2, ay=0)

    result = step(state, action, dt=1)

    assert result.vx == pytest.approx(2)
    assert result.x == pytest.approx(1)


def test_time_advances():
    state = VehicleState(
        x=0, y=0, vx=1, vy=0
    )

    result = step(
        state,
        VehicleAction(ax=0, ay=0),
        dt=0.5,
    )

    assert result.time_s == pytest.approx(0.5)


def test_invalid_time_step_is_rejected():
    state = VehicleState(
        x=0, y=0, vx=0, vy=0
    )

    with pytest.raises(ValueError):
        step(
            state,
            VehicleAction(ax=0, ay=0),
            dt=0,
        )
