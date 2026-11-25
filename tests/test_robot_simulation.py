import numpy as np
import pytest

from physical_intelligence.robot_simulation import (
    generate_robot_dataset,
    robot_physics_predict,
    robot_step,
)


def test_robot_drag_reduces_velocity_without_input():
    result = robot_step(
        state=[0.0, 0.0, 2.0, 0.0],
        action=[0.0, 0.0],
        dt=0.1,
        drag=0.25,
    )
    assert result[2] < 2.0


def test_robot_step_rejects_nonpositive_dt():
    with pytest.raises(ValueError):
        robot_step([0, 0, 0, 0], [0, 0], 0)


def test_robot_dataset_shapes_and_physics_prediction():
    x, y = generate_robot_dataset(samples=20, seed=5)
    assert x.shape == (20, 7)
    assert y.shape == (20, 4)
    assert robot_physics_predict(x).shape == (20, 4)
    assert np.isfinite(y).all()
