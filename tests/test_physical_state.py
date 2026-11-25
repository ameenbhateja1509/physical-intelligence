
import pytest

from physical_intelligence.physical_state import PhysicalState
from physical_intelligence.machine_examples import (
    validate_examples,
)


def test_three_machine_examples_are_valid():
    results = validate_examples()

    assert len(results) == 3
    assert all(item["valid"] for item in results)


def test_machine_dimensions_can_differ():
    results = validate_examples()

    dimensions = [
        item["dimensions"] for item in results
    ]

    assert dimensions == [2, 2, 1]


def test_invalid_dimension_is_rejected():
    state = PhysicalState(
        machine_type="vehicle",
        position=[1.0, 2.0],
        velocity=[1.0],
        timestamp_s=1.0,
    )

    with pytest.raises(ValueError):
        state.validate()


def test_negative_timestamp_is_rejected():
    state = PhysicalState(
        machine_type="robot",
        position=[1.0],
        velocity=[1.0],
        timestamp_s=-1.0,
    )

    with pytest.raises(ValueError):
        state.validate()
