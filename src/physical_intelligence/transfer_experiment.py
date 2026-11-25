"""PI-001 transfer test: ideal vehicle dynamics -> damped mobile robot.

Compare:
1. Vehicle-trained neural model evaluated zero-shot on robot data.
2. Neural model trained on robot examples.
3. Robot-specific analytical physics baseline.

This tests transfer across two related synthetic dynamics, not real machines.
"""
import json
from pathlib import Path

from physical_intelligence.baselines import calculate_metrics
from physical_intelligence.robot_simulation import (
    generate_robot_dataset,
    robot_physics_predict,
)
from physical_intelligence.simulation import generate_dataset
from physical_intelligence.world_model import WorldModel


def main():
    print("\nPI-001: VEHICLE-TO-ROBOT TRANSFER EXPERIMENT")

    vehicle_x, vehicle_y = generate_dataset(samples=20000, seed=42)
    robot_train_x, robot_train_y = generate_robot_dataset(
        samples=5000, seed=77
    )
    robot_test_x, robot_test_y = generate_robot_dataset(
        samples=2000, seed=909
    )

    print("Training source model on ideal vehicle dynamics...")
    vehicle_model = WorldModel(random_state=42)
    vehicle_model.train(vehicle_x, vehicle_y)

    print("Training target model on robot examples...")
    robot_model = WorldModel(random_state=77)
    robot_model.train(robot_train_x, robot_train_y)

    zero_shot = vehicle_model.predict(robot_test_x)
    target_trained = robot_model.predict(robot_test_x)
    physics = robot_physics_predict(robot_test_x)

    results = {
        "experiment": "PI-001-TRANSFER-001",
        "source_domain": "ideal 2D point-mass vehicle",
        "target_domain": "2D point-mass robot with linear velocity drag",
        "source_training_samples": len(vehicle_x),
        "target_training_samples": len(robot_train_x),
        "test_samples": len(robot_test_x),
        "robot_drag": 0.25,
        "metrics": {
            "vehicle_model_zero_shot": calculate_metrics(robot_test_y, zero_shot),
            "robot_model_target_trained": calculate_metrics(robot_test_y, target_trained),
            "robot_analytical_physics": calculate_metrics(robot_test_y, physics),
        },
        "interpretation_limit": (
            "A synthetic transfer benchmark between related point-mass models; "
            "not evidence of transfer to real robots or general physical intelligence."
        ),
    }

    output_path = Path("results/pi001_transfer_vehicle_to_robot.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=4) + "\n")

    for name, metrics in results["metrics"].items():
        print(f"\n{name}: {metrics}")
    print(f"\nSaved {output_path}")


if __name__ == "__main__":
    main()
