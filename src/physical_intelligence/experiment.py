
import json
from pathlib import Path

from physical_intelligence.simulation import generate_dataset
from physical_intelligence.baselines import (
    physics_predict,
    train_existing_model,
    calculate_metrics,
)


def main():
    print("\nPROJECT PHYSICAL INTELLIGENCE")
    print("PI-001: Reproducibility experiment\n")

    # Training data: fixed seed for reproducibility.
    train_x, train_y = generate_dataset(
        samples=10000,
        seed=42,
    )

    print("Training Random Forest...")
    model = train_existing_model(train_x, train_y)

    test_seeds = [101, 202, 303]
    all_results = []

    for seed in test_seeds:
        test_x, test_y = generate_dataset(
            samples=2000,
            seed=seed,
        )

        physics_output = physics_predict(test_x)
        ml_output = model.predict(test_x)

        physics_metrics = calculate_metrics(
            test_y,
            physics_output,
        )

        ml_metrics = calculate_metrics(
            test_y,
            ml_output,
        )

        result = {
            "test_seed": seed,
            "physics": physics_metrics,
            "random_forest": ml_metrics,
        }

        all_results.append(result)

        print(f"\nTest seed: {seed}")
        print("Physics:", physics_metrics)
        print("Random Forest:", ml_metrics)

    # Save the complete experiment record.
    output = {
        "experiment": "PI-001",
        "environment": "2D simplified vehicle",
        "training_samples": len(train_x),
        "test_samples_per_seed": 2000,
        "training_seed": 42,
        "results": all_results,
    }

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    output_path = results_dir / "pi001_vehicle_baseline.json"

    with output_path.open("w") as file:
        json.dump(output, file, indent=4)

    print("\nExperiment completed.")
    print(f"Research record saved to: {output_path}")


if __name__ == "__main__":
    main()
