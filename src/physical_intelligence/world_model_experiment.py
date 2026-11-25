import json
from pathlib import Path

from physical_intelligence.baselines import calculate_metrics, physics_predict
from physical_intelligence.simulation import generate_dataset
from physical_intelligence.world_model import WorldModel


def main():
    print("\nPI-001: LEARNED WORLD-MODEL EXPERIMENT")

    train_x, train_y = generate_dataset(samples=20000, seed=42)
    test_seeds = [101, 202, 303]

    print("Training neural world model...")
    world_model = WorldModel(random_state=42)
    world_model.train(train_x, train_y)

    results = []
    for seed in test_seeds:
        test_x, test_y = generate_dataset(samples=2000, seed=seed)

        physics_metrics = calculate_metrics(test_y, physics_predict(test_x))
        learned_metrics = calculate_metrics(test_y, world_model.predict(test_x))

        results.append({
            "test_seed": seed,
            "physics_baseline": physics_metrics,
            "learned_world_model": learned_metrics,
        })

        print(f"\nTest seed: {seed}")
        print("Physics baseline:", physics_metrics)
        print("Learned world model:", learned_metrics)

    output = {
        "experiment": "PI-001-WORLD-MODEL",
        "environment": "2D simplified point-mass vehicle",
        "training_seed": 42,
        "training_samples": len(train_x),
        "test_samples_per_seed": 2000,
        "test_seeds": test_seeds,
        "metrics_note": (
            "Position MSE and velocity MSE have different units. "
            "Overall MSE is a convenience statistic and should not "
            "be interpreted as a dimensionally homogeneous physical error."
        ),
        "results": results,
    }

    output_path = Path("results/pi001_world_model.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=4) + "\n")

    print(f"\nSaved {output_path}")


if __name__ == "__main__":
    main()
