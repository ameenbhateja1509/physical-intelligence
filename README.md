# physical-intelligence
Research project investigating transferable physical intelligence across simulated machines.

# Project PHYSICAL INTELLIGENCE

## PI-001 — Cross-Machine Physical Intelligence

### Research objective
Investigate whether a shared intelligence architecture can
understand and predict physical systems across different
simulated machine categories.

### Initial environment
A simplified 2D vehicle simulation.

### Models
- B0: Analytical physics baseline
- B1: Existing machine-learning model (Random Forest)
- B2: Learned neural transition model (MLPRegressor)

### Input
Current position, velocity, acceleration and time interval.

### Prediction target
Next position and velocity.

### Evaluation
- Position mean squared error
- Velocity mean squared error
- Overall mean squared error

### Research integrity
Training and test datasets use separate random seeds.
Results must be measured, not assumed.
A successful vehicle experiment does not establish
transfer to other machine categories.

### Future environments
1. Mobile robot
2. Industrial machine

### Current status
- Vehicle simulation and analytical baseline implemented.
- Random Forest baseline evaluated on three held-out random seeds.
- Shared physical-state schema and machine examples implemented.
- Neural world-model experiment executed in Codespaces and results committed to `results/pi001_world_model.json`.
- Automated test suite reported 11 passing tests in Codespaces.

### PI-001 neural world-model results

Training used 20,000 transitions (seed 42). Each of three held-out test datasets contained 2,000 transitions.

| Test seed | Position MSE | Velocity MSE | Overall MSE |
|---|---:|---:|---:|
| 101 | 0.0020963 | 0.0006995 | 0.0013979 |
| 202 | 0.0020932 | 0.0007332 | 0.0014132 |
| 303 | 0.0021353 | 0.0007087 | 0.0014220 |

The analytical physics baseline recorded zero error because its equations match the simulator. The learned model approximates those equations and has nonzero error. These results establish one-step prediction on this synthetic vehicle benchmark only; they do not establish physical understanding or transfer.

### Important limitations
The current vehicle is a simplified 2D point-mass system. The learned model predicts one-step transitions from simulator-generated data; it does not yet demonstrate general physical understanding or cross-machine transfer. Position MSE and velocity MSE should be interpreted separately because they have different physical units.
