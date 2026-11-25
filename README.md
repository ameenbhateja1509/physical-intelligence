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
- Neural world-model implementation and tests added; execution and metrics still need verification in Codespaces.

### Important limitations
The current vehicle is a simplified 2D point-mass system. The learned model predicts one-step transitions from simulator-generated data; it does not yet demonstrate general physical understanding or cross-machine transfer. Position MSE and velocity MSE should be interpreted separately because they have different physical units.
