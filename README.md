# RL Lane Change in SUMO

Reinforcement Learning project for autonomous lane-change decision making in a SUMO traffic scenario.

This repository implements and compares:
- Tabular Q-Learning
- Deep Q-Network (DQN) with Stable-Baselines3
- Default SUMO behavior (baseline)

The agent controls high-level lane-change actions in a two-lane road with obstacle vehicles, while SUMO handles vehicle dynamics and simulation.

## Table of Contents
- [Project Overview](#project-overview)
- [Environment and Task](#environment-and-task)
- [Methods](#methods)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Evaluation](#evaluation)
- [Results](#results)
- [Future Work](#future-work)
- [Author](#author)

## Project Overview
The objective of this project is to train an ego vehicle to perform safe and efficient lane-change decisions in a simplified traffic environment.

At each step, the agent observes:
- Current lane index of the ego vehicle
- Ego speed
- Distance to the nearest obstacle ahead in lane 0
- Distance to the nearest obstacle ahead in lane 1

It then selects one of three actions:
- Keep lane
- Change lane to the left
- Change lane to the right

The project studies how classical RL (Q-Learning) and deep RL (DQN) perform on this task, and compares them with SUMO default lane-change behavior.

## Environment and Task
The custom Gymnasium environment is defined in `env.py`.

Key properties:
- Simulator: SUMO (TraCI interface)
- Scenario config: `data/obstacles.sumocfg`
- Road: single edge with two lanes
- Obstacles: predefined vehicles in route file (`data/obstacles.rou.xml`), frozen during simulation to create a deterministic obstacle field
- Ego vehicle ID: `vehAgent`

### Observation Space
Continuous vector of size 4:
1. Lane index
2. Speed
3. Distance to closest front obstacle in lane 0
4. Distance to closest front obstacle in lane 1

### Action Space
Discrete(3):
- 0 = Keep lane
- 1 = Change left
- 2 = Change right

### Reward Design
The reward includes:
- Positive term proportional to speed (encourages progress)
- Penalty for lane changes (encourages stability)
- Large bonus on successful arrival
- Large penalty on collision

## Methods
### 1) Q-Learning (`q_learning.py`)
- Tabular RL with discretized state dimensions (speed and distances binned)
- Epsilon-greedy exploration
- Q-table saved as `q_table.npy`
- TensorBoard logs under `log/q_learning`

### 2) Deep Q-Network (`deep_rl.py`)
- Stable-Baselines3 DQN with MLP policy
- Replay buffer, target network updates, and epsilon exploration schedule
- Model saved as `dqn_lane_change.zip`
- TensorBoard logs under `log/deep_rl`

### 3) Baseline (`evaluate.py`)
- Default SUMO lane-change control mode
- Used for comparison against learned policies

## Repository Structure
- `env.py`: custom Gymnasium + SUMO environment
- `q_learning.py`: tabular Q-Learning training script
- `deep_rl.py`: DQN training script
- `evaluate.py`: evaluation pipeline for Q-Learning, DQN, and default baseline
- `run.py`: scenario runner/visual check script
- `data/`: SUMO network, route, config, and GUI settings files
- `log/`: TensorBoard logs
- `q_table.npy`: saved Q-table
- `dqn_lane_change.zip`: saved DQN model
- `evaluation_results.csv`: exported evaluation metrics

## Requirements
- Python 3.9+
- SUMO (with TraCI)
- pip packages:
  - numpy
  - pandas
  - gymnasium
  - stable-baselines3
  - torch
  - tensorboard

## Installation
1. Clone the repository:
```bash
git clone https://github.com/MOHAMMEDFLITI/rl_projet_submission.git
cd rl_projet_submission
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate
```

3. Install Python dependencies:
```bash
pip install numpy pandas gymnasium stable-baselines3 torch tensorboard
```

4. Install SUMO and set environment variable:
- Install SUMO from the official website.
- Set `SUMO_HOME` so TraCI tools are accessible.

Example (Windows PowerShell):
```powershell
$env:SUMO_HOME="C:\\Program Files (x86)\\Eclipse\\Sumo"
```

## How to Run
### Quick simulation check (GUI)
```bash
python run.py
```

### Train Q-Learning
```bash
python q_learning.py
```

### Train DQN
```bash
python deep_rl.py
```

### Evaluate all methods
```bash
python evaluate.py
```

Evaluation output is saved to:
- `evaluation_results.csv`

## Evaluation
The evaluation script runs multiple episodes for each method and records:
- Model type
- Episode index
- Total reward
- Number of steps

You can compare average metrics using the grouped summary printed by `evaluate.py`.

## Results
Replace this section with your final metrics and plots.

Example template:
- Q-Learning average reward: **[insert value]**
- DQN average reward: **[insert value]**
- Default SUMO average reward: **[insert value]**
- Best method: **[insert method]**

Optional additions:
- Collision rate per method
- Success/arrival rate
- TensorBoard screenshots

## Future Work
- Add richer observations (rear vehicles, relative velocities, TTC)
- Use more realistic traffic dynamics (moving obstacles)
- Extend to multi-agent settings
- Tune reward shaping and hyperparameters systematically
- Add training/evaluation seeds for stronger reproducibility
