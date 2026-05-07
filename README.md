# CarRacing RL

Investigating the Impact of Reward Function Design on PPO Agents in CarRacing-v3.

Repository URL: https://github.com/ghuckerby/car-racing-rl

---

## Installation

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

# 2. Install dependencies
pip install -r requirements.txt
```

---

## Running Experiments

All experiments are run via `main.py` from the project root.

**List available experiments:**

```bash
python main.py --list
```

**Evaluate only (skip training, requires a trained model in `results/`):**

```bash
python main.py --experiment baseline --eval-only
```

**Train and evaluate a specific experiment:**

> **Note:** Each experiment trains for 2 million timesteps and takes approximately 6 hours on a single machine. Pre-trained models and results are included in the `results/` folder.

```bash
python main.py --experiment baseline
python main.py --experiment time_reward
python main.py --experiment speed_reward --seed 0
```

**Train and evaluate all experiments:**

```bash
python main.py
```

Available experiment names: `baseline`, `speed_reward`, `safety_reward`, `smoothness_reward`, `time_reward`, `composite_reward`, `time_search_reward`, `time_safety_search_reward`.

---

## Reward-Learning Search

To run the Optuna-based reward weight search:

```bash
python main.py --meta-learn
```

Search configuration and phase selection are defined in `experiments/reward_search.py`. Results are saved to an SQLite database in `results/meta_learning/` and can be resumed if interrupted.

---

## Project Structure

```
main.py                         # Entry point
agents/train_ppo.py             # PPO training pipeline
envs/reward_wrappers.py         # Reward wrapper implementations
envs/telemetry_collection.py    # Telemetry metrics collector
evaluation/evaluate.py          # Evaluation and video recording
experiments/configs.py          # Experiment definitions
experiments/reward_search.py    # Optuna reward search
results/                        # Trained models, telemetry, plots
```
