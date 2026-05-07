import os
import json
import functools
import optuna

from agents.train_ppo import train_ppo_agent
from evaluation.evaluate import evaluate_agent
from envs.reward_wrappers import CompositeRewardWrapper

# Reward search using Optuna to find the best combination of reward weights for speed, safety, smoothness, and time
def objective(trial):

    # (First two runs commented out)

    # -- Full Composite Reward Search --

    # Weight ranges for all reward components
    # speed_weight = trial.suggest_float("speed_weight", 0.0, 0.2)
    # off_track_penalty = trial.suggest_float("off_track_penalty", -2.0, 0.0)
    # smooth_weight = trial.suggest_float("smooth_weight", 0.0, 2.0)
    # time_penalty = trial.suggest_float("time_penalty", 0.0, 0.5)
  
    # wrapper = functools.partial(
    #     CompositeRewardWrapper,
    #     speed_weight=speed_weight,
    #     off_track_penalty=off_track_penalty,
    #     smooth_weight=smooth_weight,
    #     time_penalty=time_penalty
    # )

    # experiment_name = f"meta_learning/meta_trial_{trial.number}_seed0"

    # -- Time Penalty Search --

    # Weights for just time penalty
    # time_penalty = trial.suggest_float("time_penalty", 0.05, 0.5)

    # wrapper = functools.partial(
    #     CompositeRewardWrapper,
    #     time_penalty=time_penalty
    # )

    # experiment_name = f"meta_learning/time_search/trial_{trial.number}_seed0"

    # -- Time and Off-Track Penalty Search --

    # Weights for time penalty and safety
    time_penalty = trial.suggest_float("time_penalty", 0.0, 0.5)
    off_track_penalty = trial.suggest_float("off_track_penalty", -2.0, 0.0)

    wrapper = functools.partial(
        CompositeRewardWrapper,
        time_penalty=time_penalty,
        off_track_penalty=off_track_penalty
    )
    
    experiment_name = f"meta_learning/time__safety_search/trial_{trial.number}_seed0"

    train_ppo_agent(
        experiment_name=experiment_name,
        total_timesteps=1_000_000,
        reward_wrapper=wrapper,
        seed=0
    )

    mean_reward, _, _ = evaluate_agent(
        experiment_name=experiment_name,
        reward_wrapper=wrapper,
        record_video=False
    )

    return mean_reward

# Run the reward search over multiple trials and save the best configuration
def run_reward_search(n_trials=10):
    os.makedirs("results/meta_learning/time__safety_search", exist_ok=True)
    storage = "sqlite:///results/meta_learning/time__safety_search/reward_search.db"

    # Create Optuna study for reward search
    study = optuna.create_study(
        study_name="time_safety_search",
        direction="maximize",
        storage=storage,
        load_if_exists=True
    )
    study.optimize(objective, n_trials=n_trials)
    best = study.best_params

    print(f"Best trial: {study.best_trial.number}")
    print(f"Best mean reward: {study.best_value:.2f}")
    print(f"Best weights: {best}")

    output_path = "results/meta_learning/time__safety_search/best_weights.json"
    with open(output_path, "w") as f:
        json.dump({"best_parameters": best, "best_value": study.best_value}, f, indent=2)