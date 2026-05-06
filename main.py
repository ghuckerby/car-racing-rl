import argparse
import os
from experiments.configs import EXPERIMENTS
from agents.train_ppo import train_ppo_agent
from evaluation.evaluate import evaluate_agent
from experiments.reward_search import run_reward_search

# Run Manually: python -m gymnasium.envs.box2d.car_racing --continuous False

def main():

    parser = argparse.ArgumentParser(description="Train and evaluate CarRacing-v3 reward wrappers")

    parser.add_argument("--experiment", type=str, nargs="+", default=None, help="Name of experiment to run. Runs all if not specified.")
    parser.add_argument("--seed", type=int, default=None, help="Seed to run for the experiment.")
    parser.add_argument("--list", action="store_true", help="List all available experiments.")
    parser.add_argument("--eval-only", action="store_true", help="Skip training and only evaluate already-trained models.")
    parser.add_argument("--meta-learn", action="store_true", help="Run Optuna reward weight search.")

    args = parser.parse_args()

    # Just list available experiments and exit
    if args.list:
        for exp in EXPERIMENTS:
            print(f"{exp['name']} (seeds :{exp['seeds']})")
        return
    
    # Run meta-learning reward search, returns to skip normal experiments
    if args.meta_learn:
        run_reward_search()
        return

    results = {}

    # Iterate through each experiment configuration (Reward Wrapper)
    for experiment in EXPERIMENTS:
        if args.experiment and experiment["name"] not in args.experiment:
            continue

        seeds = [args.seed] if args.seed is not None else experiment["seeds"]

        # Over multiple seeds
        for seed in seeds:
            run_name = f"{experiment['name']}_seed{seed}"

            # Skip training and just evaluate
            if args.eval_only:
                model_path_best = os.path.join("results", run_name, "best_model.zip")
                model_path_final = os.path.join("results", run_name, "final_model.zip")

                if not os.path.exists(model_path_best) and not os.path.exists(model_path_final):
                    print(f"Skipping {run_name} — no trained model found.")
                    continue

                print(f"\nEVALUATING: {run_name}\n")

            # Normal training and evaluation
            else:
                print(f"\nSTARTING EXPERIMENT: {run_name}\n")
                # Train the agent
                train_ppo_agent(
                    experiment_name=run_name,
                    total_timesteps=experiment["total_timesteps"],
                    reward_wrapper=experiment["reward_wrapper"],
                    seed=seed
                )

            # Evaluate the agent
            mean, std, telemetry = evaluate_agent(
                experiment_name=run_name,
                n_eval_episodes=20,
                reward_wrapper=experiment["reward_wrapper"]
            )

            results[run_name] = {"mean_reward": mean, "std_reward": std, "telemetry": telemetry}

    # Print summary of results
    print("\nEXPERIMENT RESULTS SUMMARY:\n")
    print(f"{'Experiment':<34} "
          f"{'Mean Reward':>12} "
          f"{'Std':>8} "
          f"{'Mean Speed':>12} "
          f"{'Max Speed':>12} "
          f"{'Mean Lat Vel':>15} "
          f"{'Off Track %':>12} "
          f"{'Mean Steering Chg':>20} "
          f"{'Mean Throttle':>15} "
          f"{'Mean Brake':>12} "
          f"{'Completion %':>14} "
          f"{'Completion Rate':>16} "
          f"{'Mean Completion Steps':>22}")

    for run_name, metrics in results.items():
        tel = metrics['telemetry']
        print(f"{run_name:<30} "
              f"{metrics['mean_reward']:>12.2f} "
              f"{metrics['std_reward']:>8.2f} "
              f"{tel.get('mean_speed', float('nan')):>12.2f} "
              f"{tel.get('max_speed', float('nan')):>12.2f} "
              f"{tel.get('mean_lateral_velocity', float('nan')):>15.2f} "
              f"{tel.get('off_track_percentage', float('nan')):>12.2f} "
              f"{tel.get('mean_steering_change', float('nan')):>20.2f} "
              f"{tel.get('mean_throttle', float('nan')):>15.2f} "
              f"{tel.get('mean_brake', float('nan')):>12.2f} "
              f"{tel.get('track_completion_percentage', float('nan')):>14.2%} "
              f"{tel.get('completion_rate', float('nan')):>16.2%} "
              f"{str(round(tel['mean_completion_steps'])) if tel.get('mean_completion_steps') is not None else 'N/A':>22}")

if __name__ == "__main__":
    main()