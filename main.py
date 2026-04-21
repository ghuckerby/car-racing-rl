from experiments.configs import EXPERIMENTS
from agents.train_ppo import train_ppo_agent
from evaluation.evaluate import evaluate_agent

def main():

    results = {}

    for experiment in EXPERIMENTS:
        for seed in experiment["seeds"]:
            run_name = f"{experiment['name']}_seed{seed}"
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
                reward_wrapper=experiment["reward_wrapper"]
            )

            results[run_name] = {"mean_reward": mean, "std_reward": std, "telemetry": telemetry}

    # Print summary of results
    print("\nEXPERIMENT RESULTS SUMMARY:\n")
    print(f"{'Experiment':<30} "
          f"{'Mean Reward':>12} "
          f"{'Std':>8} "
          f"{'Mean Speed':>12} "
          f"{'Max Speed':>12} "
          f"{'Mean Lat Vel':>15} "
          f"{'Off Track %':>12} "
          f"{'Mean Steering Chg':>20} "
          f"{'Mean Throttle':>15} "
          f"{'Mean Brake':>12}")

    for run_name, metrics in results.items():
        telemetry = metrics['telemetry']
        print(f"{run_name:<30} "
              f"{metrics['mean_reward']:>12.2f} "
              f"{metrics['std_reward']:>8.2f} "
              f"{telemetry['mean_speed']:>12.2f} "
              f"{telemetry['max_speed']:>12.2f} "
              f"{telemetry['mean_lateral_velocity']:>15.2f} "
              f"{telemetry['off_track_percentage']:>12.2f} "
              f"{telemetry['mean_steering_change']:>20.2f} "
              f"{telemetry['mean_throttle']:>15.2f} "
              f"{telemetry['mean_brake']:>12.2f}")

if __name__ == "__main__":
    main()