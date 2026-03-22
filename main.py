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
            mean, std = evaluate_agent(
                experiment_name=run_name,
                reward_wrapper=experiment["reward_wrapper"]
            )

            results[run_name] = {"mean_reward": mean, "std_reward": std}

    # Print summary of results
    print("\nEXPERIMENT RESULTS SUMMARY:")
    for run_name, metrics in results.items():
        print(f"{run_name}: Mean Reward = {metrics['mean_reward']:.2f}, Std = {metrics['std_reward']:.2f}")

if __name__ == "__main__":
    main()