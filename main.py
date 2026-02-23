import gymnasium as gym
from gymnasium.wrappers import FrameStackObservation, GrayscaleObservation, ResizeObservation
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
import numpy as np

from train_dqn import train_dqn_agent
from train_ppo import train_ppo_agent

# Play: python -m gymnasium.envs.box2d.car_racing

def evaluate_agent(algo_name, model_path, n_episodes=5):
    print(f"\nEvaluating {algo_name} agent from {model_path}")

    if algo_name == "PPO":
        model = PPO.load(model_path)
    elif algo_name == "DQN":
        model = DQN.load(model_path)
    else:
        raise ValueError("Unsupported algorithm. Use 'PPO' or 'DQN'.")
    
    def make_eval_env():
        is_continuous = (algo_name == "PPO")
        env = gym.make("CarRacing-v3", render_mode="human", continuous=is_continuous, max_episode_steps=2000)
        env = MaxAndSkipEnv(env, skip=4)
        env = GrayscaleObservation(env)
        env = ResizeObservation(env, shape=(84, 84))
        env = FrameStackObservation(env, stack_size=4)
        return env
    
    env = make_eval_env()

    episode_rewards = []
    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action, states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        episode_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}")

    print(f"{algo_name} Evaluation completed.")
    print(f"Average Reward over {n_episodes} episodes: {np.mean(episode_rewards):.2f}")

if __name__ == "__main__":
    # train_dqn_agent(total_timesteps=1_000_000)
    # train_ppo_agent(total_timesteps=1_000_000, n_envs=1)
    
    # evaluate_agent("DQN", "car_racing_dqn.zip", n_episodes=10)
    evaluate_agent("PPO", "car_racing_ppo.zip", n_episodes=1)