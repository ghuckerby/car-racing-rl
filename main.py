import gymnasium as gym
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage, DummyVecEnv
import numpy as np

from train_dqn import train_dqn_agent
from train_ppo import train_ppo_agent

# Play: python -m gymnasium.envs.box2d.car_racing

def evaluate_agent(algo_name, model_path, n_episodes=5):
    print(f"\nEvaluating {algo_name} agent from {model_path}")

    

if __name__ == "__main__":
    train_dqn_agent(total_timesteps=200_000)
    train_ppo_agent(total_timesteps=200_000, n_envs=4)
    
    evaluate_agent("PPO", "car_racing_ppo_baseline.zip")
    evaluate_agent("DQN", "car_racing_dqn_baseline.zip")