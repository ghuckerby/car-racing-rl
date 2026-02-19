import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import CheckpointCallback
import os 

class DiscretizeActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super(DiscretizeActionWrapper, self).__init__(env)
        self.actions = {
            0: np.array([0.0, 0.0, 0.0], dtype=np.float32),   # No action
            1: np.array([0.0, 0.3, 0.0], dtype=np.float32),   # Accelerate
            2: np.array([0.0, 0.0, 0.8], dtype=np.float32),   # Brake
            3: np.array([-0.5, 0.0, 0.0], dtype=np.float32),  # Steer left
            4: np.array([0.5, 0.0, 0.0], dtype=np.float32),   # Steer right
        }
        self.action_space = gym.spaces.Discrete(len(self.actions))

    def action(self, action):
        if isinstance(action, np.ndarray):
            action = action.item()
        return self.actions[action]
    
def train_agent(algo_name, total_timesteps=100_000):
    def make_env():
        if algo_name == "DQN":
            env = gym.make("CarRacing-v3", render_mode="rgb_array", continuous=True)
            env = DiscretizeActionWrapper(env)
        else:
            env = gym.make("CarRacing-v3", render_mode="rgb_array", continuous=True)
        return env
    
    env = make_env()

    log_dir = f"./{algo_name}_logs/"
    os.makedirs(log_dir, exist_ok=True)

    if algo_name == "PPO":
        model = PPO(
            "CnnPolicy",
            env, 
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            gamma=0.99,
            batch_size=64,
            n_epochs=10,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            tensorboard_log=log_dir
        )
    else:
        model = DQN(
            "CnnPolicy",
            env,
            verbose=1,
            buffer_size=10_000,
            learning_rate=1e-4,
            learning_starts=1000,
            batch_size=64,
            gamma=0.99,
            train_freq=4,
            exploration_fraction=0.2,
            exploration_final_eps=0.02,
            tensorboard_log=log_dir,
        )

    checkpoint_callback = CheckpointCallback(save_freq=10_000, save_path=log_dir, name_prefix="car_racing_model")

    print(f"Training {algo_name} agent")
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback, progress_bar=True)
    model.save(f"car_racing_{algo_name.lower()}_baseline")
    print(f"{algo_name} training completed and model saved.")

if __name__ == "__main__":
    train_agent("PPO", total_timesteps=10_000)
    train_agent("DQN", total_timesteps=10_000)

    print("Evaluation of PPO agent")
    env = gym.make("CarRacing-v3", render_mode="human", continuous=True)
    model = PPO.load("car_racing_ppo_baseline")
    obs, info = env.reset()
    for _ in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
    env.close()

    print("Evaluation of DQN agent")
    env = DiscretizeActionWrapper(gym.make("CarRacing-v3", render_mode="human", continuous=True))
    model = DQN.load("car_racing_dqn_baseline")
    obs, info = env.reset()
    for _ in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
    env.close()