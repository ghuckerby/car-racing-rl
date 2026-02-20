import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation, FrameStackObservation

def train_dqn_agent(total_timesteps=200_000):

    def make_env():
        env = gym.make("CarRacing-v3", render_mode="rgb_array", continuous=True)
        env = MaxAndSkipEnv(env, skip=4)  # Skip 4 frames to speed up training
        env = GrayscaleObservation(env)  # Convert to grayscale
        env = ResizeObservation(env, shape=(84, 84))  # Resize to 84x84
        env = FrameStackObservation(env, num_stack=4)  # Stack 4 frames
        return env

    env = make_env()
    model = DQN(
        "CnnPolicy",
        env,
        verbose=1,
        buffer_size=100_000,
        learning_rate=1e-4,
        gamma=0.99,
        tau=0.005,
        batch_size=64,
        train_freq=4,
        target_update_interval=1_000,
        exploration_fraction=0.1,
        exploration_final_eps=0.1,
        device="auto",
    )

    print(f"Training dqn agent for {total_timesteps} timesteps.")
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    model.save("car_racing_dqn_baseline.zip")
    print("DQN training completed and model saved.")