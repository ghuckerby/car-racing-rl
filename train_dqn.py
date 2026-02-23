import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation, FrameStackObservation

def train_dqn_agent(total_timesteps=200_000):

    def make_env():
        env = gym.make("CarRacing-v3", render_mode="rgb_array", continuous=False)
        env = MaxAndSkipEnv(env, skip=4)  # Skip 4 frames to speed up training
        env = GrayscaleObservation(env)  # Convert to grayscale
        env = ResizeObservation(env, shape=(84, 84))  # Resize to 84x84
        env = FrameStackObservation(env, stack_size=4)  # Stack 4 frames
        return env

    env = make_env()
    model = DQN(
        "CnnPolicy",
        env,
        verbose=1,
        buffer_size=100_000,
    )

    print(f"Training dqn agent for {total_timesteps} timesteps.")
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    model.save("car_racing_dqn.zip")
    print("DQN training completed and model saved.")