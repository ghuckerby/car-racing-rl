import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation, FrameStackObservation
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
    
def train_ppo_agent(total_timesteps=500_000, n_envs=4):
    def make_env():
        env = gym.make("CarRacing-v3", render_mode="rgb_array")
        env = MaxAndSkipEnv(env, skip=4)  # Skip 4 frames to speed up training
        env = GrayscaleObservation(env)  # Convert to grayscale
        env = ResizeObservation(env, shape=(84, 84))  # Resize to 84x84
        env = FrameStackObservation(env, num_stack=4)  # Stack 4 frames
        return env
    
    env = make_vec_env(make_env, n_envs=n_envs)
    env = VecFrameStack(env, n_stack=4)
    env = VecTransposeImage(env)
    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        learning_rate=1e-4,
        gamma=0.99,
        batch_size=128,
        n_epochs=10,
        n_steps=2048,
    )

    print(f"Training PPO agent for {total_timesteps} timesteps.")
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    model.save("car_racing_ppo_baseline.zip")
    print("PPO training completed and model saved.")