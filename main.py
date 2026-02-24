import gymnasium as gym
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack, VecVideoRecorder, VecTransposeImage
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.atari_wrappers import WarpFrame
import os

from train_dqn import train_dqn_agent
from train_ppo import train_ppo_agent

# Play: python -m gymnasium.envs.box2d.car_racing

def evaluate_agent(algo_name, log_dir, n_eval_episodes=20, record_video=True):
    print(f"\nEvaluating {algo_name} agent from {log_dir}")

    # Create evaluation environment
    env_kwargs_dict = {"continuous": False} if algo_name == "DQN" else {}
    env = make_vec_env(
        "CarRacing-v3",
        n_envs=1,
        env_kwargs=env_kwargs_dict,
        wrapper_class=WarpFrame
    )
    env = VecFrameStack(env, n_stack=4)
    env = VecTransposeImage(env)

    # Load model
    best_model_path = os.path.join(log_dir, "best_model")
    if algo_name == "PPO":
        model = PPO.load(best_model_path, env=env)
    elif algo_name == "DQN":
        model = DQN.load(best_model_path, env=env)
    else:
        raise ValueError("Unsupported algorithm. Use 'PPO' or 'DQN'.")

    # Evaluate policy
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes)
    print(f"{algo_name} evaluation completed. Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    # Video Recording
    if record_video:
        print(f"Recording video of best {algo_name} model.")
        env_video = VecVideoRecorder(
            env,
            log_dir,
            video_length=5_000,
            record_video_trigger=lambda x: x == 0,
            name_prefix=f"{algo_name.lower()}_best_model"
        )
        obs = env_video.reset()
        for _ in range(5_000):
            action, states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env_video.step(action)
            if done:
                break
        env_video.close()

    env.close()
    return mean_reward, std_reward


if __name__ == "__main__":
    # Train agents
    # train_dqn_agent(total_timesteps=1_000_000)
    # train_ppo_agent(total_timesteps=1_000_000)

    # Evaluate agents
    evaluate_agent("DQN", "dqn_logs/")
    evaluate_agent("PPO", "ppo_logs/")