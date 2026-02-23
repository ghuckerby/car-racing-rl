import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.atari_wrappers import WarpFrame
import os

def train_dqn_agent(total_timesteps=1_000_000):

    log_dir = "dqn_logs/"
    env_kwargs_dict = {"continuous": False}

    # Training Environment
    env = make_vec_env(
        "CarRacing-v3",
        n_envs=1,
        env_kwargs=env_kwargs_dict,
        wrapper_class=WarpFrame
    )
    env = VecFrameStack(env, n_stack=4)
    env = VecTransposeImage(env)

    # Evaluation Environment
    env_eval = make_vec_env(
        "CarRacing-v3",
        n_envs=1,
        env_kwargs=env_kwargs_dict,
        wrapper_class=WarpFrame
    )
    env_eval = VecFrameStack(env_eval, n_stack=4)
    env_eval = VecTransposeImage(env_eval)

    # Callbacks
    eval_freq = 100_000
    eval_callback = EvalCallback(
        env_eval,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=eval_freq,
        deterministic=True,
        render=False,
        n_eval_episodes=5
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=eval_freq,
        save_path=os.path.join(log_dir, "checkpoints"),
    )
    callback_list = CallbackList([eval_callback, checkpoint_callback])

    # Model Training
    model = DQN(
        "CnnPolicy",
        env,
        verbose=1,
        buffer_size=150_000,
        learning_starts=10_000,
        batch_size=64,
        tensorboard_log=os.path.join(log_dir, "tensorboard")
    )
    print(f"Training DQN agent for {total_timesteps} timesteps.")
    model.learn(total_timesteps=total_timesteps, callback=callback_list, progress_bar=True)

    # Save Model
    model.save(os.path.join(log_dir, "final_model.zip"))
    print(f"DQN training completed. Model saved to {os.path.join(log_dir, 'final_model.zip')}")

    env.close()
    env_eval.close()