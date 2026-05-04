from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.atari_wrappers import WarpFrame
from envs.telemetry_collection import TelemetryCollector
import os

def make_env(reward_wrapper = None, telemetry = True, env_seed=None):
    telemetry_instance = []

    def wrapper_class(env):
        env = WarpFrame(env)

        # Add reward wrapper for experiment
        if reward_wrapper is not None:
            env = reward_wrapper(env)
        
        # Add telemetry wrapper to collect data during training and evaluation
        if telemetry:
            tel = TelemetryCollector(env)
            telemetry_instance.append(tel)
            return tel
        
        return env

    # Seeding for evaluation only
    env = make_vec_env("CarRacing-v3", n_envs=1, wrapper_class=wrapper_class, seed=env_seed)
    env = VecFrameStack(env, n_stack=4)
    env = VecTransposeImage(env)
    tel_ref = telemetry_instance[0] if telemetry_instance else None
    return env, tel_ref
    
def train_ppo_agent(experiment_name, total_timesteps=2_000_000, reward_wrapper=None, seed=0):

    # Logging for results and checkpoints
    log_dir = os.path.join("results", experiment_name)
    os.makedirs(log_dir, exist_ok=True)

    # Environments
    env, _ = make_env(reward_wrapper)
    env_eval, _ = make_env(reward_wrapper)
    
    # Evaluation Callback
    eval_callback = EvalCallback(
        env_eval,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=100_000,
        deterministic=True,
        render=False,
        n_eval_episodes=5
    )

    # Model Training
    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        seed=seed,
        ent_coef=0.01,
        tensorboard_log=os.path.join(log_dir, "tensorboard")
    )
    
    # Train and save the final model
    model.learn(total_timesteps=total_timesteps, callback=eval_callback, progress_bar=True)
    model.save(os.path.join(log_dir, "final_model.zip"))
    print(f"PPO training completed. Model saved to {os.path.join(log_dir, 'final_model.zip')}")

    env.close()
    env_eval.close()
    return log_dir