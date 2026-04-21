import os
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecVideoRecorder

from agents.train_ppo import make_env

def evaluate_agent(experiment_name, reward_wrapper=None, n_eval_episodes=10, record_video=True):

    # Create env and load model
    log_dir = os.path.join("results", experiment_name)

    best_model_path = os.path.join(log_dir, "best_model.zip")
    final_model_path = os.path.join(log_dir, "final_model.zip")

    # Use best model if it exists otherwise use final model
    if os.path.exists(best_model_path):
        model_path = best_model_path
    elif os.path.exists(final_model_path):
        model_path = final_model_path
    else:
        raise FileNotFoundError(f"No model found for {experiment_name} in {log_dir}")

    env, tel = make_env(reward_wrapper, telemetry=True)
    model = PPO.load(model_path, env=env)
    
    # Reward statistics
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes)

    # Save telemetry collected during episodes
    if tel is not None:
        tel.save(os.path.join(log_dir, "telemetry.json"))

    # Record Video
    if record_video:

        env_video = VecVideoRecorder(
            env, log_dir, 
            video_length=5_000,
            record_video_trigger=lambda x: x == 0,
            name_prefix=f"{experiment_name}_best"
        )

        obs = env_video.reset()

        for _ in range(5_000):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _ = env_video.step(action)
            if done[0]:
                break
        env_video.close()
        
    env.close()
    return mean_reward, std_reward