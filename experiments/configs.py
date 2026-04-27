from envs.reward_wrappers import (
    SpeedRewardWrapper,
    SafetyRewardWrapper,
    SmoothnessRewardWrapper,
    TimeRewardWrapper
)

# Definitions for each experiment
EXPERIMENTS = [
    {
        "name": "baseline",
        "reward_wrapper": None,
        "seeds": [0, 1],
        "total_timesteps": 2_000_000
    },
    {
        "name": "speed_reward",
        "reward_wrapper": SpeedRewardWrapper,
        "seeds": [0, 1],
        "total_timesteps": 2_000_000
    },
    {
        "name": "safety_reward",
        "reward_wrapper": SafetyRewardWrapper,
        "seeds": [0, 1],
        "total_timesteps": 2_000_000
    },
    {
        "name": "smoothness_reward",
        "reward_wrapper": SmoothnessRewardWrapper,
        "seeds": [0, 1],
        "total_timesteps": 2_000_000
    },
    {
        "name": "time_reward",
        "reward_wrapper": TimeRewardWrapper,
        "seeds": [0, 1],
        "total_timesteps": 2_000_000
    }
]