import functools

from envs.reward_wrappers import (
    SpeedRewardWrapper,
    SafetyRewardWrapper,
    SmoothnessRewardWrapper,
    TimeRewardWrapper,
    CompositeRewardWrapper
)

# Definitions for each experiment
EXPERIMENTS = [

    # Individual reward experiments
    {
        "name": "baseline",
        "reward_wrapper": None,
        "seeds": [0],
        "total_timesteps": 2_000_000
    },
    {
        "name": "speed_reward",
        "reward_wrapper": SpeedRewardWrapper,
        "seeds": [0],
        "total_timesteps": 2_000_000
    },
    {
        "name": "safety_reward",
        "reward_wrapper": SafetyRewardWrapper,
        "seeds": [0],
        "total_timesteps": 2_000_000
    },
    {
        "name": "smoothness_reward",
        "reward_wrapper": SmoothnessRewardWrapper,
        "seeds": [0],
        "total_timesteps": 2_000_000
    },
    {
        "name": "time_reward",
        "reward_wrapper": TimeRewardWrapper,
        "seeds": [0],
        "total_timesteps": 2_000_000
    },

    # Composite reward experiments with weights from Optuna search results
    {
        "name": "composite_reward",
        "reward_wrapper": functools.partial(
            CompositeRewardWrapper,
            speed_weight=0.07219892421495004,
            off_track_penalty=-1.391544123600146,
            smooth_weight=0.15384640051156695,
            time_penalty=0.06667278608887894
        ),
        "seeds": [0],
        "total_timesteps": 2_000_000
    },
    {
        "name": "time_search_reward",
        "reward_wrapper": functools.partial(
            CompositeRewardWrapper,
            time_penalty=0.4116426000513264
        ),
        "seeds": [0],
        "total_timesteps": 2_000_000
    },
    {
        "name": "time_safety_search_reward",
        "reward_wrapper": functools.partial(
            CompositeRewardWrapper,
            time_penalty=0.19059897170635648,
            off_track_penalty=-0.506287406327415
        ),
        "seeds": [0],
        "total_timesteps": 2_000_000
    }
]