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
        "name": "extreme_smoothness_reward",
        "reward_wrapper": functools.partial(SmoothnessRewardWrapper, smooth_weight=1.0),
        "seeds": [0],
        "total_timesteps": 2_000_000
    },
    {
        "name": "time_reward",
        "reward_wrapper": TimeRewardWrapper,
        "seeds": [0],
        "total_timesteps": 2_000_000
    },
    {
        "name": "composite_reward",
        "reward_wrapper": functools.partial(

            # Replace with values from the reward search results
            CompositeRewardWrapper,
            speed_weight=0.07219892421495004,
            off_track_penalty=-1.391544123600146,
            smooth_weight=0.15384640051156695,
            time_penalty=0.06667278608887894
        ),
        "seeds": [0],
        "total_timesteps": 2_000_000
    },
    # {
    #     "name": "time_search_reward",
    #     "reward_wrapper": functools.partial(
    #         CompositeRewardWrapper,
    #         time_penalty=X
    #     ),
    #     "seeds": [0],
    #     "total_timesteps": 2_000_000
    # }
]