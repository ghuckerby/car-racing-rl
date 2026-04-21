import numpy as np
import gymnasium as gym

# Speed:
# - Does rewarding speed produce faster but less controlled driving?
class SpeedRewardWrapper(gym.Wrapper):

    # Speed weight for reward
    def __init__(self, env, speed_weight=0.1):
        super().__init__(env)
        self.speed_weight = speed_weight

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        car = self.env.unwrapped.car

        if car is not None:
            speed = np.linalg.norm(car.hull.linearVelocity)
            reward += self.speed_weight * speed

        return obs, reward, terminated, truncated, info

# Safety:
# - Does penalising off-track behaviour produce more conservative driving?
class SafetyRewardWrapper(gym.Wrapper):

    # Off-track penalty for driving on grass
    def __init__(self, env, off_track_penalty=-0.5):
        super().__init__(env)
        self.off_track_penalty = off_track_penalty

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        car = self.env.unwrapped.car

        if car is not None:
            on_grass = len(car.hull.contacts) == 0
            if on_grass:
                reward += self.off_track_penalty

        return obs, reward, terminated, truncated, info

# Smoothness:
# - Does rewarding smooth steering produce more stable driving?
class SmoothnessRewardWrapper(gym.Wrapper):

    # Smoothness penalty for large changes in steering
    def __init__(self, env, smooth_weight=0.1):
        super().__init__(env)
        self.smooth_weight = smooth_weight
        self.prev_action = None

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Smoothness = delta in angle of steering from previous action
        if self.prev_action is not None:
            action_delta = np.sum(np.abs(np.array(action) - np.array(self.prev_action)))
            reward -= self.smooth_weight * action_delta

        self.prev_action = action
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.prev_action = None
        return self.env.reset(**kwargs)

# Time:
# - Does a time penalty produce urgency, and does urgency help or hinder performance?
class TimeRewardWrapper(gym.Wrapper):

    # Time penalty for each step to encourage faster completion
    def __init__(self, env, time_penalty=0.01):
        super().__init__(env)
        self.time_penalty = time_penalty

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward -= self.time_penalty
        return obs, reward, terminated, truncated, info