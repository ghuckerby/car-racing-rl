import numpy as np
import gymnasium as gym

# Native Reward:
# -0.1/step, +1000/N per new tile visited, -100 too far off track (with episode termination)

# Speed:
    # - Does rewarding speed produce faster but less controlled driving?
    # - +0.05 * speed per step (+2.5/step, 25x native time penalty)
class SpeedRewardWrapper(gym.Wrapper):

    # Speed weight for reward
    def __init__(self, env, speed_weight=0.05):
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
    # - -0.5/step when off-track (5x native penalty per off-track step)
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
    # - Subtracts 1 x sum(action delta) per step (discourages large changes)
class SmoothnessRewardWrapper(gym.Wrapper):

    # Smoothness penalty for large changes in steering
    def __init__(self, env, smooth_weight=0.5):
        super().__init__(env)
        self.smooth_weight = smooth_weight
        self.prev_action = None

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Smoothness = delta in angle of steering from previous action
        if self.prev_action is not None:
            steering_delta = abs(float(action[0]) - float(self.prev_action[0]))
            reward -= self.smooth_weight * steering_delta

        self.prev_action = action
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.prev_action = None
        return self.env.reset(**kwargs)

# Time:
    # - Does a time penalty produce urgency, and does urgency help or hinder performance?
    # - -0.1/step (doubles the native time penalty)
class TimeRewardWrapper(gym.Wrapper):

    # Time penalty for each step to encourage faster completion
    def __init__(self, env, time_penalty=0.1):
        super().__init__(env)
        self.time_penalty = time_penalty

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward -= self.time_penalty
        return obs, reward, terminated, truncated, info
    
# Composite (for meta-learning extension):
    # - Combines all reward signals with configurable weights for a meta-learning search
    # - Set any weight to 0.0 to disable the component
class CompositeRewardWrapper(gym.Wrapper):

    def __init__(self, env, speed_weight=0.0, off_track_penalty=0.0, smooth_weight=0.0, time_penalty=0.0):
        super().__init__(env)
        self.speed_weight = speed_weight
        self.off_track_penalty = off_track_penalty
        self.smooth_weight = smooth_weight
        self.time_penalty = time_penalty
        self.prev_action = None
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        car = self.env.unwrapped.car

        # Add each component per step (all match the above wrappers)
        if car is not None:
            
            # Speed component
            speed = np.linalg.norm(car.hull.linearVelocity)
            reward += self.speed_weight * speed

            # Safety component
            on_grass = len(car.hull.contacts) == 0
            if on_grass:
                reward += self.off_track_penalty

        # Smoothness component
        if self.prev_action is not None:
            steering_delta = abs(float(action[0]) - float(self.prev_action[0]))
            reward -= self.smooth_weight * steering_delta

        self.prev_action = action

        # Time component
        reward -= self.time_penalty

        return obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        self.prev_action = None
        return self.env.reset(**kwargs)