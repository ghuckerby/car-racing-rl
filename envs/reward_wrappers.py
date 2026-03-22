import numpy as np
import gymnasium as gym

# Speed:
# - Does rewarding speed produce faster but less controlled driving?
class SpeedRewardWrapper(gym.Wrapper):
    super().__init__()

# Safety:
# - Does penalising off-track behaviour produce more conservative driving?
class SafetyRewardWrapper(gym.Wrapper):
    super().__init__()

# Smoothness:
# - Does rewarding smooth steering produce more stable driving?
class SmoothnessRewardWrapper(gym.Wrapper):
    super().__init__()

# Time:
# - Does a time penalty produce urgency, and does urgency help or hinder performance?
class TimeRewardWrapper(gym.Wrapper):
    super().__init__()

# Drift:
# - Can an agent learn an unconventional driving style from reward alone?
class DriftRewardWrapper(gym.Wrapper):
    super().__init__()