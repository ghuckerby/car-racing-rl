import numpy as np
import gymnasium as gym
import json
import os

class TelemetryCollector(gym.Wrapper):
    """Record telemetry data during runs and save to JSON"""

    def __init__(self, env):
        super().__init__(env)
        self.episode_logs = []
        self.reset_telemetry_stats()

    # Reset telemetry stats at the start of each episode
    def reset_telemetry_stats(self):
        self.step_speeds = []
        self.step_lateral_velocitys = []
        self.step_steering = []
        self.step_throttle = []
        self.step_brake = []
        self.step_off_track = []
        self.step_action_deltas = []
        self.prev_action = None
        self.step_count = 0
        self.completion_steps = None

    # Collect telemetry at each step and save episode summary at the end of each episode
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        car = self.env.unwrapped.car

        # Collect telemetry data
        if car is not None:
            vel = car.hull.linearVelocity
            speed = float(np.linalg.norm(vel))
            lateral_velocity = float(abs(vel[0]))
            off_track = len(car.hull.contacts) == 0
        else:
            speed = 0.0
            lateral_velocity = 0.0
            off_track = False

        # Action data
        action = np.array(action)
        steering = float(action[0])
        throttle = float(action[1])
        brake = float(action[2])

        if self.prev_action is not None:
            delta = float(np.sum(np.abs(action - self.prev_action)))
        else:
            delta = 0.0
        self.prev_action = action

        self.step_speeds.append(speed)
        self.step_lateral_velocitys.append(lateral_velocity)
        self.step_steering.append(steering)
        self.step_throttle.append(throttle)
        self.step_brake.append(brake)
        self.step_off_track.append(off_track)
        self.step_action_deltas.append(delta)
        self.step_count += 1

        # Check track completion
        unwrapped = self.env.unwrapped
        if unwrapped.track is not None and unwrapped.tile_visited_count >= len(unwrapped.track):
            if self.completion_steps is None:
                self.completion_steps = self.step_count

        # If episode ended, save summary of episode telemetry
        if terminated or truncated:
            unwrapped = self.env.unwrapped
            total_tiles = len(unwrapped.track) if unwrapped.track is not None else 0
            visited_tiles = unwrapped.tile_visited_count if unwrapped.track is not None else 0
            track_completed = self.completion_steps is not None
            completion_pct = float(visited_tiles / total_tiles) if total_tiles > 0 else 0.0

            self.episode_logs.append({
                "episode": len(self.episode_logs),
                "steps": self.step_count,
                "track_completed": track_completed,
                "completion_steps": self.completion_steps,
                "track_completion_percentage": completion_pct,
                "mean_speed": float(np.mean(self.step_speeds)),
                "max_speed": float(np.max(self.step_speeds)),
                "mean_lateral_velocity": float(np.mean(self.step_lateral_velocitys)),
                "off_track_percentage": float(np.mean(self.step_off_track)),
                "mean_steering_change": float(np.mean(self.step_action_deltas)),
                "mean_throttle": float(np.mean(self.step_throttle)),
                "mean_brake": float(np.mean(self.step_brake))
            })
            self.reset_telemetry_stats()

        return obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        self.reset_telemetry_stats()
        return self.env.reset(**kwargs)
    
    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.episode_logs, f, indent=4)
        print(f"Telemetry data saved to {path}")