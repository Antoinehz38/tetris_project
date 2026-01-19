import gymnasium as gym
import numpy as np

class RewardWrapper(gym.Wrapper):
    def __init__(self, env, reward_fn):
        super().__init__(env)
        self.reward_fn = reward_fn

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward = self.reward_fn(obs, reward, terminated, truncated, info)
        return obs, reward, terminated, truncated, info

