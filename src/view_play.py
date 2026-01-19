import cv2
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
from tetris_gymnasium.envs.tetris import Tetris
from stable_baselines3 import DQN, PPO

if __name__ == "__main__":
    env = gym.make(
        "tetris_gymnasium/Tetris",
        render_mode="human",
        render_upscale=40
    )
    env = FlattenObservation(env)   

    observation, info = env.reset()

    model = PPO.load("runs/tetris/ppo_tetris_final.zip")

    terminated = False
    total_reward = 0

    while not terminated:
        env.render()
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)

        key = cv2.waitKey(1)
        total_reward += reward
        print("Score:", total_reward, end="\r", flush=True)

    print("\nGame Over!")
    print("Final Score:", total_reward)
