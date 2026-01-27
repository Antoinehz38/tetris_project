import cv2
import gymnasium as gym
import numpy as np
from tetris_gymnasium.envs.tetris import Tetris  # force l'enregistrement
from src.antoine.evolution.evolution_approch import EvolutionAgent  # <-- adapte le chemin


def run_sequence(env, seq):
    r_sum = 0.0
    terminated = False
    truncated = False
    info = {}
    for a in seq:
        obs, r, terminated, truncated, info = env.step(a)
        r_sum += float(r)
        if terminated or truncated:
            break
    return terminated, truncated, r_sum, info




if __name__ == "__main__":
    env = gym.make("tetris_gymnasium/Tetris", render_mode="human", render_upscale=40)
    
    obs, info = env.reset()
    print("Attributs de env.unwrapped :")
    print(dir(env.unwrapped))

    # Si tu vois 'game', 'engine' ou 'tetris', inspecte-les aussi :
    if hasattr(env.unwrapped, 'game'):
        print("\nAttributs de env.unwrapped.game :")
        print(dir(env.unwrapped.game))

    agent = EvolutionAgent(n_features=9, W0=None)

    agent.w = np.load("weights_1.npy")
    print("Loaded weights:", agent.w)

    agent.w = np.array([0, -2, -1, 0, -1, -0.5, -0.5, -1, -1], dtype=np.float32)


    terminated = False
    truncated = False
    total_reward = 0.0
    t = 0

    while not (terminated or truncated):
        t += 1
        env.render()

        # Greedy pur (pas sampling)
        seq = agent.choose_sequence(env)

        terminated, truncated, r_sum_real, info = run_sequence(env, seq)

        total_reward += r_sum_real
        print(f"Score: {total_reward}", end="\r", flush=True)

        cv2.waitKey(1)

    print("\nGame Over!")
    print("Final Score:", total_reward)
