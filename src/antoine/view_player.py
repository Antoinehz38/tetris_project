import cv2
import gymnasium as gym
import numpy as np
from tetris_gymnasium.envs.tetris import Tetris  # force l'enregistrement
from src.antoine.linear_approch import LinearAfterStateAgent


def run_sequence(env, seq):
    """Exécute une séquence jusqu'à fin/stop et renvoie (terminated, truncated, reward_sum)."""
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
    # ENV
    env = gym.make("tetris_gymnasium/Tetris", render_mode="human", render_upscale=40)
    obs, info = env.reset()

    # AGENT (ADAPTE n_features si tes features changent)
    agent = LinearAfterStateAgent(n_features=2, alpha=0.0, gamma=0.99, epsilon=0.0, seed=42)

    # Charge des poids si tu en as
    try:
        agent.w = np.load("weights.npy")
        print("Loaded weights:", agent.w)
    except Exception as e:
        print("No weights.npy loaded (using default w):", agent.w)

    terminated = False
    truncated = False
    total_reward = 0.0
    t = 0

    while not (terminated or truncated):
        t += 1
        env.render()

        # Choix d'une séquence (greedy car epsilon=0)
        seq, x_after, r_sum_sim, done_sim = agent.choose_sequence(env)

        # Exécuter la séquence complète (comme ton policy_greedy)
        terminated, truncated, r_sum_real, info = run_sequence(env, seq)

        total_reward += r_sum_real
        print(f"Score: {total_reward}", end="\r", flush=True)

        cv2.waitKey(1)

    print("\nGame Over!")
    print("Final Score:", total_reward)
