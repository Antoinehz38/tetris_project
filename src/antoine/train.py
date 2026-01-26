import gymnasium as gym
from tetris_gymnasium.envs.tetris import Tetris
import numpy as np


from src.antoine.linear_approch import LinearAfterStateAgent


def run_episode(env, agent: LinearAfterStateAgent, max_steps=10_000):
    obs, info = env.reset()
    done = False
    steps = 0
    ep_return = 0.0

    while not done and steps < max_steps:
        # 1) plan depuis l'état courant (simulations sur deepcopy)
        seq, x_after, r_sum_sim, done_sim = agent.choose_sequence(env)

        # 2) exécuter réellement la séquence sur l'env réel
        r_sum_real = 0.0
        for a in seq:
            obs2, r, terminated, truncated, info = env.step(a)
            r_sum_real += float(r)
            done = bool(terminated or truncated)
            if done:
                break

        ep_return += r_sum_real

        # 3) update TD sur la base de l'after-state choisi (x_after)
        #    IMPORTANT: on utilise r_sum_real (réel), pas r_sum_sim
        agent.td_update(x_after=x_after, r_sum=r_sum_real, done=done, env_after_step=env)

        steps += 1

    return ep_return, steps

def train(env, agent, n_episodes=500, eps_decay=0.995, eps_min=0.01, eval_every=10):
    returns = []
    for ep in range(1, n_episodes + 1):
        print(f"Starting episode {ep}/{n_episodes}")
        ep_ret, steps = run_episode(env, agent)
        returns.append(ep_ret)

        agent.epsilon = max(eps_min, agent.epsilon * eps_decay)

        if ep % eval_every == 0:
            mean_ret, std_ret = evaluate(env, agent, n_episodes=20)
            print(f"ep={ep} train_ret={ep_ret:.1f} eval_mean={mean_ret:.1f}±{std_ret:.1f} eps={agent.epsilon:.3f} w={agent.w}")

            # sauvegarde poids
            np.save("weights.npy", agent.w)

    return returns



def evaluate(env, agent, n_episodes=10, max_steps=10_000):
    old_eps = agent.epsilon
    agent.epsilon = 0.0  # greedy, pas d'exploration

    rets = []
    for _ in range(n_episodes):
        obs, info = env.reset()
        done = False
        ep_ret = 0.0
        steps = 0

        while not done and steps < max_steps:
            seq, x_after, _, _ = agent.choose_sequence(env)

            r_sum = 0.0
            for a in seq:
                obs2, r, terminated, truncated, info = env.step(a)
                r_sum += float(r)
                done = bool(terminated or truncated)
                if done:
                    break

            ep_ret += r_sum
            steps += 1

        rets.append(ep_ret)

    agent.epsilon = old_eps
    return float(np.mean(rets)), float(np.std(rets))


if __name__ == "__main__":
    env = gym.make("tetris_gymnasium/Tetris")

    agent = LinearAfterStateAgent(n_features=2,alpha=0.01, gamma=0.99, epsilon=0.1, seed=42)

    returns = train(env, agent, n_episodes=100)

    print("Training completed.")