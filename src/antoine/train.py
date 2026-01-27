import gymnasium as gym
from tetris_gymnasium.envs.tetris import Tetris
import numpy as np


from src.antoine.linear_approch import LinearAfterStateAgent, LinearAfterStateReinforceAgent

MAX_STEPS_PER_EPISODE = 1_000


def run_episode(env, agent: LinearAfterStateAgent, max_steps=MAX_STEPS_PER_EPISODE):
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
            mean_ret, std_ret = evaluate(env, agent, n_episodes=10)
            print(f"ep={ep} train_ret={ep_ret:.1f} eval_mean={mean_ret:.1f}±{std_ret:.1f} eps={agent.epsilon:.3f} w={agent.w}")

            # sauvegarde poids
            np.save("weights.npy", agent.w)

    return returns



def evaluate(env, agent, n_episodes=10, max_steps=MAX_STEPS_PER_EPISODE):
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

def run_episode_reinforce(env, agent: LinearAfterStateReinforceAgent, max_env_steps=MAX_STEPS_PER_EPISODE):
    """
    Joue un épisode, stocke la trajectoire de décisions, puis met à jour w à la fin.
    """
    obs, info = env.reset()
    done = False
    env_steps = 0
    G = 0.0
    traj = []

    while not done and env_steps < max_env_steps:
        seq, x_chosen, xs, p, idx = agent.choose_sequence(env)

        # exécution réelle
        for a in seq:
            obs2, r, terminated, truncated, info = env.step(a)
            G += float(r)
            env_steps += 1
            done = bool(terminated or truncated)
            if done or env_steps >= max_env_steps:
                break

        traj.append((x_chosen, xs, p, idx))
    if env_steps >= max_env_steps:
        print("reached max env steps")
    adv, grad = agent.update_episode(traj, G)
    return G, env_steps, adv




def train_rl(env, agent: LinearAfterStateReinforceAgent, n_episodes=500, eval_every=10):
    returns = []
    for ep in range(1, n_episodes + 1):
        print(f"Starting episode {ep}/{n_episodes}")
        ep_ret, env_steps, adv = run_episode_reinforce(env, agent)
        returns.append(ep_ret)
        print(agent.w)

        if ep % eval_every == 0:
            print("Evaluating...")
            mean_ret, std_ret = evaluate_rl(env, agent, n_episodes=2)
            print(f"ep={ep} train_ret={ep_ret:.1f} eval_mean={mean_ret:.1f}±{std_ret:.1f} adv={adv:.3f} w={agent.w}")
            np.save("weights.npy", agent.w)

    return returns


def evaluate_rl(env, agent: LinearAfterStateReinforceAgent, n_episodes=2, max_env_steps=MAX_STEPS_PER_EPISODE):
    # évaluation greedy: on prend argmax (équivalent à beta -> inf)
    old_beta = agent.beta
    old_rng = agent.rng
    agent.beta = 1e9  # quasi-argmax via softmax très sharp

    rets = []
    for _ in range(n_episodes):
        obs, info = env.reset()
        done = False
        env_steps = 0
        G = 0.0

        while not done and env_steps < max_env_steps:
            seq, x_chosen, xs, p, idx = agent.choose_sequence(env)  # avec beta énorme => quasi greedy

            for a in seq:
                obs2, r, terminated, truncated, info = env.step(a)
                G += float(r)
                env_steps += 1
                done = bool(terminated or truncated)
                if done or env_steps >= max_env_steps:
                    break

        rets.append(G)

    agent.beta = old_beta
    return float(np.mean(rets)), float(np.std(rets))


if __name__ == "__main__":
    env = gym.make("tetris_gymnasium/Tetris")

    agent = LinearAfterStateReinforceAgent(
        n_features=9,
        alpha=0.03,
        beta=2.0,
        baseline_lr=0.05,
        seed=42
    )

    agent.w = np.array([ 0.9999983,   0.0,  -0.9388498,  -8.488421,   -0.8680423,  -0.29565483,-0.28252172,  0.05340564, -0.53585887], dtype=np.float32)

    returns = train_rl(env, agent, n_episodes=100, eval_every=10)
