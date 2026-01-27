import os
import numpy as np
import gymnasium as gym
from tetris_gymnasium.envs.tetris import Tetris
import multiprocessing as mp

MAX_STEPS_PER_EPISODE = 10_000

from src.antoine.evolution.evolution_approch import EvolutionAgent


def _eval_worker(args):
    w, n, id = args
    env = gym.make("tetris_gymnasium/Tetris")
    try:
        return float(np.mean([run_episode(env, w, id) for _ in range(n)]))
    finally:
        env.close()

def run_episode(env, w, id, max_env_steps=MAX_STEPS_PER_EPISODE):
    agent = EvolutionAgent(n_features=9, W0=w)
    obs, info = env.reset()
    done = False
    env_steps = 0
    G = 0.0
    traj = []

    while not done and env_steps < max_env_steps:
        seq = agent.choose_sequence(env)
        # exécution réelle
        for a in seq:
            obs2, r, terminated, truncated, info = env.step(a)
            G += float(r)
            env_steps += 1
            done = bool(terminated or truncated)
            if env_steps % 1000 == 0:
                print(f"env steps: {env_steps}, current G: {G} (worker {id})")
            if done or env_steps >= max_env_steps:
                break

        
    if env_steps >= max_env_steps:
        print("reached max env steps")

    return G


def train_optimized(env, agent: EvolutionAgent, n_episodes=500, sigma=0.1, lr=0.01, n_eval=5):
    # On utilise un nombre pair de workers pour les paires (+, -)
    nb_workers = os.cpu_count() or 2
    if nb_workers % 2 != 0: nb_workers -= 1 # Assurer un nombre pair
    
    half_pop = nb_workers // 2
    
    ctx = mp.get_context("spawn")

    with ctx.Pool(processes=nb_workers) as pool:
        for ep in range(1, n_episodes + 1):
            print(f"=== Episode {ep}/{n_episodes} ===")
            
            # 1. Générer la moitié des bruits
            noises = [np.random.randn(agent.n_features).astype(np.float32) for _ in range(half_pop)]
            
            # 2. Créer les candidats par PAIRES (Mirrored Sampling)
            # Candidat i   = w + sigma * noise
            # Candidat i+1 = w - sigma * noise
            candidates = []
            for n in noises:
                candidates.append(agent.w + (sigma * n)) # Positif
                candidates.append(agent.w - (sigma * n)) # Négatif (Miroir)
            
            # 3. Évaluation
            all_configs = [(c, n_eval, i) for i, c in enumerate(candidates)]
            scores = np.array(pool.map(_eval_worker, all_configs))
            
            # Sauvegarde périodique
            if ep % 1 == 0:
                print(f"Saving weights ep {ep}")
                np.save(f"weights_{ep}.npy", agent.w)

            # 4. Reconstruction du Gradient (Méthode Mirrored)
            gradient = np.zeros_like(agent.w)
            
            # On parcourt les bruits. Pour chaque bruit 'n', on a deux scores :
            # score_pos (index 2*i) et score_neg (index 2*i + 1)
            for i in range(half_pop):
                score_pos = scores[2*i]
                score_neg = scores[2*i+1]
                
                # Formule Antithetic : (F(w + noise) - F(w - noise)) * noise
                # C'est beaucoup plus stable que (F(w+noise) - F(w))
                gradient += (score_pos - score_neg) * noises[i]

            # Normalisation par le nombre de paires et sigma
            # C'est important pour que LR reste stable quand on change sigma
            gradient /= (2 * half_pop * sigma)
            
            # 5. Normalisation globale du gradient (Optionnel mais conseillé pour Tetris)
            # Évite les explosions quand les scores passent de 1000 à 100 000
            grad_norm = np.linalg.norm(gradient)
            if grad_norm > 1.0:
                gradient = gradient / grad_norm

            # Update
            agent.w += lr * gradient
            
            mean_score = np.mean(scores)
            max_score = np.max(scores)
            print(f"Ep {ep} | Avg: {mean_score:.0f} | Max: {max_score:.0f} | NormGrad: {grad_norm:.2f}")

    return agent.w


if __name__ == "__main__":
    env = gym.make("tetris_gymnasium/Tetris")

    agent = EvolutionAgent( n_features=9, W0=None)

    
    agent.w = np.array([0, -2, -1, 0, -1, -0.5, -0.5, -1, -1], dtype=np.float32)

    
    returns = train_optimized(env, agent, n_episodes=10, sigma=0.3, lr=0.05, n_eval=5)