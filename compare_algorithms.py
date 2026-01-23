"""
Script de comparaison des différents algorithmes greedy
"""
import gymnasium as gym
import numpy as np
from tetris_gymnasium.envs.tetris import Tetris

# Import des différentes politiques
import src.tetris_code.policies as policies_basic
import src.tetris_code.policies_improved as policies_improved
import src.tetris_code.policies_advanced as policies_advanced

NB_EPISODES = 5
NB_STEPS_MAX = 500

def test_policy(policy_func, policy_name):
    """Teste une politique et retourne les résultats"""
    env = gym.make("tetris_gymnasium/Tetris", render_mode=None)
    results = []

    print(f"\n{'='*60}")
    print(f"Test de {policy_name}")
    print('='*60)

    for episode in range(NB_EPISODES):
        env.reset(seed=100 + episode)  # Même seed pour tous les algos
        terminated = False
        total_reward = 0
        steps = 0

        while not terminated and steps < NB_STEPS_MAX:
            action = policy_func(env)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

        results.append(total_reward)
        print(f"Épisode {episode + 1}: Score = {total_reward} (en {steps} étapes)")

    return results

if __name__ == "__main__":
    print("="*60)
    print("COMPARAISON DES ALGORITHMES GREEDY")
    print("="*60)
    print(f"Configuration: {NB_EPISODES} épisodes, max {NB_STEPS_MAX} étapes")

    # Tester chaque algorithme
    results_basic = test_policy(policies_basic.policy_greedy, "Algorithme Greedy BASIQUE (prof)")
    results_improved = test_policy(policies_improved.policy_greedy_improved, "Algorithme Greedy AMÉLIORÉ")
    results_advanced = test_policy(policies_advanced.policy_greedy_advanced, "Algorithme Greedy AVANCÉ")

    # Afficher les résultats comparatifs
    print("\n" + "="*60)
    print("RÉSULTATS COMPARATIFS")
    print("="*60)

    algorithms = [
        ("Greedy BASIQUE (prof)", results_basic),
        ("Greedy AMÉLIORÉ", results_improved),
        ("Greedy AVANCÉ", results_advanced),
    ]

    for name, results in algorithms:
        mean = np.mean(results)
        std = np.std(results)
        min_score = np.min(results)
        max_score = np.max(results)
        print(f"\n{name}:")
        print(f"  Moyenne: {mean:.2f} ± {std:.2f}")
        print(f"  Min: {min_score:.2f}")
        print(f"  Max: {max_score:.2f}")
        print(f"  Scores: {results}")

    # Calculer l'amélioration
    print("\n" + "="*60)
    print("AMÉLIORATION PAR RAPPORT À L'ALGORITHME BASIQUE")
    print("="*60)

    baseline_mean = np.mean(results_basic)
    for name, results in algorithms[1:]:
        mean = np.mean(results)
        improvement = ((mean - baseline_mean) / baseline_mean) * 100
        print(f"{name}: {improvement:+.1f}%")

    print("\n" + "="*60)
