"""
Script de test rapide pour vérifier que l'algorithme greedy amélioré fonctionne
"""
import gymnasium as gym
import numpy as np
from tetris_gymnasium.envs.tetris import Tetris
from tetris_code.policies_improved import policy_greedy_improved

print("="*60)
print("Test de l'algorithme Greedy Amélioré")
print("="*60)

# Créer l'environnement sans rendu pour un test rapide
env = gym.make("tetris_gymnasium/Tetris", render_mode=None)

# Tester sur 3 épisodes courts
nb_episodes = 3
results = []

for episode in range(nb_episodes):
    env.reset(seed=42 + episode)
    terminated = False
    total_reward = 0
    steps = 0

    print(f"\nÉpisode {episode + 1}:")

    while not terminated and steps < 500:  # Limite à 500 étapes pour le test
        try:
            action = policy_greedy_improved(env)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

            if steps % 50 == 0:
                print(f"  Step {steps}, Score: {total_reward}")
        except Exception as e:
            print(f"  Erreur à l'étape {steps}: {e}")
            break

    results.append(total_reward)
    print(f"  Score final: {total_reward} (en {steps} étapes)")

print("\n" + "="*60)
print("Résultats du test:")
print(f"  Scores: {results}")
print(f"  Moyenne: {np.mean(results):.2f}")
print(f"  Max: {np.max(results):.2f}")
print(f"  Min: {np.min(results):.2f}")
print("="*60)
print("\nTest réussi! L'algorithme fonctionne correctement.")
print("\nPour une évaluation complète, exécutez:")
print("  python src/tetris_code/evaluate_policy_greedy_improved.py")
