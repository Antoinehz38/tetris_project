# Algorithme Greedy Amélioré pour Tetris

## Vue d'ensemble

Cette amélioration de l'algorithme greedy utilise une fonction d'évaluation sophistiquée qui combine plusieurs heuristiques pour optimiser le placement des pièces et minimiser la formation de trous.

## Fichiers créés

- `src/tetris_code/policies_improved.py` : Contient l'algorithme greedy amélioré et toutes les métriques d'évaluation
- `src/tetris_code/evaluate_policy_greedy_improved.py` : Script pour évaluer la performance sur 10 épisodes
- `src/tetris_code/view_episode_policy_greedy_improved.py` : Script pour visualiser un seul épisode

## Améliorations apportées

### Métriques d'évaluation

L'algorithme original utilisait seulement :
- **Hauteur minimale** : `min(heights)`
- **Nombre de trous** : pénalisé avec un facteur de -100

L'algorithme amélioré ajoute les métriques suivantes :

1. **Aggregate Height** (`aggregate_height`)
   - Somme de toutes les hauteurs de colonnes
   - Objectif : garder le plateau le plus bas possible
   - Poids : **-0.51**

2. **Complete Lines** (`complete_lines`)
   - Compte le nombre de lignes complètes prêtes à être effacées
   - Objectif : encourager la complétion de lignes
   - Poids : **+0.76** (positif car c'est bénéfique)

3. **Holes** (`holes`)
   - Nombre de trous (cases vides avec une case pleine au-dessus)
   - Objectif : minimiser les trous
   - Poids : **-0.36**

4. **Bumpiness** (`bumpiness`)
   - Somme des différences absolues entre colonnes adjacentes
   - Objectif : maintenir une surface plane et régulière
   - Poids : **-0.18**

5. **Deep Holes** (`deep_holes`)
   - Trous pondérés par leur profondeur
   - Les trous profonds sont plus difficiles à combler
   - Objectif : pénaliser fortement les trous profonds
   - Poids : **-0.48**

6. **Wells** (`wells`)
   - Colonnes entourées de colonnes plus hautes
   - Peuvent être utiles pour les pièces en I mais problématiques si trop profonds
   - Poids : **-0.34**

7. **Max Height** (`max_height`)
   - Hauteur maximale parmi toutes les colonnes
   - Objectif : éviter que le plateau ne devienne trop haut (game over)
   - Poids : **-0.51**

### Fonction d'évaluation finale

```python
score = (
    -0.51 * aggregate_height +
    +0.76 * complete_lines +
    -0.36 * holes +
    -0.18 * bumpiness +
    -0.48 * deep_holes +
    -0.34 * wells +
    -0.51 * max_height
)
```

### Protection contre le Game Over

L'algorithme amélioré détecte également si une séquence d'actions mène à un game over immédiat et lui attribue un score très négatif (-999999) pour l'éviter.

## Utilisation

### Visualiser un épisode

```bash
python src/tetris_code/view_episode_policy_greedy_improved.py
```

### Évaluer la performance

```bash
python src/tetris_code/evaluate_policy_greedy_improved.py
```

Cela exécutera 10 épisodes et affichera :
- Le score de chaque épisode
- Le score moyen
- Le score maximum et minimum
- L'intervalle de confiance

## Comparaison avec l'algorithme original

Pour comparer les performances :

1. Exécuter l'algorithme original :
```bash
python src/tetris_code/evaluate_policy_greedy.py
```

2. Exécuter l'algorithme amélioré :
```bash
python src/tetris_code/evaluate_policy_greedy_improved.py
```

## Avantages de l'approche améliorée

1. **Minimisation des trous** : Utilise plusieurs métriques (holes, deep_holes) pour détecter et éviter différents types de trous

2. **Surface plane** : La métrique de bumpiness encourage un profil régulier, facilitant le placement futur des pièces

3. **Gestion proactive de la hauteur** : Surveille à la fois la hauteur totale (aggregate) et la hauteur maximale pour éviter de perdre

4. **Complétion de lignes** : Encourage activement la création de lignes complètes, ce qui est le mécanisme principal de scoring dans Tetris

5. **Pondération équilibrée** : Les poids ont été choisis pour équilibrer les différents objectifs (éviter les trous vs compléter des lignes vs garder le plateau bas)

## Axes d'amélioration futurs

- Ajuster les poids par apprentissage automatique (algorithme génétique, gradient descent)
- Ajouter une anticipation sur plusieurs pièces (lookahead)
- Intégrer une analyse de la pièce suivante (next piece) pour optimiser le placement
- Ajouter des patterns spécifiques pour certaines situations (T-spin, combo setup)
