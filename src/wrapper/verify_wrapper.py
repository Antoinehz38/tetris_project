import numpy as np
from obs import TetrisObservationWrapper
import gymnasium as gym

# On crée une fausse classe pour simuler l'accès aux méthodes sans lancer le jeu entier
class MockWrapper(TetrisObservationWrapper):
    def __init__(self):
        # On triche un peu pour initialiser juste ce qu'il faut
        self.n_cols = 10
        pass

def test_calculs():
    wrapper = MockWrapper()
    
    # 1. Création d'une mini grille de test (6 lignes, 10 colonnes pour simplifier)
    # 0 = vide, 1 = bloc
    dummy_grid = np.zeros((6, 10))
    
    # On met une tour à gauche (hauteur 3)
    dummy_grid[3:, 0] = 1 
    
    # On met une tour avec un TROU à la colonne 2
    dummy_grid[3, 2] = 1  # Bloc en haut
    dummy_grid[4, 2] = 0  # TROU ICI !
    dummy_grid[5, 2] = 1  # Bloc en bas
    
    print("--- Test de la logique ---")
    
    # Test Hauteurs
    heights = wrapper._get_heights(dummy_grid)
    print(f"Hauteurs calculées : {heights}")
    # Attendu col 0: 3, col 2: 3. Le reste 0.
    
    # Test Trous
    holes = wrapper._get_holes(dummy_grid, heights)
    print(f"Trous calculés : {holes}")
    # Attendu : 1 (le trou en (4, 2))
    
    # Test Bumpiness
    bump = wrapper._get_bumpiness(heights)
    print(f"Rugosité : {bump}")
    # Col 0 (3) -> Col 1 (0) = diff 3
    # Col 1 (0) -> Col 2 (3) = diff 3
    # Col 2 (3) -> Col 3 (0) = diff 3
    # Total attendu = 9
    
    if holes == 1 and heights[0] == 3:
        print("\n✅ SUCCÈS : Tes fonctions marchent parfaitement !")
    else:
        print("\n❌ ÉCHEC : Il y a un problème de logique.")

if __name__ == "__main__":
    test_calculs()