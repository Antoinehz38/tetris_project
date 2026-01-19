import numpy as np
import gymnasium as gym

def added_reward(obs, reward, terminated, truncated, info):
    # Example: Penalize for holes in Tetris
    board = obs['board']  # Assuming observation has a 'board' key
    holes = count_holes(board)
    return reward - holes * 0.1  # Penalize 0.1 for each hole

def standard_reward(obs, reward, terminated, truncated, info):
    # Standard reward is just the game score
    return reward

def count_holes(board):
    """Calcule le nombre de trous sur le plateau."""
    nb_holes = 0
    # Parcourir les lignes et les colonnes du plateau
    for i in range(board.shape[0]):
        for j in range(board.shape[1]):
            # Si un pixel est vide (0) mais qu'il y a un bloc au-dessus (!= 0), c'est un trou
            if (i > 1) and (board[i, j] == 0) and (board[i-1, j] != 0):
                nb_holes = nb_holes + 1
    return nb_holes

def heights(board):
    """Calcule la 'hauteur' (profondeur du vide) de chaque colonne."""
    heights_column = []
    # Parcourir les colonnes du plateau
    for i in range(board.shape[1]):
        # Ignorer les colonnes de "rembourrage" (padding) qui ont un "1" en haut
        if board[0, i] != 1:
            # Commencer par le deuxième point le plus haut de la colonne courante
            j = 2
            # Descendre le long de la colonne jusqu'à rencontrer un pixel non "0"
            while (j < board.shape[0]) and (board[j, i] == 0):
                j = j + 1
            # Stocker le résultat
            heights_column.append(j)
    return heights_column
