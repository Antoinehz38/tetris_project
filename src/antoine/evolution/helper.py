import numpy as np

class FastTetrisSim:
    @staticmethod
    def get_drop_state(board: np.ndarray, piece_matrix: np.ndarray, target_x: int):
        """
        Simule directement le résultat d'une pièce placée à target_x.
        Retourne (None, None) si la position est impossible (collision immédiate ou hors limites).
        """
        h_p, w_p = piece_matrix.shape
        h_b, w_b = board.shape

        # 1. Vérifier si la pièce rentre horizontalement
        if target_x < 0 or target_x + w_p > w_b:
            return None, None # Hors limites

        # 2. Trouver la hauteur de chute (Hard Drop)
        # On commence tout en haut (y=0) et on descend tant que c'est libre
        y = 0
        
        # Optimisation : Collision check
        # On vérifie d'abord si la pièce peut spawn à y=0 (Game Over check simplifié)
        if FastTetrisSim.check_collision(board, piece_matrix, target_x, y):
            return None, None

        while not FastTetrisSim.check_collision(board, piece_matrix, target_x, y + 1):
            y += 1
            
        # 3. Placer la pièce
        final_board = board.copy()
        for r in range(h_p):
            for c in range(w_p):
                if piece_matrix[r, c] != 0:
                    final_board[y + r, target_x + c] = 1
                    
        return final_board, y

    @staticmethod
    def check_collision(board, piece, x, y):
        h_p, w_p = piece.shape
        h_b, w_b = board.shape
        for r in range(h_p):
            for c in range(w_p):
                if piece[r, c] != 0:
                    b_r, b_c = y + r, x + c
                    if b_c < 0 or b_c >= w_b or b_r >= h_b: return True
                    if b_r >= 0 and board[b_r, b_c] != 0: return True
        return False