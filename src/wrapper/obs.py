import gymnasium as gym
import numpy as np

class TetrisObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        
        # --- CONFIGURATION ---
        self.n_cols = 10
        self.col_start = 4
        self.col_end = 14 
        
        # --- CALCUL DE LA TAILLE DU VECTEUR ---
        # 1. Hauteurs (10 valeurs)
        # 2. Nombre de trous (1 valeur)
        # 3. Rugosité (Bumpiness) (1 valeur)
        # 4. Hauteur Max (1 valeur)
        # 5. Hauteur Cumulée (1 valeur) <--- AJOUTÉ
        # Total Stats = 14
        
        n_stats = 14
        n_holder = 16  # 4x4
        n_queue = 64   # 4x16
        
        total_size = n_stats + n_holder + n_queue # 14 + 16 + 64 = 94
        
        self.observation_space = gym.spaces.Box(
            low=0, high=100, shape=(total_size,), dtype=np.float32
        )

    def observation(self, obs):
        # 1. NETTOYAGE
        full_board = obs['board']
        active_mask = obs['active_tetromino_mask']
        
        play_area = full_board[:, self.col_start:self.col_end]
        mask_area = active_mask[:, self.col_start:self.col_end]
        
        # On garde seulement les blocs posés
        settled_blocks = ((play_area > 0) & (mask_area == 0)).astype(int)

        # 2. CALCUL DES STATS BRUTES
        heights = self._get_heights(settled_blocks)
        holes = self._get_holes(settled_blocks, heights)
        bumpiness = self._get_bumpiness(heights)
        max_h = np.max(heights)
        sum_h = np.sum(heights)

        # 3. NORMALISATION (Crée des petits tableaux 1D de taille 1)
        # Note : np.array([valeur]) crée un vecteur de forme (1,)
        norm_heights = heights / 20.0 
        norm_holes = np.array([holes]) / 10.0
        norm_bump = np.array([bumpiness]) / 50.0 
        norm_max_h = np.array([max_h]) / 20.0
        norm_sum_h = np.array([sum_h]) / 100.0
        
        # 4. FUSION DES STATS
        # ATTENTION : On ne met PAS de crochets [] autour des variables 'norm_...' 
        # car ce sont DÉJÀ des tableaux numpy grâce à l'étape 3.
        board_stats = np.concatenate([
            norm_heights,  # (10,)
            norm_holes,    # (1,)  <-- Pas de crochets ici !
            norm_bump,     # (1,)
            norm_max_h,    # (1,)
            norm_sum_h     # (1,)
        ])

        # 5. HOLDER & QUEUE
        holder_flat = obs['holder'].flatten().astype(np.float32)
        queue_flat = obs['queue'].flatten().astype(np.float32)

        # 6. FUSION FINALE
        final_obs = np.concatenate([
            board_stats,
            holder_flat,
            queue_flat
        ])
        
        return final_obs.astype(np.float32)

    def _get_heights(self, grid):
        heights = np.zeros(self.n_cols)
        rows, cols = grid.shape
        for c in range(cols):
            col_data = grid[:, c]
            if np.any(col_data):
                # argmax renvoie le premier index True en partant du haut (0)
                first_block_idx = np.argmax(col_data > 0)
                # La hauteur, c'est le nombre de lignes en partant du bas
                heights[c] = rows - first_block_idx
            else:
                heights[c] = 0
        return heights

    def _get_holes(self, grid, heights):
        holes = 0
        rows, cols = grid.shape
        for c in range(cols):
            h = int(heights[c])
            if h == 0: continue
            
            # On regarde tout ce qui est SOUS le sommet de la colonne
            # Si le sommet est à l'index 20 (hauteur 4 sur 24), on regarde de 21 à 24.
            start_row = rows - h + 1 
            
            # Chaque case vide (0) en dessous est un trou
            holes += np.sum(grid[start_row:, c] == 0)
        return holes

    def _get_bumpiness(self, heights):
        # Différence absolue entre colonne i et i+1
        # ex: hauteurs [2, 5, 3] -> |2-5| + |5-3| = 3 + 2 = 5
        return np.sum(np.abs(np.diff(heights)))