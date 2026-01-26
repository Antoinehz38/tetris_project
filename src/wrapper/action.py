import gymnasium as gym
import numpy as np

class TetrisActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        # Espace d'action Macro : 40 actions (10 colonnes * 4 rotations)
        self.action_space = gym.spaces.Discrete(40)
        
        # Mapping des actions micro (vérifie ces IDs)
        self.MOVE_LEFT = 0
        self.MOVE_RIGHT = 1
        self.ROTATE = 3 
        self.HARD_DROP = 5
        
        # OFFSET DU MUR GAUCHE
        # D'après tes logs précédents, le board fait 18 de large.
        # Il y a 4 colonnes de mur à gauche (indices 0, 1, 2, 3).
        # La zone jouable commence à l'index 4.
        self.WALL_OFFSET = 4 

    def action(self, action):
        raise NotImplementedError("Utilise step() directement")

    def step(self, action):
        # 1. DÉCODAGE
        desired_rot = action // 10  # 0, 1, 2, 3
        desired_col = action % 10   # 0 à 9
        
        total_reward = 0
        terminated, truncated = False, False
        info = {}
        
        # On a besoin d'une observation initiale pour savoir où on est
        # Mais step ne la donne pas. On va l'obtenir après la première rotation.
        # Si aucune rotation n'est demandée, on doit quand même faire un "No-Op" ou utiliser l'état interne
        # Astuce : On fait les rotations, et on récupère l'obs à chaque fois.
        
        current_obs = None

        # 2. ROTATIONS
        # Même si desired_rot = 0, on ne fait rien ici, on récupérera la position après.
        # Sauf qu'on a besoin de l'obs courante.
        # Dans Tetris-Gymnasium, on ne peut pas 'lire' l'obs sans faire step.
        # Donc si 0 rotation, on suppose qu'on est au spawn point standard, 
        # OU on fait une action neutre (ex: Down d'un cran, ou une action nulle si dispo).
        # Pour simplifier : Faisons les rotations demandées.
        
        for _ in range(desired_rot):
            obs, r, t, tr, i = self.env.step(self.ROTATE)
            current_obs = obs
            total_reward += r
            if t or tr: return obs, total_reward, t, tr, i

        # Si on n'a pas fait de rotation (desired_rot=0), on ne connait pas current_obs.
        # On est obligé de tricher un peu en faisant un mouvement "nul" ou en déduisant.
        # L'astuce simple : Faire un petit 'Down' (Soft Drop) qui ne change rien à la colonne
        # pour récupérer l'obs, ou juste faire le calcul sans obs si on est sûr du spawn.
        # Mieux : On fait un "Soft Drop" (2) d'un cran si rot=0 pour lire le jeu.
        if desired_rot == 0:
            obs, r, t, tr, i = self.env.step(2) # 2 = Move Down (Soft Drop)
            current_obs = obs
            total_reward += r
            if t or tr: return obs, total_reward, t, tr, i
            
        # 3. SCAN DE LA POSITION ACTUELLE
        # On regarde le masque pour trouver l'index X du bloc le plus à gauche
        mask = current_obs['active_tetromino_mask']
        
        # np.any(mask, axis=0) nous donne un tableau de booléens par colonne
        # np.argmax renvoie l'index du premier True
        # Attention : mask est (24, 18).
        
        cols_with_piece = np.where(np.any(mask, axis=0))[0]
        
        if len(cols_with_piece) == 0:
            # Cas rare : pièce invisible ou bug. On abandonne le mouvement latéral.
            current_absolute_x = self.WALL_OFFSET + 4 # Valeur par défaut (milieu)
        else:
            current_absolute_x = cols_with_piece[0] # Le pixel le plus à gauche de la pièce
            
        # Convertir en position relative (0 à 9)
        current_relative_x = current_absolute_x - self.WALL_OFFSET
        
        # 4. CALCUL DU DÉPLACEMENT
        diff = desired_col - current_relative_x
        
        # 5. DÉPLACEMENT DIRECT (Smart Move)
        if diff < 0:
            # On doit aller à gauche
            for _ in range(abs(diff)):
                obs, r, t, tr, i = self.env.step(self.MOVE_LEFT)
                total_reward += r
                if t or tr: return obs, total_reward, t, tr, i
                
        elif diff > 0:
            # On doit aller à droite
            for _ in range(abs(diff)):
                obs, r, t, tr, i = self.env.step(self.MOVE_RIGHT)
                total_reward += r
                if t or tr: return obs, total_reward, t, tr, i
                
        # 6. CHUTE FINALE
        obs, r, t, tr, i = self.env.step(self.HARD_DROP)
        total_reward += r
        
        return obs, total_reward, t, tr, i