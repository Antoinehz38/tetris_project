import gymnasium as gym
import numpy as np

class TetrisRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.prev_holes = 0
        self.prev_bumpiness = 0
        self.prev_sum_height = 0
        self.prev_max_height = 0 # <--- AJOUT

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_holes = 0
        self.prev_bumpiness = 0
        self.prev_sum_height = 0
        self.prev_max_height = 0 # <--- AJOUT
        return obs, info

    def step(self, action):
        # 1. Env step
        obs, reward, terminated, truncated, info = self.env.step(action)

        # 2. Dénormalisation (Récupération des vraies valeurs)
        current_holes = obs[10] * 10.0
        current_bumpiness = obs[11] * 50.0
        current_sum_height = obs[13] * 100.0
        current_max_height = obs[12] * 20.0 # On récupère aussi la hauteur max

        # 3. CALCUL DU SHAPING
        shaping_reward = 0

        # --- A. PUNITION DES TROUS (Priorité 1) ---
        # On reste sévère : un trou est une dette technique.
        holes_diff = current_holes - self.prev_holes
        if holes_diff > 0:
            shaping_reward -= (holes_diff * 0.5) # -0.5 par trou créé

        # --- B. PUNITION DE LA HAUTEUR (Ta demande) ---
        
        # 1. Delta Hauteur Cumulée (Est-ce qu'on a empilé globalement ?)
        # On punit chaque bloc ajouté qui augmente la masse totale
        height_diff = current_sum_height - self.prev_sum_height
        if height_diff > 0:
            shaping_reward -= (height_diff * 0.03) 
        
        # 2. Delta Hauteur MAX (Est-ce qu'on s'approche du plafond ?)
        # C'est LA punition anti-tours. Si le point le plus haut monte, on tape fort.
        # On punit davantage si on est déjà haut (exponentiel)
        max_h_diff = current_max_height - self.prev_max_height
        if max_h_diff > 0:
            # Si on est bas (hauteur 2), punition faible.
            # Si on est haut (hauteur 15), punition énorme.
            danger_factor = (current_max_height / 20.0) # 0.1 à 1.0
            shaping_reward -= (max_h_diff * 1.0 * danger_factor)

        # --- C. RUGOSITÉ (Gardons le plat) ---
        bump_diff = current_bumpiness - self.prev_bumpiness
        if bump_diff > 0:
            shaping_reward -= (bump_diff * 0.02)

        # --- D. LE JACKPOT (LIGNES) ---
        # Il faut compenser toutes ces punitions de hauteur par un gros bonus
        # sinon l'IA va arrêter de jouer pour ne pas monter.
        if reward > 0:
            # reward est souvent 1, 4, 10...
            # On donne un bonus énorme pour dire "Ça valait le coup de monter un peu"
            shaping_reward += (reward * 80.0) 
            
        # --- E. SURVIE / MORT ---
        if terminated:
            shaping_reward -= 100.0 # La mort reste la pire chose
        else:
            shaping_reward += 0.01 # Petit biscuit pour être en vie

        # 4. Updates
        self.prev_holes = current_holes
        self.prev_bumpiness = current_bumpiness
        self.prev_sum_height = current_sum_height
        self.prev_max_height = current_max_height # N'oublie pas d'init ça dans __init__ et reset !
        
        return obs, shaping_reward, terminated, truncated, info