import argparse
import os
import torch
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation, RecordEpisodeStatistics
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from tetris_gymnasium.envs.tetris import Tetris

# --- Fonctions utilitaires pour le calcul des récompenses ---
# (Reprises de src/tetris_code/policies.py pour être autonomes dans l'entraînement)

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

def holes(board):
    """Calcule le nombre de trous sur le plateau."""
    nb_holes = 0
    # Parcourir les lignes et les colonnes du plateau
    for i in range(board.shape[0]):
        for j in range(board.shape[1]):
            # Si un pixel est vide (0) mais qu'il y a un bloc au-dessus (!= 0), c'est un trou
            if (i > 1) and (board[i, j] == 0) and (board[i-1, j] != 0):
                nb_holes = nb_holes + 1
    return nb_holes

# --- Wrapper pour modifier la récompense ---

class TetrisRewardWrapper(gym.Wrapper):
    def __init__(self, env, reward_mode):
        super().__init__(env)
        self.reward_mode = reward_mode

    def step(self, action):
        # Exécuter l'action dans l'environnement original
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Récupérer l'état du plateau
        board = obs["board"]
        
        # Modifier la récompense selon le mode choisi
        if self.reward_mode == "holes":
            # Objectif : Minimiser les trous.
            # On donne une pénalité proportionnelle au nombre de trous.
            # L'agent cherchera à faire tendre cette valeur vers 0 (donc maximiser -N).
            current_holes = holes(board)
            reward = -float(current_holes)
            
        elif self.reward_mode == "height":
            # Objectif : Minimiser la hauteur de la pile.
            # La fonction 'heights' retourne l'index de la première brique rencontrée en partant du haut (la profondeur).
            # Plus cet index est GRAND, plus la pile est BASSE.
            # On veut donc maximiser min(heights).
            h_list = heights(board)
            if h_list:
                # On utilise min() car c'est la colonne la plus haute (le plus petit index) qui définit la hauteur max du jeu.
                min_depth = min(h_list)
                reward = float(min_depth)
            else:
                reward = 0.0
        
        # Si le mode est "standard", on laisse la récompense telle quelle (score du jeu)
        
        return obs, reward, terminated, truncated, info

# --- Configuration de l'environnement et du main ---

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_env(reward_mode="standard"):
    env = gym.make("tetris_gymnasium/Tetris")
    
    # Appliquer le wrapper de récompense si un mode spécifique est demandé
    if reward_mode in ["holes", "height"]:
        env = TetrisRewardWrapper(env, reward_mode)
        
    env = RecordEpisodeStatistics(env)
    env = FlattenObservation(env)
    env = Monitor(env)
    return env

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", type=str, default="ppo", choices=["dqn", "double_dqn", "ppo"])
    p.add_argument("--steps", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--logdir", type=str, default="runs/tetris")
    p.add_argument("--save_every", type=int, default=50_000)
    
    # Nouvel argument pour choisir le type de récompense
    p.add_argument("--reward", type=str, default="standard", choices=["standard", "holes", "height"], 
                   help="Choose the reward function: 'standard' (game score), 'holes' (minimize holes), or 'height' (minimize stack height).")

    p.add_argument("--buffer_size", type=int, default=100_000)
    p.add_argument("--learning_starts", type=int, default=10_000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--train_freq", type=int, default=4)
    p.add_argument("--target_update_interval", type=int, default=1_000)
    p.add_argument("--exploration_fraction", type=float, default=0.4)
    p.add_argument("--exploration_final_eps", type=float, default=0.05)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=1e-3)

    args = p.parse_args()

    os.makedirs(args.logdir, exist_ok=True)
    
    # On passe l'argument reward à la création de l'environnement
    env = make_env(reward_mode=args.reward)

    checkpoint_cb = CheckpointCallback(
        save_freq=args.save_every,
        save_path=args.logdir,
        name_prefix=f"{args.algo}_tetris_{args.reward}", # Ajout du mode dans le nom du fichier
        save_replay_buffer=True,
        save_vecnormalize=True,
    )

    if args.algo in ["dqn", "double_dqn"]:
        model = DQN(
            "MlpPolicy",
            env,
            device=device,
            gamma=args.gamma,
            learning_rate=args.lr,
            buffer_size=args.buffer_size,
            learning_starts=args.learning_starts,
            batch_size=args.batch_size,
            train_freq=args.train_freq,
            target_update_interval=args.target_update_interval,
            exploration_fraction=args.exploration_fraction,
            exploration_final_eps=args.exploration_final_eps,
            tensorboard_log=args.logdir,
            verbose=1,
            seed=args.seed,
        )

    else:  # ppo
        model = PPO(
            "MlpPolicy",
            env,
            device=device,
            verbose=1,
            seed=args.seed,
            tensorboard_log=args.logdir,
            n_steps=2048,
            batch_size=64,
            learning_rate=3e-4,
            gamma=0.99,
        )

    model.learn(total_timesteps=args.steps, callback=checkpoint_cb)
    
    # Sauvegarde finale avec le nom du reward
    model.save(os.path.join(args.logdir, f"{args.algo}_tetris_{args.reward}_final"))
    env.close()

if __name__ == "__main__":
    main()