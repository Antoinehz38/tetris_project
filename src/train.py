import argparse
import os, torch

import gymnasium as gym
from gymnasium.wrappers import FlattenObservation, RecordEpisodeStatistics

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

from tetris_gymnasium.envs.tetris import Tetris  

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from wrapper.obs import TetrisObservationWrapper
from wrapper.action import TetrisActionWrapper
from wrapper.reward import TetrisRewardWrapper 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def make_env():
    env = gym.make("tetris_gymnasium/Tetris")
    env = RecordEpisodeStatistics(env)
    env = TetrisActionWrapper(env)
    env = TetrisObservationWrapper(env)
    env = TetrisRewardWrapper(env) 
    env = Monitor(env)
    return env


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", type=str, default="dqn", choices=["dqn", "double_dqn", "ppo"])
    p.add_argument("--steps", type=int, default=500_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--logdir", type=str, default="runs/tetris")
    p.add_argument("--save_every", type=int, default=50_000)

    p.add_argument("--buffer_size", type=int, default=500_000)
    p.add_argument("--learning_starts", type=int, default=10_000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--train_freq", type=int, default=4)
    p.add_argument("--target_update_interval", type=int, default=1_000)
    p.add_argument("--exploration_fraction", type=float, default=0.5)
    p.add_argument("--exploration_final_eps", type=float, default=0.05)
    p.add_argument("--gamma", type=float, default=0.999)
    p.add_argument("--lr", type=float, default=1e-4)


    args = p.parse_args()

    os.makedirs(args.logdir, exist_ok=True)
    env = make_vec_env(
        make_env,               # Ta fonction qui crée un env
        n_envs=16,              # Nombre de jeux en parallèle (essaie 8 ou 16 selon ton CPU)
        seed=args.seed,
        vec_env_cls=SubprocVecEnv # Active le VRAI multi-processing (plusieurs coeurs CPU)
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=args.save_every,
        save_path=args.logdir,
        name_prefix=f"{args.algo}_tetris",
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
        policy_kwargs = dict(
            net_arch=dict(
                pi=[128, 128],   # réseau de la policy
                vf=[128, 128]    # réseau de la value function
            )
        )

        model = PPO(
            "MlpPolicy",
            env,
            device=device,
            policy_kwargs=policy_kwargs,
            verbose=1,
            seed=args.seed,
            tensorboard_log=args.logdir,
            n_steps=2048,
            batch_size=1024,
            learning_rate=3e-4,
            ent_coef=0.01,
            gamma=0.999,
        )

    model.learn(total_timesteps=args.steps, callback=checkpoint_cb)
    model.save(os.path.join(args.logdir, f"{args.algo}_tetris_final"))
    env.close()


if __name__ == "__main__":
    main()


