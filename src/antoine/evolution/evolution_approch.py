import numpy as np 

from src.antoine.states_helper import make_info_from_obs
from src.antoine.evolution.helper import FastTetrisSim

import copy


SEQUENCE_TO_TRY =[]

for k in range(10): 
    SEQUENCE_TO_TRY.append( [0 for i in range(k-1)] + [5] )
    SEQUENCE_TO_TRY.append( [1 for i in range(k-1)] + [5] )
    SEQUENCE_TO_TRY.append( [3] + [0 for i in range(k-1)] + [5] )
    SEQUENCE_TO_TRY.append( [3] + [1 for i in range(k-1)] + [5] )
    SEQUENCE_TO_TRY.append( [3,3] + [0 for i in range(k-1)] + [5] )
    SEQUENCE_TO_TRY.append( [3,3] + [1 for i in range(k-1)] + [5] )
    SEQUENCE_TO_TRY.append( [3,3,3] + [0 for i in range(k-1)] + [5] )
    SEQUENCE_TO_TRY.append( [3,3,3] + [1 for i in range(k+1)] + [5] )



class EvolutionAgent:
    def __init__(self, n_features, W0=None):
        self.n_features = n_features
        if W0 is not None:
            self.w = W0
        else:
            self.w = np.zeros(n_features, dtype=np.float32)

        self.seqs = SEQUENCE_TO_TRY  # séquences candidates (remplies à chaque appel de choose_sequence)
        
    def features(self, board) -> np.ndarray:
        return make_info_from_obs(board)


    def value(self, x: np.ndarray) -> float:
        return float(np.dot(self.w, x))

  
    def simulate_sequence_copy(self, env, seq):
        """
        Simule seq sur une COPIE d'env et renvoie:
          x_after : features de l'état final après la seq
          r_sum   : reward cumulée pendant la seq
          done    : terminal/truncated atteint pendant la seq
        """
        sim = copy.deepcopy(env)
        r_sum = 0.0
        done = False

        obs2 = None
        for a in seq:
            obs2, r, terminated, truncated, info = sim.step(a)
            r_sum += float(r)
            done = bool(terminated or truncated)
            if done:
                break

        board2 = obs2["board"]
        x_after = self.features(board2).astype(np.float32)
        return x_after, r_sum, done

   
    
    def greedy_best_afterstate(self, env):
        """
        Calcule le meilleur after-state (greedy) depuis l'état courant.
        Renvoie: best_seq, best_x, best_score
        score = r_sum + gamma * V(x_after)
        """
        best_score = -1e30
        best_seq, best_x = None, None

        for seq in self.seqs:
            x_after, r_sum, done = self.simulate_sequence_copy(env, seq)
            score = self.value(x_after)

            if score > best_score:
                best_score = score
                best_seq = seq
                best_x = x_after

        return best_seq, best_x, best_score


    
    def choose_sequence(self, env):
        """
        epsilon-greedy sur le score lookahead-1:
          avec proba epsilon: seq au hasard
          sinon: meilleure seq
        Renvoie: chosen_seq, x_after_chosen, r_sum_chosen, done_chosen
        """
        if len(self.seqs) == 0:
            raise RuntimeError("Aucune séquence candidate.")


        seq, x_after, _ = self.greedy_best_afterstate(env)
        
        return seq
