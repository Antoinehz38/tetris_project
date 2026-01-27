import copy
import numpy as np


from src.antoine.states_helper import make_info_from_obs

class LinearAfterStateAgent:
    """
    V(after_state) = w^T x
    Update TD(0): w <- w + alpha * (target - w^T x) * x
    target = r + gamma * V(next_after_state_greedy)   (ou r si terminal)
    """

    def __init__(self, n_features, alpha=1e-3, gamma=0.99, epsilon=0.1, seed=0):
        self.w = np.zeros(n_features, dtype=np.float32)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.rng = np.random.default_rng(seed)


    def features(self, board) -> np.ndarray:
        return make_info_from_obs(board)
    
    def cost(self, x: np.ndarray) -> float:
    # x = [bias, holes, agg_h, max_h, bump, rtrans, ctrans, wells, hdepth] (déjà normalisé)
        return float(
            2.0 * x[1] +                      # holes
            1.0 * x[2] +                      # aggregate height
            1.0 * x[4] +                      # bumpiness
            0.5 * (x[5] + x[6]) +             # transitions
            1.0 * x[8]                        # holes depth
            # wells ignoré au début (x[7])
        )

    def enumerate_candidate_sequences(self, env):
        """
        Doit renvoyer une liste de séquences d'actions.
        Exemple: [[ROT, ROT, LEFT, DROP], [RIGHT, DROP], ...]
        """
        sequences_of_actions_to_try = []
        for k in range(10): 
            sequences_of_actions_to_try.append( [0 for i in range(k-1)] + [5] )
            sequences_of_actions_to_try.append( [1 for i in range(k-1)] + [5] )
            sequences_of_actions_to_try.append( [3] + [0 for i in range(k-1)] + [5] )
            sequences_of_actions_to_try.append( [3] + [1 for i in range(k-1)] + [5] )
            sequences_of_actions_to_try.append( [3,3] + [0 for i in range(k-1)] + [5] )
            sequences_of_actions_to_try.append( [3,3] + [1 for i in range(k-1)] + [5] )
            sequences_of_actions_to_try.append( [3,3,3] + [0 for i in range(k-1)] + [5] )
            sequences_of_actions_to_try.append( [3,3,3] + [1 for i in range(k+1)] + [5] )
        return sequences_of_actions_to_try

    # --- UTILS ---
    def value(self, x: np.ndarray) -> float:
        return float(np.dot(self.w, x))

    def simulate_sequence(self, env, seq):
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

        for seq in self.enumerate_candidate_sequences(env):
            x_after, r_sum, done = self.simulate_sequence(env, seq)
            #v = self.value(x_after)
            #score = r_sum if done else (r_sum + self.gamma * v)
            score = -self.cost(x_after)

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
        seqs = self.enumerate_candidate_sequences(env)
        if len(seqs) == 0:
            raise RuntimeError("Aucune séquence candidate.")

        if self.rng.random() < self.epsilon:
            seq = seqs[self.rng.integers(len(seqs))]
            x_after, r_sum, done = self.simulate_sequence(env, seq)
            return seq, x_after, r_sum, done

        seq, x_after, _ = self.greedy_best_afterstate(env)
        # on resimule pour obtenir r_sum/done cohérents (ou adapte pour les renvoyer direct)
        x_after, r_sum, done = self.simulate_sequence(env, seq)
        return seq, x_after, r_sum, done

    def td_update(self, x_after, r_sum, done, env_after_step):
        """
        Mise à jour des poids :

        prédiction: v_pred = w^T x_after
        cible:
          - si terminal: target = r_sum
          - sinon: target = r_sum + gamma * max_seq' V(x_after_next_greedy)

        erreur TD: delta = target - v_pred
        update: w <- w + alpha * delta * x_after
        """
        v_pred = self.value(x_after)

        if done:
            target = float(r_sum)
        else:
            # cible bootstrap: valeur du meilleur after-state à partir du nouvel état env
            _, x_next, _ = self.greedy_best_afterstate(env_after_step)
            target = float(r_sum) + self.gamma * self.value(x_next)

        delta = target - v_pred
        self.w += self.alpha * delta * x_after  # <- UPDATE DES POIDS
        return delta, target, v_pred
