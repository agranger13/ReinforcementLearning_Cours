import numpy as np
class LineWorldMDP:
    def __init__(self, size: int):
        self.size=size
        self.p=self.init_env()

    def actions(self):
        return np.array([0, 1]) # 0: Gauche, 1: Droite

    def states(self):
        return np.arange(self.size)

    def rewards(self):
        return np.array([-1, 0, 1])

    def is_state_terminal(self,s):
        return self.size-1 == s or 0 == s

    def init_env(self):
        S = self.states()
        A = self.actions()
        R = self.rewards()
        p = np.zeros((len(S), len(A), len(S), len(R)))

        for i in range(1, self.size - 2):
            p[i, 1, i + 1, 1] = 1.0

        for i in range(2, self.size - 1):
            p[i, 0, i - 1, 1] = 1.0

        #win
        p[self.size - 2, 1, self.size - 1, 2] = 1.0
        #loose
        p[1, 0, 0, 0] = 1.0
        return p

    def transition_probability(self, s: int, a: int, s_p: int, r: float) -> float:
        return self.p[s, a, s_p, r]
