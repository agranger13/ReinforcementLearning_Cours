import numpy as np


class GridWorldMDP:
    def __init__(self, length: int, height: int):
        assert (length >= 3)
        assert (height >= 3)
        self.height = height
        self.length = length
        self.max_cell = length * height

        self.cell_fail = length - 1
        self.cell_goal = self.max_cell - 1

        self.p = self.init_env()

    def actions(self):
        return np.array([0, 1, 2, 3])  # 0: Bas, 1: Gauche, 2: Droite, 3: Haut

    def states(self):
        return np.arange(self.max_cell)

    def rewards(self):
        return np.array([-1, 0, 1])

    def is_state_terminal(self,s):
        return s == self.cell_fail or s == self.cell_goal

    def init_env(self):
        S = self.states()
        A = self.actions()
        R = self.rewards()
        p = np.zeros((len(S), len(A), len(S), len(R)))

        # Definie les bordures de la grid world
        for i in range(0, self.max_cell):
            if i != self.cell_goal and i != self.cell_fail:
                # A gauche
                if i % self.length != 0:
                    if i - 1 == self.cell_goal:
                        r = 2
                    elif i - 1 == self.cell_fail:
                        r = 0
                    else:
                        r = 1
                    p[i, 1, i - 1, r] = 1
                else :
                    p[i, 1, i, 1] = 1

                # A droite
                if i % self.length != self.length - 1:
                    if i + 1 == self.cell_goal:
                        r = 2
                    elif i + 1 == self.cell_fail:
                        r = 0
                    else:
                        r = 1
                    p[i, 2, i + 1, r] = 1
                else :
                    p[i, 2, i, 1] = 1

                # En haut
                if i >= self.length:
                    if i - self.length == self.cell_goal:
                        r = 2
                    elif i - self.length == self.cell_fail:
                        r = 0
                    else:
                        r = 1
                    p[i, 3, i - self.length, r] = 1
                else :
                    p[i, 3, i, 1] = 1

                # En bas
                if i < self.max_cell - self.length:
                    if i + self.length == self.cell_goal:
                        r = 2
                    elif i + self.length == self.cell_fail:
                        r = 0
                    else:
                        r = 1
                    p[i, 0, i + self.length, r] = 1
                else :
                    p[i, 0, i, 1] = 1
        return p

    def transition_probability(self, s: int, a: int, s_p: int, r: float) -> float:
        return self.p[s, a, s_p, r]
