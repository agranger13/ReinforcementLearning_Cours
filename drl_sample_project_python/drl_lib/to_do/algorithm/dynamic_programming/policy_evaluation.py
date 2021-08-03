import numpy as np

from drl_sample_project_python.drl_lib.to_do.tools.convert_struct import convert_V


def policy_evaluation(MDPenv, theta, gamma, pi):
    V = np.zeros((len(MDPenv.states()),))
    while True:
        delta = 0
        for s in MDPenv.states():
            v = V[s]
            V[s] = 0

            for a in MDPenv.actions():
                for s_p in MDPenv.states():
                    for r_idx, r in enumerate(MDPenv.rewards()):
                        V[s] += pi[s, a] * MDPenv.transition_probability(s, a, s_p, r_idx) * (r + gamma * V[s_p])
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break
    return convert_V(MDPenv, V)
