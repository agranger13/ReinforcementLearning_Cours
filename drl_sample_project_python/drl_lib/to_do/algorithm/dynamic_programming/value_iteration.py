import numpy as np

from drl_sample_project_python.drl_lib.to_do.tools.convert_struct import convert_V, convert_PolicyAndValueFunction


def value_iteration(MDPenv,theta,gamma):
    V = np.zeros((len(MDPenv.states()),))

    pi = np.zeros((len(MDPenv.states()), len(MDPenv.actions())))
    # for s in MDPenv.states():
    #     pi[s, np.random.choice(MDPenv.actions())] = 1.0
    while True :
        delta = 0
        for s in MDPenv.states():
            v = V[s]
            V[s] = 0

            best_a = -1
            best_a_score = None
            for a in MDPenv.actions():
                a_score = 0.0

                for s_p in MDPenv.states():
                    for r_idx,r in enumerate(MDPenv.rewards()):
                        a_score += MDPenv.transition_probability(s,a,s_p,r_idx) * (r + gamma * V[s_p])

                if best_a_score is None or best_a_score < a_score :
                    best_a = a
                    best_a_score = a_score

            V[s] = best_a_score
            delta = max(delta, abs(v-V[s]))
            pi[s, :] = 0.0
            pi[s, best_a] = 1.0
        if delta < theta:
            break

    return convert_PolicyAndValueFunction(MDPenv, pi, V)
