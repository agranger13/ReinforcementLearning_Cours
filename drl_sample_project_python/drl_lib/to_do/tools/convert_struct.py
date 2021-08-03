from drl_sample_project_python.drl_lib.do_not_touch.result_structures import *


def convert_PolicyAndValueFunction(MDPenv, pi, V) -> PolicyAndValueFunction:
    return PolicyAndValueFunction(convert_pi(MDPenv, pi),convert_V(MDPenv, V))


def convert_V(MDPenv, V) -> ValueFunction:
    dict_V = {}
    for i, s in enumerate(MDPenv.states()):
        dict_V[int(s)] = float(V[i])
    return dict_V


def convert_pi(MDPenv, pi) -> Policy:
    dict_return_pi = {}
    for i, s in zip(pi, MDPenv.states()):
        dict_return_pi[int(s)] = {}
        dict_pi = {}
        for j, k in enumerate(i):
            dict_pi[int(j)] = float(k)
        dict_return_pi[s] = dict_pi
    return dict_return_pi
