from typing import Dict

import numpy as np

from .algorithm.dynamic_programming.policy_evaluation import policy_evaluation
from .algorithm.dynamic_programming.policy_iteration import policy_iteration
from .algorithm.dynamic_programming.value_iteration import value_iteration
from .environment.grid_world.GridWorldMDP import GridWorldMDP
from .environment.line_world.LineWorldMDP import LineWorldMDP
from ..do_not_touch.mdp_env_wrapper import Env1
from ..do_not_touch.result_structures import ValueFunction, PolicyAndValueFunction

def policy_evaluation_on_line_world() -> ValueFunction:
    """
    Creates a Line World of 7 cells (leftmost and rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Policy Evaluation Algorithm in order to find the Value Function of a uniform random policy
    Returns the Value function (V(s)) of this policy
    """
    print("################### policy_evaluation_on_line_world ##############")
    env = LineWorldMDP(7)
    pi=np.zeros((len(env.states()),len(env.actions())))
    pi[:,:]=0.5
    return policy_evaluation(MDPenv=env,theta=0.001,gamma=0.99,pi=pi)

def policy_iteration_on_line_world() -> PolicyAndValueFunction:
    """
    Creates a Line World of 7 cells (leftmost and rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Policy Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Returns the Policy (Pi(s,a)) and its Value Function (V(s))
    """
    print("\n\n################### policy_iteration_on_line_world ##############")
    env = LineWorldMDP(7)
    return policy_iteration(env,theta=0.001,gamma=0.99)


def value_iteration_on_line_world() -> PolicyAndValueFunction:
    """
    Creates a Line World of 7 cells (leftmost and rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Value Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Returns the Policy (Pi(s,a)) and its Value Function (V(s))
    """
    print("\n\n################### value_iteration_on_line_world ##############")
    env = LineWorldMDP(7)
    return value_iteration(env,theta=0.001,gamma=0.99)


def policy_evaluation_on_grid_world() -> ValueFunction:
    """
    Creates a Grid World of 5x5 cells (upper rightmost and lower rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Policy Evaluation Algorithm in order to find the Value Function of a uniform random policy
    Returns the Value function (V(s)) of this policy
    """
    print("\n\n################### policy_evaluation_on_grid_world ##############")
    env = GridWorldMDP(5,5)
    pi=np.zeros((len(env.states()),len(env.actions())))
    pi[:,:]=0.25
    return policy_evaluation(MDPenv=env,theta=0.001,gamma=0.99999 ,pi=pi)


def policy_iteration_on_grid_world() -> PolicyAndValueFunction:
    """
    Creates a Grid World of 5x5 cells (upper rightmost and lower rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Policy Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Returns the Policy (Pi(s,a)) and its Value Function (V(s))
    """
    print("\n\n################### value_iteration_on_line_world ##############")
    env = GridWorldMDP(5,5)
    return policy_iteration(MDPenv=env,theta=0.001,gamma=0.99999)


def value_iteration_on_grid_world() -> PolicyAndValueFunction:
    """
    Creates a Grid World of 5x5 cells (upper rightmost and lower rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Value Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Returns the Policy (Pi(s,a)) and its Value Function (V(s))
    """
    print("\n\n################### value_iteration_on_line_world ##############")
    env = GridWorldMDP(5,5)
    return value_iteration(MDPenv=env,theta=0.001,gamma=0.99999)


def policy_evaluation_on_secret_env1() -> ValueFunction:
    """
    Creates a Secret Env1
    Launches a Policy Evaluation Algorithm in order to find the Value Function of a uniform random policy
    Returns the Value function (V(s)) of this policy
    """
    print("\n\n################### policy_evaluation_on_secret_env1 ##############")
    env = Env1()
    pi=np.zeros((len(env.states()),len(env.actions())))
    for s in env.states():
        n = 0
        for _ in env.actions():
            n += 1
        for a in env.actions():
            pi[s, a] = 1 / n
    return policy_evaluation(MDPenv=env,theta=0.001,gamma=0.99999,pi=pi)


def policy_iteration_on_secret_env1() -> PolicyAndValueFunction:
    """
    Creates a Secret Env1
    Launches a Policy Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Returns the Policy (Pi(s,a)) and its Value Function (V(s))
    """
    print("\n\n################### policy_iteration_on_secret_env1 ##############")
    env = Env1()
    return policy_iteration(MDPenv=env,theta=0.001,gamma=0.99)


def value_iteration_on_secret_env1() -> PolicyAndValueFunction:
    """
    Creates a Secret Env1
    Launches a Value Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Prints the Policy (Pi(s,a)) and its Value Function (V(s))
    """
    print("\n\n################### value_iteration_on_secret_env1 ##############")
    env = Env1()
    return value_iteration(MDPenv=env,theta=0.001,gamma=0.99)

def demo():
    print(policy_evaluation_on_line_world())
    print(policy_iteration_on_line_world())
    print(value_iteration_on_line_world())

    print(policy_evaluation_on_grid_world())
    print(policy_iteration_on_grid_world())
    print(value_iteration_on_grid_world())
    #
    print(policy_evaluation_on_secret_env1())
    print(policy_iteration_on_secret_env1())
    print(value_iteration_on_secret_env1())
