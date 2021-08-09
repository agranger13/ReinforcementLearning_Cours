import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from drl_sample_project_python.drl_lib.do_not_touch.contracts import SingleAgentEnv
from drl_sample_project_python.drl_lib.do_not_touch.result_structures import PolicyAndActionValueFunction


def monte_carlo_on_policy_first_visit(env :SingleAgentEnv,gamma,eps,max_iter=1000):
    Q = {}
    pi = {}
    Returns = {}

    win = 0
    lose = 0
    draw = 0

    wins = []
    loses = []
    draws = []

    average_rewards = []
    rewards_history = np.zeros(100)
    for _ in tqdm(range(max_iter)):
        # rewards_history[_ % 100] = env.score()
        #
        # if env.is_win():
        #     win += 1
        # elif env.is_draw():
        #     draw += 1
        # elif env.is_loss():
        #     lose += 1
        #
        #
        # if _ % 100 == 0:
        #     average_rewards.append(np.mean(rewards_history))
        #     print("\nWin:", win, " | Lose :", lose, " | Draw:", draw, " | Eps:", eps)
        #     wins.append(win)
        #     loses.append(lose)
        #     draws.append(draw)
        #     win = 0
        #     draw = 0
        #     lose = 0

        eps *= 0.9999
        env.reset()
        S = []
        A = []
        R = []
        while not env.is_game_over():
            s = str(env.state_id())
            S.append(s)
            available_actions = env.available_actions_ids()
            if s not in pi:
                pi[s] = {}
                Q[s] = {}
                Returns[s] = {}
                for a in available_actions:

                    pi[s][a]=1/len(available_actions)
                    Q[s][a] = 0.0
                    Returns[s][a] = []
            # print(len(available_actions))
            # print(list(pi[s].values()))
            action = np.random.choice(a=available_actions, p=list(pi[s].values()))
            A.append(action)

            old_r = env.score()
            env.act_with_action_id(action)
            r = env.score()
            R.append(r - old_r )

        G = 0
        for t in reversed(range(len(S))):
            G = gamma * G + R[t]

            if S[t] in S[:t] and A[t] in A[:t]:
                continue

            Returns[S[t]][A[t]].append(G)
            Q[S[t]][A[t]] = np.mean(Returns[S[t]][A[t]])
            max_a = None
            best_a = None
            for a in Q[S[t]]:
                if max_a is None or Q[S[t]][a] > max_a:
                    max_a = Q[S[t]][a]
                    best_a = a

            for a in Q[S[t]]:
                if a == best_a :
                    pi[S[t]][a] = 1 - eps + eps/len(Q[S[t]])
                else :
                    pi[S[t]][a] = eps/len(Q[S[t]])


    # plt.plot(average_rewards)
    # plt.show()
    # plt.plot(wins, label='Win')
    # plt.plot(loses, label='Lose')
    # plt.plot(draws, label='Draw')
    # plt.legend(['Win', 'Lose', 'Draw'])
    # plt.title("MC On Policy first-visit")
    # plt.show()

    return PolicyAndActionValueFunction(pi,Q)