import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from drl_sample_project_python.drl_lib.do_not_touch.contracts import SingleAgentEnv
from drl_sample_project_python.drl_lib.do_not_touch.result_structures import PolicyAndActionValueFunction


def monte_carlo_off_policy(env :SingleAgentEnv,gamma,max_iter=1000):
    Q = {}
    pi = {}
    C = {}
    b = {}

    win = 0
    lose = 0
    draw = 0

    wins = []
    loses = []
    draws = []

    for _ in tqdm(range(max_iter)):
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
                C[s] = {}
                b[s] = {}
                for a in available_actions:

                    pi[s][a]=1/len(available_actions)
                    Q[s][a] = 0.0
                    C[s][a] = 0.0
                    b[s][a] = 1/len(available_actions)
            # print(len(available_actions))
            # print(list(pi[s].values()))
            action = np.random.choice(a=available_actions, p=list(b[s].values()))
            A.append(action)

            old_r = env.score()
            env.act_with_action_id(action)
            r = env.score()
            R.append(r - old_r )

        G = 0.0
        W = 1.0
        for t in reversed(range(len(S))):
            G = gamma * G + R[t]
            C[S[t]][A[t]] = C[S[t]][A[t]] + W
            Q[S[t]][A[t]] += (W/C[S[t]][A[t]]) * (G - Q[S[t]][A[t]])

            max_a = None
            best_a = None
            for a in Q[S[t]]:
                if max_a is None or Q[S[t]][a] > max_a:
                    max_a = Q[S[t]][a]
                    best_a = a
            for a in pi[S[t]]:
                pi[S[t]][a] = 0
            pi[S[t]][best_a] = 1

            if A[t] != best_a :
                break
            W = W * 1/b[S[t]][A[t]]

    # for i in range(5000):
    #     env.reset()
    #     while not env.is_game_over():
    #         s = str(env.state_id())
    #         available_actions = env.available_actions_ids()
    #         action = np.random.choice(a=available_actions, p=list(pi[s].values()))
    #         env.act_with_action_id(action)
    #
    #         if env.is_game_over():
    #             if env.is_win():
    #                 win += 1
    #             elif env.is_draw():
    #                 draw += 1
    #             elif env.is_loss():
    #                 lose += 1
    #
    #             if i % 100 == 0:
    #                 print("\nWin:", win, " | Lose :", lose, " | Draw:", draw)
    #                 wins.append(win)
    #                 loses.append(lose)
    #                 draws.append(draw)
    #                 win = 0
    #                 draw = 0
    #                 lose = 0
    #
    # plt.plot(wins, label='Win')
    # plt.plot(loses, label='Lose')
    # plt.plot(draws, label='Draw')
    # plt.legend(['Win', 'Lose', 'Draw'])
    # plt.title("MC Off-policy")
    # plt.show()

    return PolicyAndActionValueFunction(pi,Q)