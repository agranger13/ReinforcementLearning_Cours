import numpy as np
from drl_sample_project_python.drl_lib.do_not_touch.contracts import SingleAgentEnv


class TicTacToe(SingleAgentEnv):
    def __init__(self, reward_win, reward_draw, reward_lose):
        self.board = np.full(9,3)
        self.cell_arr = np.arange(9)

        self.user_num = 0
        self.comp_num = 1
        self.nex_player = 0

        self.current_score = 0.0

        self.game_over = False

        self.reward_win = reward_win
        self.reward_draw = reward_draw
        self.reward_lose = reward_lose

        self.winner=3

    def is_game_over(self):
        return self.game_over

    def is_win(self) -> bool:
        return self.score() == 10

    def is_loss(self) -> bool:
        return self.score() < 0

    def is_draw(self) -> bool:
        return self.score() == 1

    def state_id(self) -> int:
        return self.board

    def score(self) -> float:
        return self.current_score

    def reset(self):
        self.board = np.full(9,3)
        self.nex_player = np.random.randint(0, 2)
        # self.nex_player = 0
        if self.nex_player == self.comp_num:
            open_slots = self.available_actions_ids()
            comp_input = np.random.choice(open_slots)
            self.place_letter(comp_input, self.comp_num)
        self.game_over = False
        self.current_score = 0.0
        self.winner = 3

    def available_actions_ids(self) -> np.ndarray:
        return np.where(self.board == 3)[0]

    def act_with_action_id(self,action_id):
        self.place_letter(action_id, self.user_num)
        if self.check_if_win(self.user_num) :
            return

        open_slots = self.available_actions_ids()
        comp_input = np.random.choice(open_slots)
        self.place_letter(comp_input, self.comp_num)

        if self.check_if_win(self.comp_num) :
            return

    def check_if_win(self, last_player):
        for i in range(0, 3):
            # Checks rows and columns for match
            rows_win = (self.board.reshape(3,3)[i ,:] == last_player).all()
            cols_win = (self.board.reshape(3,3)[:, i] == last_player).all()
            if rows_win or cols_win:
                return self.finish_game(last_player)

        diag1_win = (np.diag(self.board.reshape(3,3)) == last_player).all()
        diag2_win = (np.diag(np.fliplr(self.board.reshape(3,3))) == last_player).all()

        if diag1_win or diag2_win:
            return self.finish_game(last_player)

        if len(self.available_actions_ids()) == 0:
            return self.finish_game(3)

    def finish_game(self, last_player):
        if last_player == self.user_num:
            self.current_score += self.reward_win
            self.winner=0
            # print("finish win")
        elif last_player == self.comp_num:
            self.current_score += self.reward_lose
            # print("finish lose")
            self.winner=1
        else :
            self.current_score += self.reward_draw
            self.winner = 2
            # print("finish draw")
        # print(self.board.reshape(3,3))

        self.game_over = True
        return 2

    def place_letter(self, current_input, current_num):
        self.board[current_input] = current_num

