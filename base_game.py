import numpy as np
import random
from player import ConnectPlayer
# from connect_nn import NNAction, AlphaZeroNN
from collections import Counter
from human_play import HumanPlay


class ConnectFourGame:
    def __init__(self, nn_a=None, nn_b=None):
        self.board = np.zeros((6, 7))
        self.player_list = [ConnectPlayer('a', 1, nn_a), ConnectPlayer('b', 2, nn_b)]
        random.shuffle(self.player_list)
        # self.player_list[0].counter = 1
        # self.player_list[1].counter = 2
        self.player_turn = [1, 0]
        self.grid_increase = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
        self.grid_increase = [np.array(i) for i in self.grid_increase]

    def action_board(self, action, counter):
        for i in range(6):
            if self.board[5 - i, action] == 0:
                self.board[5 - i, action] = counter
                break

    def grid_check(self, start_pos, counter):
        consecutive_counter = 1
        if self.board[start_pos] != counter:
            return False
        for grid in self.grid_increase:
            for i in range(1, 4):
                current_pos = tuple(np.array(start_pos) + i * grid)
                if current_pos[0] < 0 or current_pos[1] < 0:
                    consecutive_counter = 1
                    break
                try:
                    if self.board[current_pos] == counter:
                        consecutive_counter += 1
                    else:
                        consecutive_counter = 1
                        break
                    if consecutive_counter == 4:
                        return True
                except Exception as e:
                    # print(e)
                    consecutive_counter = 1
                    break
        return False

    def identify_win(self, counter):
        for row in range(6):
            for column in range(7):
                if self.grid_check((5 - row, 6 - column), counter):
                    return True
        return False

    def run_game(self):
        four_row = False
        player_turn = 0
        player = None
        while not four_row:
            player = self.player_list[player_turn]
            action = player.action(self.board)
            self.action_board(action, player.counter)
            four_row = self.identify_win(player.counter)
            player_turn = self.player_turn[player_turn]
            if np.count_nonzero(self.board) == 42:
                break
                # print(self.board)
        if np.count_nonzero(self.board) == 42:
            return 'draw'
        else:
            return player

    def check_end(self, state, counter):
        self.board = state
        opponent_counter = 1 if counter == 2 else 2
        if self.identify_win(counter):
            return 1
        if self.identify_win(opponent_counter):
            return -1
        else:
            return 0




class HumanGame:
    def __init__(self, nn_class, human_play):
        self.nn_a = nn_class(load_filepath='q_model.h5', epsilon=0)
        self.b_human = human_play()
        self.game = None

    def run_game(self):
        self.game = ConnectFourGame(self.nn_a, self.b_human)
        self.game.run_game()



# if __name__ == '__main__':
#     # rl = MultipleGames(NNAction)
#     # rl.simulate_many_games(100000)
#
#     rl = AlphaZero()
#     rl.iterate_models()

    # rl = HumanGame(NNAction, HumanPlay)
    # rl.run_game()
