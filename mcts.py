import numpy as np
from base_game import ConnectFourGame
import math


class MCTS:
    def __init__(self, nnet, counter, cput=1, num_sim=50, epsilon=0):
        self.p = {}
        self.n = {}
        self.q = {}
        self.nnet = nnet
        # self.game = None
        self.cput = cput
        self.num_sim = num_sim
        self.counter = counter
        self.opponent_counter = 1 if self.counter == 2 else 2
        self.original_counter = counter
        self.epsilon = epsilon

    def action_prob(self, state, counter, opponent_counter):
        self.counter = counter
        self.opponent_counter = opponent_counter
        string_state = np.array_str(self.transform_state(state))
        for i in range(self.num_sim):
            self.counter = counter
            self.opponent_counter = opponent_counter
            search_state = state.copy()
            self.search(search_state)
        return self.n[string_state] / self.n[string_state].sum()

    def action(self, state, counter, opponent_counter):
        action_prob = self.action_prob(state, counter, opponent_counter)
        if np.random.uniform(0, 1) > self.epsilon:
            action = np.random.randint(0, 6)
        else:
            action = np.random.choice(len(action_prob), p=action_prob)
        return action

    def transform_state(self, state):
        own_array = state.copy()
        opponent_array = state.copy()
        own_array[state == self.counter] = 1
        own_array[state != self.counter] = 0
        opponent_array[state == self.opponent_counter] = 1
        opponent_array[state != self.opponent_counter] = 0
        return np.stack([own_array, opponent_array]).reshape((2, 6, 7))

    def search(self, state):
        curr_game = ConnectFourGame()
        game_status = curr_game.check_end(state, self.original_counter)
        self.opponent_counter = 2 if self.counter == 1 else 1
        string_state = np.array_str(self.transform_state(state))
        # print(string_state)
        if game_status != 0:
            return -game_status

        if string_state not in self.p.keys():
            v = self.nnet.predict(state, self.counter, self.opponent_counter)
            self.p[string_state] = v
            # some stuff about valid moves here
            self.n[string_state] = np.zeros(7)
            self.q[string_state] = np.array([-100.0 for i in range(7)])
            return -v.mean()

        cur_best = -1000
        best_act = -1

        for a in range(7):
            if state[0, a] == 0:
                if self.q[string_state][a] != -100:
                    u = self.q[string_state][a] + self.cput * self.p[string_state][a] * math.sqrt(
                        self.n[string_state].sum()) / (
                                                      1 + self.n[string_state][a])
                else:
                    u = self.cput * self.p[string_state][a] * math.sqrt(self.n[string_state].sum() + 1e-8)

                if u > cur_best:
                    cur_best = u
                    best_act = a

        a = best_act
        curr_game.action_board(a, self.counter)
        self.counter = self.opponent_counter
        if np.count_nonzero(curr_game.board) != 42:
            v = self.search(state=curr_game.board)
        else:
            v = -1

        if self.q[string_state][a] != -100:
            self.q[string_state][a] = (self.n[string_state][a] * self.q[string_state][a] + v) / (
                self.n[string_state][a] + 1)
            self.n[string_state][a] += 1
        else:
            self.q[string_state][a] = v
            self.n[string_state][a] = 1

        return -v

        # if __name__ == '__main__':
        # mcts = MCTS(AlphaZeroNN(epsilon=0))
        # # mcts.search(np.zeros((6, 7)), 2)
        # # mcts.search(np.zeros((6, 7)), 2)
        # print(mcts.action_prob(np.zeros((6, 7)), 1))
        # print(mcts.p, mcts.q, mcts.n)
