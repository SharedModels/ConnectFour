from player import ConnectPlayer
from connect_nn import NNAction, AlphaZeroNN
from collections import Counter
from human_play import HumanPlay
from mcts import MCTS
from base_game import ConnectFourGame


class MultipleGames:
    def __init__(self, nn_class):
        self.nn_class = nn_class
        self.nn_a = self.nn_class(counter=1, opponent_counter=2)
        self.nn_b = self.nn_class(counter=2, opponent_counter=1)
        self.game = None
        self.epsilon = 1

    def save_data(self):
        for player in self.game.player_list:
            state, action, reward = player.output_game_data()
            player.decision.nnet.save_data(state, action, reward)

    def train_nns(self):
        [nn.train() for nn in [self.nn_a, self.nn_b]]

    def train_target_q(self):
        [nn.train_target_q() for nn in [self.nn_a, self.nn_b]]

    def retrieve_player_by_name(self, name):
        for player in self.game.player_list:
            if player.name == name:
                return player

    def swap_models(self, bad, good):
        good_player = self.retrieve_player_by_name(good)
        bad_player = self.retrieve_player_by_name(bad)
        bad_player.nn.swap(good_player.nn)

    def simulate_many_games(self, n):
        wins = []
        for i in range(n):
            self.game = ConnectFourGame(MCTS(self.nn_a, counter=1, epsilon=self.epsilon),
                                        MCTS(self.nn_b, counter=2, epsilon=self.epsilon))
            player = self.game.run_game()
            if player != 'draw':
                player.reward = 1
                wins.append(player.name)
            print(Counter(wins[-200:]))

            self.save_data()
            self.train_nns()
            if self.epsilon > 0.1:
                self.epsilon *= 0.999
            if i % 10 == 0:
                self.train_target_q()
            if i % 10 == 0:
                [nn.save() for nn in [self.nn_a, self.nn_b]]



class AlphaZero:
    def __init__(self, simulation_length=50, iteration_length=100, win_per=55):
        self.simulation_length = simulation_length
        self.iteration_length = iteration_length
        self.win_per = win_per
        self.current_nn = AlphaZeroNN()
        self.new_nn = None
        self.game = None
        self.mcts = None

    def save_data(self):
        for player in self.game.player_list:
            state, action, reward = player.output_game_data()
            self.current_nn.save_data(state, action, reward, player.counter, player.opponent_counter)
            # print(player.nn.rl_memory)

    def run_simulation(self):
        for i in range(self.simulation_length):
            mcts = MCTS(self.current_nn)
            self.game = ConnectFourGame(mcts, mcts)
            player = self.game.run_game()
            if player != 'draw':
                player.reward = 1
            self.save_data()

        # TODO Check this works
        self.new_nn = AlphaZeroNN()
        self.new_nn.copy(self.current_nn)
        self.new_nn.train()
        self.current_nn.rl_memory = []

    def compare_nns(self):
        wins = []
        for i in range(self.simulation_length):
            self.game = ConnectFourGame(MCTS(self.current_nn), MCTS(self.new_nn))
            player = self.game.run_game()
            if player != 'draw':
                wins.append(player.name)

        print(Counter(wins))
        if Counter(wins)['b'] > self.simulation_length * self.win_per / self.simulation_length:
            self.current_nn = AlphaZeroNN()
            self.current_nn.copy(self.new_nn)
            self.current_nn.save()

    def iterate_models(self):
        for i in range(self.iteration_length):
            self.run_simulation()
            self.compare_nns()


class HumanGame:
    def __init__(self, nn_class, human_play):
        self.nn_a = nn_class(load_filepath='q_model.h5', epsilon=0)
        self.b_human = human_play()
        self.game = None

    def run_game(self):
        self.game = ConnectFourGame(MCTS(self.nn_a, num_sim=50, counter=1, epsilon=0), self.b_human)
        self.game.run_game()


if __name__ == '__main__':
    rl = MultipleGames(NNAction)
    rl.simulate_many_games(100000)

    # rl = AlphaZero()
    # rl.iterate_models()

    # rl = HumanGame(NNAction, HumanPlay)
    # rl.run_game()
