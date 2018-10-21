import numpy as np
from keras.layers import Dense, Activation, Conv2D, Flatten, Dropout, BatchNormalization, Input, Softmax
from keras.models import Sequential, load_model
import pandas as pd


class NNAction:
    def __init__(self, input_dim=(2, 6, 7), hidden_dim=128, output_dim=7, epsilon=0.99, bellman_value=0.95,
                 buffer_size=40000, load_filepath=None, counter=None, opponent_counter=None):
        """
        class to predict actions for battleships game, using reinforcement learning.
        :param input_dim: tuple
         Input shape for NN
        :param hidden_dim: integer
         hidden layer size (not used for cnn as too complex)
        :param output_dim: integer
        output size for NN
        :param epsilon: float
        Value for e greedy policy, decays by 0.99 every training session
        :param bellman_value: float
        value to use in bellman equation
        :param buffer_size: integer
        size of rl buffer
        """
        self.input_dim = input_dim
        self.reshape_dim = (1,) + self.input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.q_model = self.build_cnn_model()
        self.target_q_model = self.build_cnn_model()
        self.epsilon = epsilon
        self.rl_memory = []
        self.bellman_value = bellman_value
        self.buffer_size = buffer_size
        self.counter = counter
        self.opponent_counter = opponent_counter
        if load_filepath is not None:
            self.q_model = load_model(load_filepath)

    def build_cnn_model(self):
        """
        CNN for battleships, seems to work a lot better than flat version
        :return: keras model
        """
        model = Sequential()
        # model.add(Input(shape = (self.input_dim)))
        # model.add(BatchNormalization(axis=1))
        model.add(Conv2D(128, input_shape=(self.input_dim), kernel_size=(3), activation='relu',
                         data_format='channels_first', padding='same'))
        model.add(BatchNormalization(axis=3))
        # model.add(Activation('relu'))
        model.add(Conv2D(filters=128, kernel_size=(3), activation='relu', padding='same',
                         data_format='channels_first'))  # , strides=(3, 3)))
        model.add(BatchNormalization(axis=3))
        model.add(Conv2D(filters=128, kernel_size=(3), activation='relu', padding='valid',
                         data_format='channels_first'))  # , strides=(3, 3)))
        model.add(BatchNormalization(axis=3))
        model.add(Conv2D(filters=128, kernel_size=(3), activation='relu', padding='valid',
                         data_format='channels_first'))  # , strides=(3, 3)))
        model.add(BatchNormalization(axis=3))
        # model.add(Activation('relu'))
        # model.add(Conv2D(64, kernel_size=(1), activation='linear', data_format='channels_first'))
        # model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(1024))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(512))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.output_dim))
        model.add(Dropout(0.5))
        model.add(Softmax())
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'mse'])
        return model

    def transform_state_cnn(self, state):
        """
        Transform state into shape for cnn, add in a 1d channel. This might just be because used COnv2D where not nesc.
        :param state: numpy array shape (10, 10)
        :return: numpy array shape (1, 10, 10, 1)
        """
        own_array = state.copy()
        opponent_array = state.copy()
        own_array[state == self.counter] = 1
        own_array[state != self.counter] = 0
        opponent_array[state == self.opponent_counter] = -1
        opponent_array[state != self.opponent_counter] = 0
        # print('state: ', state)
        # print('ownarray', own_array)
        # print('opponentarray', opponent_array)
        # return state.reshape(self.reshape_dim)
        return np.stack([own_array, opponent_array]).reshape(self.reshape_dim)

    def action(self, state, c, oc):
        """
        Perform an action in battleships. Uses e-greedy policy where there is self.epsilon chance to just use a random
        action.
        :param state: numpy array shape (10, 10)
        :return: integer location of attack
        """
        transformed_state = self.transform_state_cnn(state)
        self.counter = c
        self.opponent_counter = oc
        # print(state, transformed_state)
        if np.random.uniform(0, 1) > self.epsilon:
            action_prob = self.q_model.predict(transformed_state)[0]
            print(action_prob)
            # Anything that is non-zero in the state is an invalid move, this makes it so invalid moves cant be used
            # print(action_prob)
            action_prob[state[0,] != 0] = -100
            if np.isnan(action_prob).any():
                ValueError('nans')
            # print(state)
            # print(action_prob)
            action = np.argmax(action_prob)
        else:
            action = np.random.randint(self.output_dim)

        return action

    def predict(self, state, c, oc):
        self.counter = c
        self.opponent_counter = oc
        transformed_state = self.transform_state_cnn(state)
        action_prob = self.q_model.predict(transformed_state)[0]
        print(state, action_prob)
        return action_prob

    def save_data(self, state_data, action_data, reward_data):
        """
        Save incoming data into the class rl_memory. Unless this is the last state, keep the next state in the tuple for
        calculating q values in the future. All lists must be the same length
        :param state_data: list of arrays
        :param action_data: list of tuples
        :param reward_data: list of integers
        """
        x_list = []
        y_list = []
        for i in range(len(state_data)):
            if i == len(state_data) - 1:
                self.rl_memory.append((state_data[i], action_data[i], reward_data[i], None))
            else:
                self.rl_memory.append((state_data[i], action_data[i], reward_data[i],
                                       state_data[i + 1]))
            x_list.append(state_data[i].reshape(6, 7))
            y_list.append(action_data[i])
        pd.DataFrame({'x': x_list, 'y': y_list}).to_csv('train_csv.csv')

    def prepare_train(self, batch):
        """
        Prepare the training data. Predict reward for all other actions that werent taken, add on reward for the next
        step for the action that was taken.
        :param batch: list of tuples
        :return: train_x numpy array of states, train_y numpy array of rewards
        """
        train_x = []
        train_y = []
        for i in batch:
            # print(self.transform_state_cnn(i[0]))
            target_q = self.target_q_model.predict(self.transform_state_cnn(i[0]))[0]
            if i[3] is None:
                state_reward = i[2]
            else:
                state_reward = i[2] + self.bellman_value * self.target_q_model.predict(self.transform_state_cnn(i[3]))[
                    0].max()

            target_q[i[1]] = state_reward
            # target_q[i[0].flatten() != 0] = 0
            train_x.append(self.transform_state_cnn(i[0]))
            train_y.append(target_q.reshape((1, 7)))
            # print(self.transform_state_cnn(i[0]))
        return np.concatenate(train_x), np.concatenate(train_y)

    def pick_rl(self):
        """
        Pick random batch of states for training
        :return: list
        """
        idx = np.random.choice(range(len(self.rl_memory)), 128, replace=True)
        return [self.rl_memory[i] for i in idx]

    def train(self):
        """
        Train the rl model, also reduces the size of self.epsilon
        """
        batch = self.pick_rl()

        train_x, train_y = self.prepare_train(batch)
        self.q_model.fit(train_x, train_y)
        self.rl_memory = self.rl_memory[-self.buffer_size:]
        if self.epsilon > 0.1:
            self.epsilon *= 0.9999999

    def train_target_q(self):
        """
        Update the target q model weights.
        """
        weights = self.q_model.get_weights()
        target_weights = self.target_q_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i]
        self.target_q_model.set_weights(target_weights)

    @staticmethod
    def swap_models(swap_from, swap_to):
        weights = swap_from.get_weights()
        target_weights = swap_to.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i]
        swap_from.set_weights(target_weights)

    def save(self):
        """
        Save the keras model
        """
        self.q_model.save('q_model.h5')


class AlphaZeroNN(NNAction):
    # def __init__(self, input_dim=(2, 6, 7), hidden_dim=128, output_dim=7, epsilon=0.99, bellman_value=0.95,
    #              buffer_size=40000, load_filepath=None):
    #     """
    #     class to predict actions for battleships game, using reinforcement learning.
    #     :param input_dim: tuple
    #      Input shape for NN
    #     :param hidden_dim: integer
    #      hidden layer size (not used for cnn as too complex)
    #     :param output_dim: integer
    #     output size for NN
    #     :param epsilon: float
    #     Value for e greedy policy, decays by 0.99 every training session
    #     :param bellman_value: float
    #     value to use in bellman equation
    #     :param buffer_size: integer
    #     size of rl buffer
    #     """
    #     self.input_dim = input_dim
    #     self.reshape_dim = (1,) + self.input_dim
    #     self.hidden_dim = hidden_dim
    #     self.output_dim = output_dim
    #     self.q_model = self.build_cnn_model()
    #     self.target_q_model = self.build_cnn_model()
    #     self.epsilon = epsilon
    #     self.rl_memory = []
    #     self.bellman_value = bellman_value
    #     self.buffer_size = buffer_size
    #     if load_filepath is not None:
    #         self.q_model = load_model(load_filepath)

    def copy(self, other_class):
        self.swap_models(other_class.q_model, self.q_model)
        self.swap_models(other_class.target_q_model, self.target_q_model)
        self.epsilon = other_class.epsilon
        self.rl_memory = other_class.rl_memory.copy()


if __name__ == '__main__':
    nn = NNAction()
    for data in range(10):
        nn.save_data([np.random.rand(10, 10), np.random.rand(10, 10)],
                     [(np.random.randint(0, 10), np.random.randint(0, 10)),
                      (np.random.randint(0, 10), np.random.randint(0, 10))],
                     [np.random.randint(0, 2), np.random.randint(0, 2)])
    nn.train()
