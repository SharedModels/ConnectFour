import math


class ConnectPlayer(object):
    def __init__(self, name,counter, nn):
        self.name = name
        self.counter = counter
        self.opponent_counter = 1 if self.counter == 2 else 2
        self.state_record = []
        self.action_record = []
        self.reward = -1
        self.decision = nn

    def action(self, state):
        # build in invalid move here
        self.state_record.append(state.copy())
        action = self.decision.action(state, self.counter, self.opponent_counter)
        self.action_record.append(action)
        return action

    def constant_reward(self):
        return [self.reward for i in range(len(self.state_record))]

    def delayed_reward(self):
        return list(reversed([self.reward * math.pow(0.9, i) for i in range(len(self.state_record))]))

    def output_game_data(self):
        return self.state_record, self.action_record, self.constant_reward()


# class MCTSConnectPlayer(ConnectPlayer):
#
#     def action(self, state):
#         self.state_record.append(state.copy())
#         action_prob = self.decision.action_prob(state, self.counter)
#         action = np.random.choice(len(action_prob), p=action_prob)
#         self.action_record.append(action)
#         return action


