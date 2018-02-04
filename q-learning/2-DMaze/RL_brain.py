import pandas as pd
import numpy as np

class QLearningTable:
    def __init__(self, actions, learning_rate = 0.01, reward_decay = 0.9, e_greedy = 0.9):
        self.actions = actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(
            columns = self.actions,
            dtype = np.float64
        )

    def choose_action(self, observation):
        # print(observation)
        # exit(0)
        self.check_state_exist(observation)
        # print(self.q_table)
        # exit(0)
        all_state = self.q_table.loc[observation, :]
        if np.random.uniform() > self.epsilon:
            action = np.random.choice(self.actions)
        else:
            all_state = all_state.reindex(np.random.permutation(all_state.index))
            action = all_state.idxmax()
        return action
    
    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        # print(type(a))
        # exit(0)
        q_predict = self.q_table.ix[s, a]
        if s == 'terminal':
            q_target = r
        else:
            q_target = r + self.gamma * self.q_table.ix[s_, :].max()
        # print(a)
        # print(self.q_table.columns)
        # exit(0)
        # print(self.q_table.ix[s, a])
        # exit(0)
        # print(q_target)
        self.q_table.ix[s, a] += self.lr * (q_target - q_predict)
        # print(self.q_table)


    def check_state_exist(self, state):
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(
                pd.Series(
                    [0] * len(self.actions),
                    index = self.q_table.columns,
                    name = state,
                )
            )