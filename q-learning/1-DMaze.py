import numpy as np
import pandas as pd
import time

N_STATES = 6
ACTIONS = ['left', 'right']
EPSILON = 0.9
ALPHA = 0.1
GAMMA = 0.9
MAX_EPISODES = 13
FRESH_TIME = 0.3

def build_q_table(n_state, actions):
    table = pd.DataFrame(
        np.zeros((n_state, len(actions))),
        columns = actions
    )
    return table

# print(build_q_table(N_STATES, ACTIONS))

def choose_action(state, q_table):
    state_action = q_table.iloc[state, :]
    if np.random.uniform() > EPSILON or state_action.all() == 0:
        action_name = np.random.choice(ACTIONS)
    else:
        action_name = state_action.argmax()
    return action_name

def get_env_feedback(S, A):
    if A == 'right':
        if S == N_STATES-2:
            S_ = 'terminal'
            R = 1
        else:
            R = 0
            S_ = S + 1
    else:
        R = 0
        if S == 0:
            S_ = S
        else:
            S_ = S - 1
    return S_, R

def update_env(S, episode, step_counter):
    # This is how environment be updated
    env_list = ['-']*(N_STATES-1) + ['T']   # '---------T' our environment
    if S == 'terminal':
        interaction = 'Episode %s: total_steps = %s' % (episode+1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print('\r                                ', end='')
    else:
        env_list[S] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)

def rl():
    q_table = build_q_table(N_STATES, ACTIONS)
    for i in range(MAX_EPISODES):
        s = 0
        is_terminal = False
        step_counter = 0
        update_env(s, i, step_counter)
        while not is_terminal:
            a = choose_action(s, q_table)
            # print(a)
            s_, r = get_env_feedback(s, a)
            # q_table.iloc[s, a] = q_table.iloc[s, a] + ALPHA*(r + GAMMA * max(q_table.iloc[S_, :].max())
            q_predict = q_table.ix[s, a]
            if s_ == 'terminal':
                q_target = r
                is_terminal = True
            else:
                q_target = r + GAMMA * q_table.iloc[s_, :].max()
            q_table.ix[s, a] += ALPHA * (q_target-q_predict)
            s = s_
            step_counter += 1
            update_env(s, i, step_counter)
            
    return q_table

def main():
    rl()

if __name__ == '__main__':
    main()
