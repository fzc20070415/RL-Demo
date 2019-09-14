"""
Design:
    1. Start from the most left.
    2. Randomly choose an action.
    3. Evaluate the value of that action, update Q-Table.
    4. End game if treasure is reached. Restart game again.
"""
import time
import random
import numpy as np

# Parameters
EPSILON = 0.9   # Greedy Policy
ALPHA = 0.1     # Learing Rate
LAMBDA = 0.9    # Discount Factor
TIME_GAP = 0.5
TREASURE_LOCATION = 10
MAX_EPISODE = 20

LEFT = 0
RIGHT = 1

def choose_action(q_table, state):
    """
    Parameters:
        q_table - Q Table
        state - the current state we are at
    Return:
        action_name - the action we choose
    """
    # Fill Q-Table if not initialized
    # if len(q_table) <= state:
        # q_table.append([0,0])

    state_action = q_table[state]
    if random.random() > EPSILON or state_action==[0,0]:
        # Use random choice
        action_name = LEFT if random.random()<0.5 else RIGHT
    else:
        # Act Greedy
        action_name = LEFT if state_action[0]>state_action[1] else RIGHT
    return action_name

def get_env_feedback(S, A):
    """
    Parameters:
        S - the current state
        A - the action we choose to perform
    Return:
        SS - the updated state we are at
        R - the reward of the current action
    """
    # How environment react to our action
    R = 0
    if A == RIGHT:
        if S == TREASURE_LOCATION - 2:
            SS = 'terminal'
            R = 1
        else:
            SS = S + 1
    else:
        if S == 0:
            SS = 0
        else:
            SS = S - 1
    return SS, R

def update_env(S, episode, step_counter, q_table=None):
    """
    Parameters:
        S - the current state
        episode - 
        step_counter - the total number of actions performed in the current episode
    Return:
        None
    """
    # Refresh the display
    env_list = ['-']*(TREASURE_LOCATION-1) + ['T']
    if S == 'terminal':
        output = "Episode {}: Total_steps = {}".format(episode, step_counter)
        # print("\r{}".format(output), end='')
        print("\r{}".format(output))
        # time.sleep(2)
        # print("\r{}".format(" "*20), end='')
        # if q_table:
            # print(*q_table, sep='\n')
    else:
        env_list[S] = 'o'
        # if q_table:
            # print(*q_table, sep='\n')
        print("\rStep {}: {}".format(step_counter, ''.join(env_list)), end='')
        time.sleep(TIME_GAP)

def update_q_table(q_table, S, A, SS, R):
    """
    Parameters:
        q_table - Q Table
        S - the state before action
        A - the action chosen
        SS - the state after action
        R - the reward from env feedback
    Return:
        None
    """
    q_predict = q_table[S][A]
    if SS != "terminal":
        q_target = R + LAMBDA * max(q_table[SS])
    else:
        q_target = R

    ### KEY STEP: Update Q Table ###
    q_table[S][A] += ALPHA * (q_target - q_predict)

def RL():
    # Main Loop
    q_table = [[0]*2 for i in range(TREASURE_LOCATION-1)]

    for episode in range(MAX_EPISODE):
        step_counter = 0    # Initialize step counter
        S = 0               # Initial state at 0
        is_terminated = False

        update_env(S, episode, step_counter)
        while not is_terminated:
            A = choose_action(q_table, S)
            SS, R = get_env_feedback(S, A)
            update_q_table(q_table, S, A, SS, R)
            if SS == "terminal":
                is_terminated = True
            S = SS
            step_counter += 1
            update_env(S, episode, step_counter, q_table)
    return q_table


if __name__ == "__main__":
    Q_TABLE = RL()
    print("Q Table:")
    print(*Q_TABLE, sep='\n')
    
