from maze_env import Maze
from RL_brain import QL
import time

# Parameters
EPSILON = 0.9   # Greedy Policy
ALPHA = 0.1     # Learing Rate
LAMBDA = 0.9    # Discount Factor
MAX_EPISODE = 50
MAZE_SIZE = 5
TRAP_SET = [[0,1],[2,2],[3,3],[3,0]]
TREASURE_SET = [[4,4]]

env = Maze(MAZE_SIZE)
RL = QL(MAZE_SIZE, EPSILON, LAMBDA, ALPHA)

def update():
    for episode in range(MAX_EPISODE):
        O = env.reset()     # return initial observation
        step_number = 0
        env.set_trap(TRAP_SET)
        env.set_treasure(TREASURE_SET)
        is_terminated = False
        env.render()

        while not is_terminated:
            A = RL.choose_action(O)
            # print(O)
            OO, R, is_terminated = env.step(A)
            # print(O,OO)
            RL.learn(O, A, R, OO, is_terminated)
            # print(O,OO)
            O = OO.copy()

            print("Episode {} Step {}:".format(episode, step_number))
            env.render()    # Print env

            time.sleep(0.3)
            step_number += 1
        
        print("\n\nEpisode {}: Completed in {} steps\n\n".format(episode, step_number))
        # RL.print()
        time.sleep(3)

if __name__ == "__main__":
    update()