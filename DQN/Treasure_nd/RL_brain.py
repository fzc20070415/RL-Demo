import random
import numpy as np

class QL:
    def __init__(self, size, EPSILON, LAMBDA, ALPHA):
        self.size = size
        # self.q_table = [[[0,0,0,0] for i in range(size)] for i in range(size)]
        self.q_table = np.zeros([size, size, 4])
        self.EPSILON = EPSILON
        self.LAMBDA = LAMBDA
        self.ALPHA = ALPHA
    
    def learn(self, O, A, R, OO, is_terminated):
        """
        Parameters:
            O - current observation
            A - current action
            R - reward of current action
            OO - next observation
            is_terminated - 
        Return:
            None
        """
        q_predict = self.q_table[O[0],O[1]][A]
        if is_terminated:
            q_target = R
        else:
            q_target = R + self.LAMBDA * max(self.q_table[OO[0], OO[1]])
        
        # Update q_table
        self.q_table[O[0],O[1]][A] += self.ALPHA * (q_target - q_predict)
        # print(O,A,OO)

    def choose_action(self, O):
        """
        Parameters:
            O - current observation
        Return:
            A - action
        """
        state_table = self.q_table[O[0],O[1]]
        if random.random() > self.EPSILON or (state_table == 0).all():
            random_number = random.random()
            if random_number < 0.25:
                A = 0
            elif random_number < 0.5:
                A = 1
            elif random_number < 0.75:
                A = 2
            else:
                A = 3
        else:
            A = np.random.choice(np.where(state_table==np.max(state_table))[0])
        return A

    def print(self):
        print(self.q_table)