import numpy as np
LEFT = 0
RIGHT = 1
UP = 2
DOWN = 3

class Maze:
    def __init__(self, size):
        # self.map = np.zeros([size,size])
        self.size = size
        self.map = []
        self.reset()

    def reset(self):
        self.map = [['.']*self.size for i in range(self.size)]
        self.curser = [0,0]
        return self.curser

    def render(self):
        self.map[self.curser[0]][self.curser[1]] = '@'
        for line in self.map:
            print(' '.join(line))
    
    def set_trap(self, trap_set):
        self.trap_set = trap_set
        for item in trap_set:
            self.map[item[0]][item[1]] = '*'
    
    def set_treasure(self, treasure_set):
        self.treasure_set = treasure_set
        for item in treasure_set:
            self.map[item[0]][item[1]] = '$'

    def step(self, A):
        """
        Parameters:
            A - current action
        Return:
            OO - observation after action
            R - reward of this action
            is_terminated - 
        """
        self.map[self.curser[0]][self.curser[1]] = '.'
        if A == LEFT and not self.curser[1] == 0:
            self.curser[1] -= 1
        elif A == RIGHT and not self.curser[1] == self.size - 1:
            self.curser[1] += 1
        elif A == UP and not self.curser[0] == 0:
            self.curser[0] -= 1
        elif A == DOWN and not self.curser[0] == self.size - 1:
            self.curser[0] += 1
        OO = self.curser


        if OO in self.treasure_set:
            is_terminated = True
            R = 1
        elif OO in self.trap_set:
            is_terminated = True
            R = -2
        else:
            is_terminated = False
            R = 0
        
        return OO, R, is_terminated
