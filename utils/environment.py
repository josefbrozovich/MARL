import numpy as np
import torch

# Creates grid world map, state is [S_1_k, S_2_k, S_3_k, S_4_k, r_1_k-1, r_2_k-1, r_3_k-1, r_4_k-1]
# actions is [a_1, a_2, a_3, a_4]
# 0=up, 1=right, 2=down, 3=left 

class GridWorld:
    def __init__(self, rooms=np.array([1,1,1,2,0,0,0,0]), size_of_maze=15):#, seed=None):

        # gives L of map
        self.size_of_maze = size_of_maze

        # put four agents in the middle of the map, and have last rewards be 0
        self.state = np.array([111, 112, 113, 127])

        # checking valiidity of actions
        if np.shape(rooms) != (8,):
            raise ValueError("The rooms should be an array of length 8")
        counts = np.bincount(rooms)
        if counts[1] != 3:
            raise ValueError("The rooms have 3 normal rooms")
        if counts[2] != 1:
            raise ValueError("The rooms have 1 goal state")
        # 

        self.border_map = np.zeros((2*self.size_of_maze-1,self.size_of_maze))
        # self.boarder map has pattern of
        # vertical
        # horizontal
        # vertical
        # horizontal
        # ...
        # vertical

        # creating boarder on far right side of grid
        right_boarder = np.zeros(2*self.size_of_maze-1)
        right_boarder[::2] = 1
        self.border_map[:,self.size_of_maze-1] = right_boarder

        # Creating custom maps

        if rooms[0] > 0:
            # creating 1st room
            self.border_map[0:6,0:3] = np.array([
                [0,0,0], # vertical 1
                [0,0,0], # horizontal 1
                [0,0,0], # vertical 2
                [0,0,0], # horizontal 2
                [0,0,0], # vertical 3
                [1,1,1], # horizontal 3
                ])
            
            if rooms[0] == 2:
                # creating 1st room goal state
                self.goal_state = np.array([0,1,2,
                                            15,16,17,
                                            30,31,32])
            
        if rooms[1] > 0:
            # creating 2nd room
            self.border_map[0:6,5:9] = np.array([
                [1,0,0,0], # vertical 1
                [0,0,0,0], # horizontal 1
                [1,0,0,0], # vertical 2
                [0,0,0,0], # horizontal 2
                [1,0,0,0], # vertical 3
                [0,1,1,1], # horizontal 3
                ])

            if rooms[1] == 2:
                # creating 2nd room goal state
                self.goal_state = np.array([6,7,8,
                                            21,22,23,
                                            36,37,38])

        if rooms[2] > 0:
            # creating 3rd room
            self.border_map[0:6,11:15] = np.array([
                [1,0,0,1], # vertical 1
                [0,0,0,0], # horizontal 1
                [1,0,0,1], # vertical 2
                [0,0,0,0], # horizontal 2
                [1,0,0,1], # vertical 3
                [0,0,0,0], # horizontal 3
                ])

            if rooms[2] == 2:
                # creating 3rd room goal state
                self.goal_state = np.array([12,13,14,
                                            27,28,29,
                                            42,43,44])

        if rooms[3] > 0:
            # creating 4th room
            self.border_map[11:18,11:15] = np.array([
                [0,1,1,1], # horizontal 6
                [1,0,0,1], # vertical 7
                [0,0,0,0], # horizontal 7
                [1,0,0,1], # vertical 8
                [0,0,0,0], # horizontal 8
                [1,0,0,1], # vertical 9
                [0,0,0,0], # horizontal 9
                ])

            if rooms[3] == 2:
                # creating 4th room goal state
                self.goal_state = np.array([102, 103, 104,
                                            117, 118, 119,
                                            132, 133, 134])
        if rooms[4] > 0:
            # creating 5th room
            self.border_map[23:,11:15] = np.array([
                [0,1,1,1], # horizontal 12
                [0,0,0,1], # vertical 13
                [0,0,0,0], # horizontal 13
                [0,0,0,1], # vertical 14
                [0,0,0,0], # horizontal 14
                [0,0,0,1], # vertical 15
                ])

            if rooms[4] == 2:
                # creating 5th room goal state
                self.goal_state = np.array([192, 193, 194, 
                                            207, 208, 209, 
                                            222, 223, 224])

        if rooms[5] > 0:
            # creating 6th room
            self.border_map[23:,5:9] = np.array([
                [0,1,1,1], # horizontal 12
                [0,0,0,1], # vertical 13
                [0,0,0,0], # horizontal 13
                [0,0,0,1], # vertical 14
                [0,0,0,0], # horizontal 14
                [0,0,0,1], # vertical 15
                ])
            
            if rooms[5] == 2:
                self.goal_state = np.array([186, 187, 188, 
                                            201, 202, 203, 
                                            216, 217, 218])
            
        if rooms[6] > 0:
            # creating 7th room
            self.border_map[23:,0:3] = np.array([
                [0,0,0], # horizontal 12
                [0,0,1], # vertical 13
                [0,0,0], # horizontal 13
                [0,0,1], # vertical 14
                [0,0,0], # horizontal 14
                [0,0,1], # vertical 15
                ])
            
            if rooms[6] == 2:
                # creating 7th room goal state
                self.goal_state = np.array([180,181,182,
                                            195,196,197,
                                            210,211,212])

        if rooms[7] > 0:
            # creating 8th room
            self.border_map[11:18,0:3] = np.array([
                [0,0,0], # horizontal 6
                [0,0,1], # vertical 7
                [0,0,0], # horizontal 7
                [0,0,1], # vertical 8
                [0,0,0], # horizontal 8
                [0,0,1], # vertical 9
                [1,1,1], # horizontal 9
                ])

            if rooms[7] == 2:
                # creating 8th room
                self.goal_state = np.array([90,91,92,
                                            105,106,107,
                                            120,121,122])


    def state_to_border(self, state):
        row, column = np.divmod(state, self.size_of_maze)
        border = np.zeros((4,))  # up, right, down, left
        if row == 0:
            border[0] = 1
        else:
            border[0] = self.border_map[row * 2 - 1, column]
        border[1] = self.border_map[row * 2, column]
        if row == self.size_of_maze - 1:
            border[2] = 1
        else:
            border[2] = self.border_map[row * 2 + 1, column]
        if column == 0:
            border[3] = 1
        else:
            border[3] = self.border_map[row * 2, column - 1]
        return border

    def reset(self):
        self.state = 0
        # put all 4 agents in the middle of the grid
        self.state = np.array([111, 112, 113, 127])

        return torch.from_numpy(self.state).float(), False

    def state_transition_func(self, state, action):
        for i in range(4):
            assert state[i] in range(self.size_of_maze**2), "Error: The state input is invalid!"

        # checking valiidity of rooms
        if np.shape(action) != (4,):
            raise ValueError("Each agent should have its own action")
        if np.all(~((action >= 0) & (action <= 4))):
            raise ValueError("Each action should be between 0 and 3")
        # 

        next_state = state
        for i in range(4):
            border = self.state_to_border(state[i])
            if action[i] == 0:
                if border[0] == 0:
                    next_state[i] = state[i] - self.size_of_maze
            elif action[i] == 1:
                if border[1] == 0:
                    next_state[i] = state[i] + 1
            elif action[i] == 2:
                if border[2] == 0:
                    next_state[i] = state[i] + self.size_of_maze
            elif action[i] == 3:
                if border[3] == 0:
                    next_state[i] = state[i] - 1

        return next_state

    def step(self, action):

        # checking valiidity of rooms
        if np.shape(action) != (4,):
            raise ValueError("Each agent should have its own action")
        if np.all(~((action >= 0) & (action <= 4))):
            raise ValueError("Each action should be between 0 and 3")
        # 
        self.state = self.state_transition_func(self.state, action)
        reward = 0
        done = False
        for i in range(4):
            if self.state[i] in self.goal_state:
                reward += 10
                done = True
            else:
                reward -= 1
        return torch.tensor(self.state).float(), reward, done

