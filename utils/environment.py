import numpy as np
import torch

# Creates grid world map, state is [S_1_k, S_2_k, S_3_k, S_4_k, r_1_k-1, r_2_k-1, r_3_k-1, r_4_k-1]
# actions is [a_1, a_2, a_3, a_4]
# 0=up, 1=right, 2=down, 3=left 

class GridWorld:
    def __init__(self, rooms=np.array([1,1,1,2,0,0,0,0]), size_of_maze=30):#, seed=None):

        # gives L of map
        self.size_of_maze = size_of_maze

        # put four agents in the middle of the map, and have last rewards be 0
        self.state = [404, 405, 434, 435, 0, 0, 0, 0]

        # checking valiidity of rooms
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
            self.border_map[0:12,0:6] = np.array([
                [0,0,0,0,0,0], # vertical 1
                [0,0,0,0,0,0], # horizontal 1
                [0,0,0,0,0,1], # vertical 2
                [0,0,0,0,0,0], # horizontal 2
                [0,0,0,0,0,1], # vertical 3
                [0,0,0,0,0,0], # horizontal 3
                [0,0,0,0,0,1], # vertical 4
                [0,0,0,0,0,0], # horizontal 4
                [0,0,0,0,0,1], # vertical 5
                [0,0,0,0,0,0], # horizontal 5
                [0,0,0,0,0,1], # vertical 6
                [1,1,1,1,1,1], # horizontal 6
                ])
            
            if rooms[0] == 2:
                # creating 1st room goal state
                self.goal_state = np.array([0,1,2,3,4,5,
                                            30,31,32,33,34,35,
                                            60,61,62,63,64,65,
                                            90,91,92,93,94,95,
                                            120,121,122,123,124,125,
                                            150,151,152,153,154,155])

            
        if rooms[1] > 0:
            # creating 2nd room
            self.border_map[0:12,11:18] = np.array([
                [1,0,0,0,0,0,0], # vertical 1
                [0,0,0,0,0,0,0], # horizontal 1
                [1,0,0,0,0,0,1], # vertical 2
                [0,0,0,0,0,0,0], # horizontal 2
                [1,0,0,0,0,0,1], # vertical 3
                [0,0,0,0,0,0,0], # horizontal 3
                [1,0,0,0,0,0,1], # vertical 4
                [0,0,0,0,0,0,0], # horizontal 4
                [1,0,0,0,0,0,1], # vertical 5
                [0,0,0,0,0,0,0], # horizontal 5
                [1,0,0,0,0,0,1], # vertical 6
                [0,1,1,1,1,1,1], # horizontal 6
                ])

            if rooms[1] == 2:
                # creating 2nd room goal state
                self.goal_state = np.array([12,13,14,15,16,17,
                                            42,43,44,45,46,47,
                                            72,73,74,75,76,77,
                                            102,103,104,105,106,107,
                                            132,133,134,135,136,137,
                                            162,163,164,165,166,167])

        if rooms[2] > 0:
            # creating 3rd room
            self.border_map[0:12,23:30] = np.array([
                [1,0,0,0,0,0,1], # vertical 1
                [0,0,0,0,0,0,0], # horizontal 1
                [1,0,0,0,0,0,1], # vertical 2
                [0,0,0,0,0,0,0], # horizontal 2
                [1,0,0,0,0,0,1], # vertical 3
                [0,0,0,0,0,0,0], # horizontal 3
                [1,0,0,0,0,0,1], # vertical 4
                [0,0,0,0,0,0,0], # horizontal 4
                [1,0,0,0,0,0,1], # vertical 5
                [0,0,0,0,0,0,0], # horizontal 5
                [1,0,0,0,0,0,1], # vertical 6
                [0,1,1,1,1,1,0], # horizontal 6
                ])

            if rooms[2] == 2:
                # creating 3rd room goal state
                self.goal_state = np.array([24,25,26,27,28,29,
                                            54,55,56,57,58,59,
                                            84,85,86,87,88,89,
                                            114,115,116,117,118,119,
                                            144,145,146,147,148,149,
                                            174,175,176,177,178,179])


        if rooms[3] > 0:
            # creating 4th room
            self.border_map[23:36,23:30] = np.array([
                [0,1,1,1,1,1,1], # horizontal 12
                [1,0,0,0,0,0,1], # vertical 13
                [0,0,0,0,0,0,0], # horizontal 13
                [1,0,0,0,0,0,1], # vertical 14
                [0,0,0,0,0,0,0], # horizontal 14
                [1,0,0,0,0,0,1], # vertical 15
                [0,0,0,0,0,0,0], # horizontal 15
                [1,0,0,0,0,0,1], # vertical 16
                [0,0,0,0,0,0,0], # horizontal 16
                [1,0,0,0,0,0,1], # vertical 17
                [0,0,0,0,0,0,0], # horizontal 17
                [1,0,0,0,0,0,1], # vertical 18
                [0,1,1,1,1,1,0], # horizontal 18
                ])

            if rooms[3] == 2:
                # creating 4th room goal state
                self.goal_state = np.array([384,385,386,387,388,389,
                                       414,415,416,417,418,419,
                                       444,445,446,447,448,449,
                                       474,475,476,477,478,479,
                                       504,505,506,507,508,509, 
                                       534,535,536,537,538,539])
        if rooms[4] > 0:
            # creating 5th room
            self.border_map[47:60,23:30] = np.array([
                [0,1,1,1,1,1,1], # horizontal 24
                [1,0,0,0,0,0,1], # vertical 24
                [0,0,0,0,0,0,0], # horizontal 25
                [1,0,0,0,0,0,1], # vertical 25
                [0,0,0,0,0,0,0], # horizontal 26
                [1,0,0,0,0,0,1], # vertical 26
                [0,0,0,0,0,0,0], # horizontal 27
                [1,0,0,0,0,0,1], # vertical 27
                [0,0,0,0,0,0,0], # horizontal 28
                [1,0,0,0,0,0,1], # vertical 28
                [0,0,0,0,0,0,0], # horizontal 29
                [0,0,0,0,0,0,1], # vertical 29
                ])

            if rooms[4] == 2:
                # creating 5th room goal state
                self.goal_state = np.array([744,745,746,747,748,749,
                                            774,775,776,777,778,779,
                                            804,805,806,807,808,809,
                                            834,835,836,837,838,839,
                                            864,865,866,867,868,869,
                                            894, 895, 896, 897, 898, 899])

        if rooms[5] > 0:
            # creating 6th room
            self.border_map[47:60,11:18] = np.array([
                [0,1,1,1,1,1,1], # horizontal 24
                [1,0,0,0,0,0,1], # vertical 24
                [0,0,0,0,0,0,0], # horizontal 25
                [1,0,0,0,0,0,1], # vertical 25
                [0,0,0,0,0,0,0], # horizontal 26
                [1,0,0,0,0,0,1], # vertical 26
                [0,0,0,0,0,0,0], # horizontal 27
                [1,0,0,0,0,0,1], # vertical 27
                [0,0,0,0,0,0,0], # horizontal 28
                [1,0,0,0,0,0,1], # vertical 28
                [0,0,0,0,0,0,0], # horizontal 29
                [0,0,0,0,0,0,1], # vertical 29
                ])
            
            if rooms[5] == 2:
                self.goal_state = np.array([732,733,734,735,736,737, 
                                            762,763,764,765,766,767,
                                            792,793,794,795,796,797,
                                            822,823,824,825,826,827,
                                            852,853,854,855,856,857,
                                            882,883,884,885,886,887])
            
        if rooms[6] > 0:
            # creating 7th room
            self.border_map[47:60,0:6] = np.array([
                [0,1,1,1,1,1], # horizontal 24
                [0,0,0,0,0,1], # vertical 24
                [0,0,0,0,0,0], # horizontal 25
                [0,0,0,0,0,1], # vertical 25
                [0,0,0,0,0,0], # horizontal 26
                [0,0,0,0,0,1], # vertical 26
                [0,0,0,0,0,0], # horizontal 27
                [0,0,0,0,0,1], # vertical 27
                [0,0,0,0,0,0], # horizontal 28
                [0,0,0,0,0,1], # vertical 28
                [0,0,0,0,0,0], # horizontal 29
                [0,0,0,0,0,1], # vertical 29
                ])
            
            if rooms[6] == 2:
                # creating 7th room goal state
                self.goal_state = np.array([720,721,722,723,724,725,
                                            750,751,752,753,754,755, 
                                            780,781,782,783,784,785, 
                                            810,811,812,813,814,815, 
                                            840,841,842,843,844,845, 
                                            870,871,872,873,874,875])

        if rooms[7] > 0:
            # creating 8th room
            self.border_map[24:37,0:6] = np.array([
                [0,1,1,1,1,1], # horizontal 12
                [0,0,0,0,0,1], # vertical 13
                [0,0,0,0,0,0], # horizontal 13
                [0,0,0,0,0,1], # vertical 14
                [0,0,0,0,0,0], # horizontal 14
                [0,0,0,0,0,1], # vertical 15
                [0,0,0,0,0,0], # horizontal 15
                [0,0,0,0,0,1], # vertical 16
                [0,0,0,0,0,0], # horizontal 16
                [0,0,0,0,0,1], # vertical 17
                [0,0,0,0,0,0], # horizontal 17
                [0,0,0,0,0,1], # vertical 18
                [1,1,1,1,1,1], # horizontal 18
                ])

            if rooms[7] == 2:
                # creating 8th room
                self.goal_state = np.array([360,361,362,363,364,365,
                                            390,391,392,393,394,395,
                                            420,421,422,423,424,425,
                                            450,451,452,453,454,455,
                                            480,481,482,483,484,485, 
                                            510,511,512,513,514,515])

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
        self.state = np.array([404, 405, 434, 435, 0, 0, 0, 0])

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
        for i in range(4):
            if self.state[i] in self.goal_state:
                reward += 10
                self.state[i+4] = 10
            else:
                reward -= 1
                self.state[i+4] = -1
        if reward == 4*10:
            reward = 1000
            done = True
        else:
            done = False
        return torch.tensor(self.state).float(), reward, done


# setup_rooms = np.array([1,1,1,2,0,0,0,0])

# print("np.shape(setup_rooms)[0]")
# print(np.shape(setup_rooms)[0])

# print("np.shape(setup_rooms)[1]")
# print(np.shape(setup_rooms)[1])

# env = GridWorldMazeEnv(setup_rooms)


# done = False

# while done == False:

#     print("env.state")
#     print(env.state)

#     user_action_1 = input("action for first agent: ")
#     user_action_2 = input("action for second agent: ")
#     user_action_3 = input("action for third agent: ")
#     user_action_4 = input("action for fourth agent: ")

#     user_action = np.array([int(user_action_1), int(user_action_2), int(user_action_3), int(user_action_4)])

#     print("user_action")
#     print(user_action)

#     env.step(user_action)
