import numpy as np
import torch
from utils.environment import GridWorld
from utils.env_configs import env_configs
from methods.DQN_Single import DQN
import pandas as pd

train_data, validation_data, test_data = env_configs()

agent1 = DQN()
agent2 = DQN()
agent3 = DQN()
agent4 = DQN()

Epochs = 100

episode_mean_train = np.zeros((Epochs,1))
episode_std_train = np.zeros((Epochs,1))
episode_done_train = np.zeros((Epochs,1))
episode_steps_train = np.zeros((Epochs,1))
episode_steps_std_train = np.zeros((Epochs,1))

single_episode_train = np.zeros((len(train_data),1))
single_episode_train_done = np.zeros((len(train_data),1))
single_episode_steps_train = np.zeros((len(train_data),1))

episode_mean_test = np.zeros((Epochs,1))
episode_std_test = np.zeros((Epochs,1))
episode_done_test = np.zeros((Epochs,1))
episode_steps_test = np.zeros((Epochs,1))
episode_steps_std_test = np.zeros((Epochs,1))


single_episode_test = np.zeros((len(test_data),1))
single_episode_test_done = np.zeros((len(test_data),1))
single_episode_steps_test = np.zeros((len(test_data),1))


# grid size
L = 15
# goal size
L_G = 3
# size of grid in hash
L_L = np.ceil(L_G/2)
# number of grids in hash
L_W = np.ceil(L/L_L)
max_steps = 10000

nothing_reward = -6
tile_explore_reward = 6
goal_state_reward = 501


for e in range(Epochs):
    
    j = 0
    for config in train_data:

        env = GridWorld(config)

        i1 = env.state[0]
        i2 = env.state[1]
        i3 = env.state[2]
        i4 = env.state[3]

        row1 = i1 // 15
        col1 = i1 % 15

        row2 = i2 // 15
        col2 = i2 % 15

        row3 = i3 // 15
        col3 = i3 % 15

        row4 = i4 // 15
        col4 = i4 % 15

        state_torch1 = torch.zeros(64+2)
        state_torch1[0] = row1
        state_torch1[1] = col1
        next_state_torch1 = state_torch1

        state_torch2 = torch.zeros(64+2)
        state_torch2[0] = row2
        state_torch2[1] = col2
        next_state_torch2 = state_torch2

        state_torch3 = torch.zeros(64+2)
        state_torch3[0] = row3
        state_torch3[1] = col3
        next_state_torch3 = state_torch3

        state_torch4 = torch.zeros(64+2)
        state_torch4[0] = row4
        state_torch4[1] = col4
        next_state_torch4 = state_torch4

        steps = 0
        episodic_reward = 0
        while steps < max_steps:

            a1 = agent1.select_action(state_torch1)
            a2 = agent2.select_action(state_torch2)
            a3 = agent3.select_action(state_torch3)
            a4 = agent4.select_action(state_torch3)

            # input need to be torch tensor
            next_state, reward, done = env.step(np.array([a1,a2,a3,a4]))

            next_state_torch1[0] = next_state[0] // 15
            next_state_torch1[1] = next_state[0] % 15

            next_state_torch2[0] = next_state[1] // 15
            next_state_torch2[1] = next_state[1] % 15

            next_state_torch3[0] = next_state[2] // 15
            next_state_torch3[1] = next_state[2] % 15

            next_state_torch4[0] = next_state[3] // 15
            next_state_torch4[1] = next_state[3] % 15

            # agent 1 training
            # check what tile state is currently in
            tile_row = next_state_torch1[0] // L_L
            tile_column = next_state_torch1[1] // L_L
            tile_index = tile_row*L_W+tile_column

            # penalty for iteration
            reward = nothing_reward
            # information gain, rewars
            if next_state_torch1[int(tile_index)+2] == 0:
                reward += tile_explore_reward
                next_state_torch1[int(tile_index)+2] = 1
                next_state_torch2[int(tile_index)+2] = 1
                next_state_torch3[int(tile_index)+2] = 1
                next_state_torch4[int(tile_index)+2] = 1
            # Goal state reward
            if done == True:
                reward += goal_state_reward
            episodic_reward += reward

            agent1.train(state_torch1, a1, reward, next_state_torch1, done)

            # agent 2 training
            # check what tile state is currently in
            tile_row = next_state_torch2[0] // L_L
            tile_column = next_state_torch2[1] // L_L
            tile_index = tile_row*L_W+tile_column

            # penalty for iteration
            reward = nothing_reward
            # information gain, rewars
            if next_state_torch2[int(tile_index)+2] == 0:
                reward += tile_explore_reward
                next_state_torch1[int(tile_index)+2] = 1
                next_state_torch2[int(tile_index)+2] = 1
                next_state_torch3[int(tile_index)+2] = 1
                next_state_torch4[int(tile_index)+2] = 1
            # Goal state reward
            if done == True:
                reward += goal_state_reward
            episodic_reward += reward

            agent2.train(state_torch2, a2, reward, next_state_torch2, done)


            # agent 3 training
            # check what tile state is currently in
            tile_row = next_state_torch3[0] // L_L
            tile_column = next_state_torch3[1] // L_L
            tile_index = tile_row*L_W+tile_column

            # penalty for iteration
            reward = nothing_reward
            # information gain, rewars
            if next_state_torch3[int(tile_index)+2] == 0:
                reward += tile_explore_reward
                next_state_torch1[int(tile_index)+2] = 1
                next_state_torch2[int(tile_index)+2] = 1
                next_state_torch3[int(tile_index)+2] = 1
                next_state_torch4[int(tile_index)+2] = 1
            # Goal state reward
            if done == True:
                reward += goal_state_reward
            episodic_reward += reward

            agent3.train(state_torch3, a3, reward, next_state_torch3, done)

            # agent 4 training
            # check what tile state is currently in
            tile_row = next_state_torch4[0] // L_L
            tile_column = next_state_torch4[1] // L_L
            tile_index = tile_row*L_W+tile_column

            # penalty for iteration
            reward = nothing_reward
            # information gain, rewars
            if next_state_torch4[int(tile_index)+2] == 0:
                reward += tile_explore_reward
                next_state_torch1[int(tile_index)+2] = 1
                next_state_torch2[int(tile_index)+2] = 1
                next_state_torch3[int(tile_index)+2] = 1
                next_state_torch4[int(tile_index)+2] = 1
            # Goal state reward
            if done == True:
                reward += goal_state_reward
            episodic_reward += reward

            agent3.train(state_torch4, a4, reward, next_state_torch4, done)

            
            state_torch1 = next_state_torch1
            state_torch2 = next_state_torch2
            state_torch3 = next_state_torch3
            state_torch4 = next_state_torch4


            steps += 1

            if done:
                break

    # print(f' ')
    # print(f'Training Episode {e + 1}, Average Reward: {episodic_reward},  goal state found: {done}, average step: {steps}')


        single_episode_steps_train[j] = steps
        single_episode_train_done[j] = done
        single_episode_train[j] = episodic_reward
        j = j+1

    # agent.update_epsilon()
    agent1.target_update()
    agent2.target_update()
    agent3.target_update()
    agent4.target_update()

    episode_mean_train[e] = np.mean(single_episode_train)
    episode_std_train[e] = np.std(single_episode_train)
    episode_done_train[e] = np.mean(single_episode_train_done)
    episode_steps_train[e] = np.mean(single_episode_steps_train)
    episode_steps_std_train[e] = np.std(single_episode_steps_train)

    print(f' ')
    print(f'Training Episode {e + 1}, Average Reward: {episode_mean_train[e]}, STD: {episode_std_train[e]}, %goal state found: {episode_done_train[e]}, average step: {episode_steps_train[e]}, std step: {episode_steps_std_train[e]}')


    j = 0
    for config in test_data:

        env = GridWorld(config)

        i1 = env.state[0]
        i2 = env.state[0]
        i3 = env.state[0]
        i4 = env.state[0]

        row1 = i1 // 15
        col1 = i1 % 15

        row2 = i2 // 15
        col2 = i2 % 15

        row3 = i3 // 15
        col3 = i3 % 15

        row4 = i4 // 15
        col4 = i4 % 15

        state_torch1 = torch.zeros(64+2)
        state_torch1[0] = row1
        state_torch1[1] = col1
        next_state_torch1 = state_torch1

        state_torch2 = torch.zeros(64+2)
        state_torch2[0] = row2
        state_torch2[1] = col2
        next_state_torch2 = state_torch2

        state_torch3 = torch.zeros(64+2)
        state_torch3[0] = row3
        state_torch3[1] = col3
        next_state_torch3 = state_torch3

        state_torch4 = torch.zeros(64+2)
        state_torch4[0] = row4
        state_torch4[1] = col4
        next_state_torch4 = state_torch4

        steps = 0
        episodic_reward = 0
        while steps < max_steps:

            a1 = agent1.select_action(state_torch1)
            a2 = agent2.select_action(state_torch2)
            a3 = agent3.select_action(state_torch3)
            a4 = agent4.select_action(state_torch3)

            # input need to be torch tensor
            next_state, reward, done = env.step(np.array([a1,a2,a3,a4]))

            next_state_torch1[0] = next_state[0] // 15
            next_state_torch1[1] = next_state[0] % 15

            next_state_torch2[0] = next_state[1] // 15
            next_state_torch2[1] = next_state[1] % 15

            next_state_torch3[0] = next_state[2] // 15
            next_state_torch3[1] = next_state[2] % 15

            next_state_torch4[0] = next_state[3] // 15
            next_state_torch4[1] = next_state[3] % 15

            # agent 1 training
            # check what tile state is currently in
            tile_row = next_state_torch1[0] // L_L
            tile_column = next_state_torch1[1] // L_L
            tile_index = tile_row*L_W+tile_column

            # penalty for iteration
            reward = nothing_reward
            # information gain, rewars
            if next_state_torch1[int(tile_index)+2] == 0:
                reward += tile_explore_reward
                next_state_torch1[int(tile_index)+2] = 1
                next_state_torch2[int(tile_index)+2] = 1
                next_state_torch3[int(tile_index)+2] = 1
                next_state_torch4[int(tile_index)+2] = 1
            # Goal state reward
            if done == True:
                reward += goal_state_reward
            episodic_reward += reward

            # agent 2 training
            # check what tile state is currently in
            tile_row = next_state_torch2[0] // L_L
            tile_column = next_state_torch2[1] // L_L
            tile_index = tile_row*L_W+tile_column

            # penalty for iteration
            reward = nothing_reward
            # information gain, rewars
            if next_state_torch2[int(tile_index)+2] == 0:
                reward += tile_explore_reward
                next_state_torch1[int(tile_index)+2] = 1
                next_state_torch2[int(tile_index)+2] = 1
                next_state_torch3[int(tile_index)+2] = 1
                next_state_torch4[int(tile_index)+2] = 1
            # Goal state reward
            if done == True:
                reward += goal_state_reward
            episodic_reward += reward

            # agent 3 training
            # check what tile state is currently in
            tile_row = next_state_torch3[0] // L_L
            tile_column = next_state_torch3[1] // L_L
            tile_index = tile_row*L_W+tile_column

            # penalty for iteration
            reward = nothing_reward
            # information gain, rewars
            if next_state_torch3[int(tile_index)+2] == 0:
                reward += tile_explore_reward
                next_state_torch1[int(tile_index)+2] = 1
                next_state_torch2[int(tile_index)+2] = 1
                next_state_torch3[int(tile_index)+2] = 1
                next_state_torch4[int(tile_index)+2] = 1
            # Goal state reward
            if done == True:
                reward += goal_state_reward
            episodic_reward += reward

            # agent 4 training
            # check what tile state is currently in
            tile_row = next_state_torch4[0] // L_L
            tile_column = next_state_torch4[1] // L_L
            tile_index = tile_row*L_W+tile_column

            # penalty for iteration
            reward = nothing_reward
            # information gain, rewars
            if next_state_torch4[int(tile_index)+2] == 0:
                reward += tile_explore_reward
                next_state_torch1[int(tile_index)+2] = 1
                next_state_torch2[int(tile_index)+2] = 1
                next_state_torch3[int(tile_index)+2] = 1
                next_state_torch4[int(tile_index)+2] = 1
            # Goal state reward
            if done == True:
                reward += goal_state_reward
            episodic_reward += reward
            
            state_torch1 = next_state_torch1
            state_torch2 = next_state_torch2
            state_torch3 = next_state_torch3
            state_torch4 = next_state_torch4

            steps += 1

            if done:
                break

    # print(f' ')
    # print(f'Training Episode {e + 1}, Average Reward: {episodic_reward},  goal state found: {done}, average step: {steps}')


        single_episode_steps_test[j] = steps
        single_episode_test_done[j] = done
        single_episode_test[j] = episodic_reward
        j = j+1

    episode_mean_test[e] = np.mean(single_episode_test)
    episode_std_test[e] = np.std(single_episode_test)
    episode_done_test[e] = np.mean(single_episode_test_done)
    episode_steps_test[e] = np.mean(single_episode_steps_test)
    episode_steps_std_test[e] = np.std(single_episode_steps_test)

    print(f'Testing Episode {e + 1}, Average Reward: {episode_mean_test[e]}, STD: {episode_std_test[e]}, %goal state found: {episode_done_test[e]}, average step: {episode_steps_test[e]},  std step: {episode_steps_std_test[e]}')



# Data_np = np.concatenate((episode_mean_train,episode_std_train, episode_done_train, episode_mean_test, episode_std_test, episode_done_test), axis=1)

Data_np = np.concatenate((episode_mean_train, episode_std_train, episode_done_train, episode_steps_train, episode_steps_std_train, episode_mean_test, episode_std_test, episode_done_test, episode_steps_test, episode_steps_std_test), axis=1)


column_labels = ['Mean_Train', 'STD_Train', 'Finish Train', 'Train Steps', 'Train Steps STD', 'Mean_Test', 'STD_Test', 'Finish Test', 'Test Steps', 'Test Steps STD']

DF = pd.DataFrame(Data_np, columns=column_labels) 
DF.to_csv("results/DQN_for_MARL_15x15_more_four_agent_explore_2.csv", index=False)



