import numpy as np
import torch
from utils.environment_single import GridWorld
from utils.env_configs import env_configs
from methods.DQN_Single import DQN
import pandas as pd

train_data, validation_data, test_data = env_configs()

agent = DQN()

Epochs = 50

episode_mean_train = np.zeros((Epochs,1))
episode_std_train = np.zeros((Epochs,1))
episode_done_train = np.zeros((Epochs,1))
episode_steps_train = np.zeros((Epochs,1))

single_episode_train = np.zeros((len(train_data),1))
single_episode_train_done = np.zeros((len(train_data),1))
single_episode_steps_train = np.zeros((len(train_data),1))
# single_episode_train = np.zeros((100,1))
# single_episode_train_done = np.zeros((100,1))
# single_episode_steps_train = np.zeros((100,1))

episode_mean_test = np.zeros((Epochs,1))
episode_std_test = np.zeros((Epochs,1))
episode_done_test = np.zeros((Epochs,1))
episode_steps_test = np.zeros((Epochs,1))

single_episode_test = np.zeros((len(test_data),1))
single_episode_test_done = np.zeros((len(test_data),1))
single_episode_steps_test = np.zeros((len(test_data),1))
# single_episode_test = np.zeros((100,1))
# single_episode_test_done = np.zeros((100,1))
# single_episode_steps_test = np.zeros((100,1))


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
#    if True:
    # for _ in range(100):

        # config = train_data[_,:]
        # config = train_data[e,:]

        env = GridWorld(config)

        i1 = env.state[0]

        row1 = i1 // 15
        col1 = i1 % 15

        # fix this later
        state_torch = torch.zeros(64+2)

        state_torch[0] = row1
        state_torch[1] = col1
        next_state_torch = state_torch

        steps = 0
        episodic_reward = 0
        while steps < max_steps:

            a = agent.select_action(state_torch)

            # input need to be torch tensor
            next_state, reward, done = env.step(np.array([a]))

            next_state_torch[0] = next_state[0] // 15
            next_state_torch[1] = next_state[0] % 15

            # check what tile state is currently in
            tile_row = next_state_torch[0] // L_L
            tile_column = next_state_torch[1] // L_L
            tile_index = tile_row*L_W+tile_column

            # penalty for iteration
            reward = nothing_reward
            # information gain, rewars
            if next_state_torch[int(tile_index)+2] == 0:
                reward += tile_explore_reward
                next_state_torch[int(tile_index)+2] = 1
            # Goal state reward
            if done == True:
                reward += goal_state_reward
            episodic_reward += reward

            agent.train(state_torch, a, reward, next_state_torch, done)
            state_torch = next_state_torch

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
    agent.target_update()

    episode_mean_train[e] = np.mean(single_episode_train)
    episode_std_train[e] = np.std(single_episode_train)
    episode_done_train[e] = np.mean(single_episode_train_done)
    episode_steps_train[e] = np.mean(single_episode_steps_train)

    print(f' ')
    print(f'Training Episode {e + 1}, Average Reward: {episode_mean_train[e]}, STD: {episode_std_train[e]}, %goal state found: {episode_done_train[e]}, average step: {episode_steps_train[e]}')


    j = 0
    for config in test_data:
    # if True:
    # for _ in range(100):

    #     # config = test_data[np.random.randint(len(test_data)),:]
    #     config = test_data[_,:]

        env = GridWorld(config)
        i1 = env.state[0]

        row1 = i1 // 15
        col1 = i1 % 15

        # fix this later
        state_torch = torch.zeros(64+2)

        state_torch[0] = row1
        state_torch[1] = col1
        next_state_torch = state_torch

        steps = 0
        episodic_reward = 0
        while steps < max_steps:

            a = agent.select_action(state_torch)

            # input need to be torch tensor
            next_state, reward, done = env.step(np.array([a]))

            next_state_torch[0] = next_state[0] // 15
            next_state_torch[1] = next_state[0] % 15

            # check what tile state is currently in
            tile_row = next_state_torch[0] // L_L
            tile_column = next_state_torch[1] // L_L
            tile_index = tile_row*L_W+tile_column

            # penalty for iteration
            reward = nothing_reward
            # information gain, rewars
            if next_state_torch[int(tile_index)+2] == 0:
                reward += tile_explore_reward
                next_state_torch[int(tile_index)+2] = 1
            # Goal state reward
            if done == True:
                reward += goal_state_reward

            episodic_reward += reward

            state_torch = next_state_torch

            steps += 1

            if done:
                break

    # print(f'Testing Episode {e + 1}, Average Reward: {episodic_reward},  goal state found: {done}, average step: {steps}')

        single_episode_steps_test[j] = steps
        single_episode_test_done[j] = done
        single_episode_test[j] = episodic_reward
        j = j+1

    episode_mean_test[e] = np.mean(single_episode_test)
    episode_std_test[e] = np.std(single_episode_test)
    episode_done_test[e] = np.mean(single_episode_test_done)
    episode_steps_test[e] = np.mean(single_episode_steps_test)

    print(f'Testing Episode {e + 1}, Average Reward: {episode_mean_test[e]}, STD: {episode_std_test[e]}, %goal state found: {episode_done_test[e]}, average step: {episode_steps_test[e]}')

# Data_np = np.concatenate((episode_mean_train,episode_std_train, episode_done_train, episode_mean_test, episode_std_test, episode_done_test), axis=1)

Data_np = np.concatenate((episode_mean_train,episode_std_train, episode_done_train, episode_steps_train, episode_mean_test, episode_std_test, episode_done_test, episode_steps_test), axis=1)


column_labels = ['Mean_Train', 'STD_Train', 'Finish Train', 'Train Steps', 'Mean_Test', 'STD_Test', 'Finish Test', 'Test Steps']

DF = pd.DataFrame(Data_np, columns=column_labels) 
DF.to_csv("results/DQN_for_MARL_15x15_more_one_agent_explore.csv", index=False)



