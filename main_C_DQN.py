import numpy as np
import torch
from utils.environment import GridWorld
from utils.env_configs import env_configs
from methods.Central_DQN import DQN
import pandas as pd 

train_data, validation_data, test_data = env_configs()

agent = DQN()

Epochs = 200

episode_mean_train = np.zeros((Epochs,1))
episode_std_train = np.zeros((Epochs,1))
episode_done_train = np.zeros((Epochs,1))

single_episode_train = np.zeros((len(train_data),1))
single_episode_train_done = np.zeros((len(train_data),1))


episode_mean_test = np.zeros((Epochs,1))
episode_std_test = np.zeros((Epochs,1))
episode_done_test = np.zeros((Epochs,1))

single_episode_test = np.zeros((len(test_data),1))
single_episode_test_done = np.zeros((len(test_data),1))


for e in range(Epochs):
    
    j = 0
    # for config in train_data:
        # make environment
    if True:
        config = train_data[0,:]

        env = GridWorld(config)
        state, done = env.reset()

        i1, i2, i3, i4, r1, r2, r3, r4 = state

        row1 = i1 // 15
        col1 = i1 % 15

        row2 = i2 // 15
        col2 = i2 % 15

        row3 = i3 // 15
        col3 = i3 % 15

        row4 = i4 // 15
        col4 = i4 % 15

        state_torch = torch.tensor([row1,col1,r1,
                                    row2,col2,r2,
                                    row3,col3,r3,
                                    row4,col4,r4])

        next_state_torch = state_torch

        steps = 0
        episodic_reward = 0

        while steps < 200 and not done:

            # input need to be torch tensor
            action = agent.select_action(state_torch)

            a_0 = action // 64
            a_1 = (action % 64) // 16
            a_2 = (action % 16) // 4
            a_3 = action % 4

            # input need to be torch tensor
            next_state, reward, done = env.step(np.array([a_0, a_1, a_2, a_3]))
            episodic_reward += reward

            next_state_torch[0] = next_state[0] // 15
            next_state_torch[1] = next_state[0] % 15
            next_state_torch[2] = next_state[4]
            next_state_torch[3] = next_state[1] // 15
            next_state_torch[4] = next_state[1] % 15
            next_state_torch[5] = next_state[5]
            next_state_torch[6] = next_state[2] // 15
            next_state_torch[7] = next_state[2] % 15
            next_state_torch[8] = next_state[6]
            next_state_torch[9] = next_state[3] // 15
            next_state_torch[10] = next_state[3] % 15
            next_state_torch[11] = next_state[7]

            # input needs to be torch tensor
            agent.train(state_torch, action, reward, next_state_torch, done)
            state = next_state
            state_torch = next_state_torch

            steps += 1

        single_episode_train_done[j] = done
        single_episode_train[j] = episodic_reward
        j = j+1


    episode_mean_train[e] = np.mean(single_episode_train)
    episode_std_train[e] = np.std(single_episode_train)
    episode_done_train[e] = np.mean(single_episode_train_done)

    print(f' ')
    print(f'Training Episode {e + 1}, Average Reward: {episode_mean_train[e]}, STD: {episode_std_train[e]}, %finish: {episode_done_train[e]}')

#     j = 0
#     # for config in test_data:

#         # if e == Epochs-1 and j == len(test_data)-1:
#         #     print("configuration", config)

#         env = GridWorld(config)
#         state, done = env.reset()

#         steps = 0

#         episodic_reward = 0

#             # ind1, ind2, ind3, ind4, re1, re2, re3, re4 = env.state

#             # row1 = ind1//15
#             # col1 = ind1%15
#             # row2 = ind2//15
#             # col2 = ind2%15
#             # row3 = ind3//15
#             # col3 = ind3%15
#             # row4 = ind4//15
#             # col4 = ind4%15

#             # action = agent.select_action(torch.tensor([row1,col1,re1,row2,col2,re2,row3,col3,re3,row4,col4,re4]))


#         while steps < 2000 and not done:

# #            state = env.state

#             action = agent.select_action(state)

#             a_0 = action // 64
#             a_1 = (action % 64) // 16
#             a_2 = (action % 16) // 4
#             a_3 = action % 4

#             next_state, reward, done = env.step(np.array([a_0, a_1, a_2, a_3]))
#             episodic_reward += reward
#             state = next_state

#             steps += 1

#             # if e == Epochs-1 and j == len(test_data)-1:
#             #     print("step")
#             #     for i in range(4):
#             #         row = env.state[i] // 15
#             #         column = env.state[i] % 15

#             #         print("element", env.state[i], "row", row, "column", column)

#         single_episode_test_done[j] = done
#         single_episode_test[j] = episodic_reward
#         j = j+1


#     episode_mean_test[e] = np.mean(single_episode_test)
#     episode_std_test[e] = np.std(single_episode_test)
#     episode_done_test[e] = np.mean(single_episode_test_done)


#     print(f'Testing Episode {e + 1}, Average Reward: {episode_mean_test[e]}, STD: {episode_std_test[e]}, %finish: {episode_done_test[e]}')

Data_np = np.concatenate((episode_mean_train,episode_std_train, episode_done_train, episode_mean_test, episode_std_test, episode_done_test), axis=1)

column_labels = ['Mean_Train', 'STD_Train', 'Finish Train', 'Mean_Test', 'STD_Test', 'Finish Test']

DF = pd.DataFrame(Data_np, columns=column_labels) 
DF.to_csv("results/DQN_for_MARL_15x15_more_one_env_xy.csv", index=False)




