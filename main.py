import numpy as np
import torch
from utils.environment import GridWorld
from utils.env_configs import env_configs
from methods.DQN import DQN
import pandas as pd 




train_data, validation_data, test_data = env_configs()

agent = DQN()

Epochs = 20

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
    for config in train_data:
        # make environment

        env = GridWorld(config)
        state, done = env.reset()

        steps = 0

        episodic_reward = 0

        while steps < 200 and not done:

            action = agent.select_action(state)

            a_0 = action // 64
            a_1 = (action % 64) // 16
            a_2 = (action % 16) // 4
            a_3 = action % 4

            next_state, reward, done = env.step(np.array([a_0, a_1, a_2, a_3]))
            episodic_reward += reward
            agent.train(state, action, reward, next_state, done)
            state = next_state

            steps += 1

        single_episode_train_done[j] = done
        single_episode_train[j] = episodic_reward
        j = j+1


    episode_mean_train[e] = np.mean(single_episode_train)
    episode_std_train[e] = np.std(single_episode_train)
    episode_done_train[e] = np.mean(single_episode_train_done)


    print(f' ')
    print(f'Training Episode {e + 1}, Average Reward: {episode_mean_train[e]}, STD: {episode_std_train[e]}, %finish: {episode_done_train[e]}')

    j = 0
    for config in test_data:

        # if e == Epochs-1 and j == len(test_data)-1:
        #     print("configuration", config)

        env = GridWorld(config)
        state, done = env.reset()

        steps = 0

        episodic_reward = 0

        while steps < 200 and not done:

            action = agent.select_action(state)

            a_0 = action // 64
            a_1 = (action % 64) // 16
            a_2 = (action % 16) // 4
            a_3 = action % 4

            next_state, reward, done = env.step(np.array([a_0, a_1, a_2, a_3]))
            episodic_reward += reward
            state = next_state

            steps += 1

            # if e == Epochs-1 and j == len(test_data)-1:
            #     print("step")
            #     for i in range(4):
            #         row = env.state[i] // 15
            #         column = env.state[i] % 15

            #         print("element", env.state[i], "row", row, "column", column)

        single_episode_test_done[j] = done
        single_episode_test[j] = episodic_reward
        j = j+1


    episode_mean_test[e] = np.mean(single_episode_test)
    episode_std_test[e] = np.std(single_episode_test)
    episode_done_test[e] = np.mean(single_episode_test_done)


    print(f'Testing Episode {e + 1}, Average Reward: {episode_mean_test[e]}, STD: {episode_std_test[e]}, %finish: {episode_done_test[e]}')

Data_np = np.concatenate((episode_mean_train,episode_std_train, episode_done_train, episode_mean_test, episode_std_test, episode_done_test), axis=1)

column_labels = ['Mean_Train', 'STD_Train', 'Finish Train', 'Mean_Test', 'STD_Test', 'Finish Test']

DF = pd.DataFrame(Data_np, columns=column_labels) 
DF.to_csv("results/DQN_for_MARL_15x15_more.csv", index=False)




