import numpy as np
import torch
from utils.environment import GridWorld
from utils.env_configs import env_configs
from methods.DQN import DQN
import pandas as pd 




train_data, validation_data, test_data = env_configs()

agent = DQN()


Epochs = 100

episode_mean_train = np.zeros((Epochs,1))
episode_std_train = np.zeros((Epochs,1))
single_episode_train = np.zeros((len(train_data),1))

episode_mean_test = np.zeros((Epochs,1))
episode_std_test = np.zeros((Epochs,1))
single_episode_test = np.zeros((len(test_data),1))

for e in range(Epochs):
    
    j = 0
    for config in train_data:
        # make environment

        env = GridWorld(config)
        state, done = env.reset()

        steps = 0

        episodic_reward = 0

        while steps < 500 and not done:

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

        single_episode_train[j] = episodic_reward
        j = j+1


    episode_mean_train[e] = np.mean(single_episode_train)
    episode_std_train[e] = np.std(single_episode_train)

    print(f' ')
    print(f'Training Episode {e + 1}, Average Reward: {episode_mean_train[e]}, STD: {episode_std_train[e]}')

    j = 0
    for config in test_data:
        # make environment

        env = GridWorld(config)
        state, done = env.reset()

        steps = 0

        episodic_reward = 0

        while steps < 500 and not done:

            action = agent.select_action(state)

            a_0 = action // 64
            a_1 = (action % 64) // 16
            a_2 = (action % 16) // 4
            a_3 = action % 4

            next_state, reward, done = env.step(np.array([a_0, a_1, a_2, a_3]))
            episodic_reward += reward
            state = next_state

            steps += 1

        single_episode_test[j] = episodic_reward
        j = j+1


    episode_mean_test[e] = np.mean(single_episode_test)
    episode_std_test[e] = np.std(single_episode_test)

    print(f'Testing Episode {e + 1}, Average Reward: {episode_mean_test[e]}, STD: {episode_std_test[e]}')

Train_m_std_test_m_std = np.concatenate((episode_mean_train,episode_std_train, episode_mean_test, episode_std_test), axis=1)

column_labels = ['Mean_Train', 'STD_Train', 'Mean_Test', 'STD_Test']

DF = pd.DataFrame(Train_m_std_test_m_std, columns=column_labels) 
DF.to_csv("results/DQN_for_MARL.csv", index=False)




