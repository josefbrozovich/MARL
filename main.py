import numpy as np
import torch
from utils.environment import GridWorld
from utils.env_configs import env_configs
from methods.DQN import DQN


train_data, validation_data, test_data = env_configs()

agent = DQN()


Epochs = 100

for e in range(Epochs):

    episodic_reward = 0
    for config in train_data:
        # make environment

        env = GridWorld(config)
        state, done = env.reset()

        steps = 0
        while steps < 1000 and not done:

            action = agent.select_action(state)

            a_0 = action // 64
            a_1 = (action % 64) // 16
            a_2 = (action % 16) // 4
            a_3 = action % 4
            # return np.array([a_0, a_1, a_2, a_3])

            next_state, reward, done = env.step(np.array([a_0, a_1, a_2, a_3]))
            episodic_reward += reward
            agent.train(state, action, reward, next_state, done)
            state = next_state

            steps += 1


    episodic_reward/len(train_data)

    print(f'Episode {e + 1}, Average Reward: {episodic_reward}')

