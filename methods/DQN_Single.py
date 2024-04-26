import torch
import random
import numpy as np
from collections import deque


"""
A simple Q-network class
"""
class QNetwork(torch.nn.Module):
    def __init__(self, input_dim=66, output_dim=4, hidden_dim_1=20, hidden_dim_2=15):
        """
        Args:
            input_dim (int): state dimension.
            output_dim (int): number of actions.
            hidden_dim (int): hidden layer dimension (fully connected layer)
        """
        super().__init__()
        self.linear1 = torch.nn.Linear(input_dim, hidden_dim_1)
        self.linear2 = torch.nn.Linear(hidden_dim_1, hidden_dim_2)
        self.linear3 = torch.nn.Linear(hidden_dim_2, output_dim)

    def forward(self, state):
        """
        Returns a Q value
        Args:
            state (torch.Tensor): state, 2-D tensor of shape (n, input_dim)
        Returns:
            torch.Tensor: Q values, 2-D tensor of shape (n, output_dim)
        """

        x1 = torch.nn.functional.relu(self.linear1(state))
        x2 = torch.nn.functional.relu(self.linear2(x1))
        x3 = self.linear3(x2)

        return x3

class DQN:
    def __init__(self, gamma=0.99, eps = 1.9):
        self.gamma = gamma
        self.eps = eps
        self.dqn = QNetwork()  # Q network
        self.dqn_target = QNetwork()  # Target Q network
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.loss_fn = torch.nn.MSELoss()  # loss function
        self.optim = torch.optim.Adam(self.dqn.parameters(), lr=5e-4)  # optimizer for training
        self.replay_memory_buffer = deque(maxlen=10000)  # replay buffer

        self.batch_size = 100


    def select_action(self, state):

        if np.random.rand() > (1-self.eps):
            action = np.random.randint(4)
            return action
        else:
            Q_hat = self.dqn(state)
            action = torch.argmax(Q_hat)
            return action

    def train(self, s0, a0, r, s1, done):
        """
        Train the Q network
        Args:
            s0: current state, a numpy array with size 4
            a0: current action, 0 or 1
            r: reward
            s1: next state, a numpy array with size 4
            done: done=True means that the episode terminates and done=False means that the episode does not terminate.
        """

        self.add_to_replay_memory(s0, a0, r, s1, done)

        if len(self.replay_memory_buffer) < self.batch_size:
            return
        
        """
        state_batch: torch.Tensor with shape (self.batch_size, 4), a mini-batch of current states
        action_batch: torch.Tensor with shape (self.batch_size, 1), a mini-batch of current actions
        reward_batch: torch.Tensor with shape (self.batch_size, 1), a mini-batch of rewards
        next_state_batch: torch.Tensor with shape (self.batch_size, 4), a mini-batch of next states
        done_list: torch.Tensor with shape (self.batch_size, 1), a mini-batch of 0-1 integers,
                   where 1 means the episode terminates for that sample;
                         0 means the episode does not terminate for that sample.
        """
        mini_batch = self.get_random_sample_from_replay_mem()
        state_batch = torch.from_numpy(np.vstack([i[0] for i in mini_batch])).float()
        action_batch = torch.from_numpy(np.vstack([i[1] for i in mini_batch])).int()
        reward_batch = torch.from_numpy(np.vstack([i[2] for i in mini_batch])).float()
        next_state_batch = torch.from_numpy(np.vstack([i[3] for i in mini_batch])).float()
        done_list = torch.from_numpy(np.vstack([i[4] for i in mini_batch]).astype(np.uint8)).float()

        done_list = done_list.reshape(-1,1)

        next_state_values = self.dqn_target(next_state_batch).max(1)[0]
        next_state_values = next_state_values.reshape(-1,1)
        target_values = reward_batch+self.gamma*next_state_values*(torch.ones((len(done_list),1))-done_list)

        state_action_values = self.dqn(state_batch)
        action_batch = action_batch.long()
        state_action_values = state_action_values[torch.arange(state_action_values.size(0)), action_batch.squeeze()]
        state_action_values = state_action_values.reshape(-1,1)

        loss = self.loss_fn(state_action_values,target_values)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
       
        return

    def add_to_replay_memory(self, state, action, reward, next_state, done):
        """
        Add samples to replay memory
        Args:
            state: current state, a numpy array with size 4
            action: current action, 0 or 1
            reward: reward
            next_state: next state, a numpy array with size 4
            done: done=True means that the episode terminates and done=False means that the episode does not terminate.
        """
        self.replay_memory_buffer.append((state, action, reward, next_state, done))

    def get_random_sample_from_replay_mem(self):
        """
        Random samples from replay memory without replacement
        Returns a self.batch_size length list of unique elements chosen from the replay buffer.
        Returns:
            random_sample: a list with len=self.batch_size,
                           where each element is a tuple (state, action, reward, next_state, done)
        """
        random_sample = random.sample(self.replay_memory_buffer, 1)
        return random_sample


    def update_epsilon(self):
        # Decay epsilon
        if self.eps >= 0.01:
            self.eps *= 0.95

    def target_update(self):
        # Update the target Q network (self.dqn_target) using the original Q network (self.dqn)
        self.dqn_target.load_state_dict(self.dqn.state_dict())

