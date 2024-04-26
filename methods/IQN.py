import torch
import random
import numpy as np

"""
A simple Q-network class
"""
class QNetwork(torch.nn.Module):
    def __init__(self, input_dim=12, output_dim=4, hidden_dim_1=16, hidden_dim_2=16):
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

class IQN:
    def __init__(self, gamma=0.99, eps = 1.9):
        self.gamma = gamma
        self.eps = eps
        self.dqn = QNetwork()  # Q network
        self.dqn_target = QNetwork()  # Target Q network
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.loss_fn = torch.nn.MSELoss()  # loss function
        self.optim = torch.optim.Adam(self.dqn.parameters(), lr=5e-4)  # optimizer for training

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

        if done:
            self.update_epsilon()
            self.target_update()

        next_state_value = torch.max(self.dqn_target(s1))
        target_values = r+self.gamma*next_state_value*(1-done)

        state_action_value = self.dqn(s0)
        state_action_value = state_action_value[a0]

        loss = self.loss_fn(state_action_value,target_values)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        return


    def update_epsilon(self):
        # Decay epsilon
        if self.eps >= 0.01:
            self.eps *= 0.95

    def target_update(self):
        # Update the target Q network (self.dqn_target) using the original Q network (self.dqn)
        self.dqn_target.load_state_dict(self.dqn.state_dict())

