import torch
import random
import numpy as np

"""
A simple Q-network class
"""
class Qnetwork(torch.nn.Module):
    def __init__(self, input_dim=72, hidden_dim_1=4, hidden_dim_2=8, hidden_dim_3 = 4, output_dim=1):
        """
        Args:
            input_dim (int): state dimension.
            output_dim (int): number of actions.
            hidden_dim (int): hidden layer dimension (fully connected layer)
        """
        super().__init__()



        self.linearA1 = torch.nn.Linear(input_dim-6, hidden_dim_1)
        self.linearA2 = torch.nn.Linear(input_dim-6, hidden_dim_1)
        self.linearA3 = torch.nn.Linear(input_dim-6, hidden_dim_1)
        self.linearA4 = torch.nn.Linear(input_dim-6, hidden_dim_1)

        self.linearW1 = torch.nn.Linear(input_dim, hidden_dim_1 , bias=False)
        self.linearb1 = torch.nn.Linear(input_dim, hidden_dim_2)

        self.linearW2 = torch.nn.Linear(input_dim,  hidden_dim_2, bias=False)
        self.linearb2 = torch.nn.Linear(input_dim, output_dim)

    def forward(self, state):
        """
        Returns a Q value
        Args:
            state (torch.Tensor): state, 2-D tensor of shape (n, input_dim)
        Returns:
            torch.Tensor: Q values, 2-D tensor of shape (n, output_dim)
        """

        state1 = torch.cat((state[0:2],state[list(range(8,72))]))
        state2 = torch.cat((state[2:4],state[list(range(8,72))]))
        state3 = torch.cat((state[4:6],state[list(range(8,72))]))
        state4 = torch.cat((state[6:8],state[list(range(8,72))]))
        
        x1 = torch.nn.functional.relu(self.linearA1(state1))
        x2 = torch.nn.functional.relu(self.linearA2(state2))
        x3 = torch.nn.functional.relu(self.linearA3(state3))
        x4 = torch.nn.functional.relu(self.linearA4(state4))

        x1max = torch.max(x1)
        x2max = torch.max(x2)
        x3max = torch.max(x3)
        x4max = torch.max(x4)

        input_W1 = torch.tensor([x1max,x2max,x3max,x4max])

        W1 = self.linearW1(torch.abs(state))
        b1 = self.linearb1(state)

        W2 = self.linearW2(torch.abs(state))
        b2 = self.linearb2(torch.relu(state))

        input_W2 = torch.nn.functional.elu(torch.matmul(W1,input_W1)+b1)
        output = torch.matmul(W2,input_W2)+b2

        return output

    def get_actions(self, state):

        state1 = torch.cat((state[0:2],state[list(range(8,72))]))
        state2 = torch.cat((state[2:4],state[list(range(8,72))]))
        state3 = torch.cat((state[4:6],state[list(range(8,72))]))
        state4 = torch.cat((state[6:8],state[list(range(8,72))]))


        x1 = torch.nn.functional.relu(self.linearA1(state1))
        x2 = torch.nn.functional.relu(self.linearA2(state2))
        x3 = torch.nn.functional.relu(self.linearA3(state3))
        x4 = torch.nn.functional.relu(self.linearA4(state4))

        a1 = torch.argmax(x1)
        a2 = torch.argmax(x2)
        a3 = torch.argmax(x3)
        a4 = torch.argmax(x4)

        return a1, a2, a3, a4


class QMIX:
    def __init__(self, gamma=0.99, eps = 1.9):
        self.gamma = gamma
        self.eps = eps
        self.dqn = Qnetwork()  # Q network
        self.dqn_target = Qnetwork()  # Target Q network
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.loss_fn = torch.nn.MSELoss()  # loss function
        self.optim = torch.optim.Adam(self.dqn.parameters(), lr=5e-4)  # optimizer for training

    def get_actions(self, state):

        a1,a2,a3,a4 = self.dqn.get_actions(state)

        return a1, a2, a3, a4

    def forward(self, state):
        Qest = self.dqn.forward(state)
        return Qest

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


        state_action_value = self.dqn(s0)[0]

        # target_values = torch.tensor(target_values)

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

