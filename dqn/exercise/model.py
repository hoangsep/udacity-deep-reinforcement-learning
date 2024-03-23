import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        "*** YOUR CODE HERE ***"
        # self.fc1 = nn.Linear(state_size, 32)
        # self.fc2 = nn.Linear(32, 64)
        # self.fc3 = nn.Linear(64, 128)
        # self.fc4 = nn.Linear(128, 128)
        # self.fc5 = nn.Linear(128, 128)
        # self.fc6 = nn.Linear(128, 64)
        # self.fc7 = nn.Linear(64, 32)
        # self.fc8 = nn.Linear(32, action_size)

        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        # x = F.relu(self.fc1(state))
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        # x = F.relu(self.fc4(x))
        # x = F.relu(self.fc5(x))
        # x = F.relu(self.fc6(x))
        # x = F.relu(self.fc7(x))
        # x = self.fc8(x)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
        # return x