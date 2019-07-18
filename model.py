import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorNet(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=100, fc2_units=100, fc3_units=50):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(ActorNet, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.fc4 = nn.Linear(fc3_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.dropout(x,p=0.1)
        x = F.relu(self.fc2(x))
        x = F.dropout(x,p=0.1)
        x = F.relu(self.fc3(x))
        return F.tanh(self.fc4(x))   ## added tanh


class CriticNet(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=100, fc2_units=100, fc3_units=50):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(CriticNet, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size+action_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.fc4 = nn.Linear(fc3_units, 1)

    def forward(self, state,action):
        """Build a network that maps state and action -> one value"""
        x = torch.cat((state, action), dim=1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x,p=0.1)
        x = F.relu(self.fc2(x))
        x = F.dropout(x,p=0.1)
        x = F.relu(self.fc3(x))
        return self.fc4(x)