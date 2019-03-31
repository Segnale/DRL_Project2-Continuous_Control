import torch
import torch.nn as nn
import torch.nn.functional as F

class Policy(nn.Module):
    """Policy Model."""

    def __init__(self, state_size, action_size, seed = 0, fc1_units=200, fc2_units=200, fcC_units = 124):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in fsirst hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Policy, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fcA = nn.Linear(fc2_units, action_size)
        # self.Aout = nn.Tanh()

        # self.fcC1 = nn.Linear(state_size, fc1_units)
        self.fcC2 = nn.Linear(fc1_units, fcC_units)
        self.fcC = nn.Linear(fcC_units,1)

        self.std = nn.Parameter(torch.zeros(1, action_size))

    def forward(self, state, action = None):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc(state))
        a = F.relu(self.fc2(x))
        a = F.relu(self.fcA(a))
        # a = self.Aout(a)

        # v = F.relu(self.fcC1(state))
        v = F.relu(self.fcC2(x))
        v = self.fcC(v)
        
        a = torch.tanh(a)
        dist = torch.distributions.Normal(a, F.softplus(self.std))
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)

        return action, log_prob, dist.entropy(), v